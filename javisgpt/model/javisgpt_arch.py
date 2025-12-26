#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import math
import re
import time
import random

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss, LayerNorm
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.processing_utils import ProcessorMixin
from transformers.utils import is_torchdynamo_compiling

from javisgpt.constants import *
from javisgpt.model.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel, Qwen2VLModel, Qwen2VLPreTrainedModel, Qwen2VLCausalLMOutputWithPast
from javisgpt.model.qwen2_vl import Qwen2VLProcessor, Qwen2VLImageProcessor, Qwen2VLConfig
from javisgpt.model.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLModel, Qwen2_5_VLPreTrainedModel, Qwen2_5_VLCausalLMOutputWithPast
from javisgpt.model.qwen2_5_vl import Qwen2_5_VLProcessor, Qwen2_5_VLConfig
from javisgpt.model.beats.BEATs import BEATsConfig, BEATs
from javisgpt.utils import rank0_print, rank_print, calc_zero_loss
from javisgpt.constants import *
from javisgpt.model.blocks import AVSync, MLP, smart_pad


if BASE_ARCH == "Qwen2VL":
    CFG_CLS = Qwen2VLConfig
    PROCESSOR_CLS = Qwen2VLProcessor
    PRETRAINED_MODEL_CLS = Qwen2VLPreTrainedModel
    MODEL_CLS = Qwen2VLModel
    VISUAL_CLS = Qwen2VisionTransformerPretrainedModel
    OUTPUT_CLS = Qwen2VLCausalLMOutputWithPast
elif BASE_ARCH == "Qwen2_5_VL":
    CFG_CLS = Qwen2_5_VLConfig
    PROCESSOR_CLS = Qwen2_5_VLProcessor
    PRETRAINED_MODEL_CLS = Qwen2_5_VLPreTrainedModel
    MODEL_CLS = Qwen2_5_VLModel
    VISUAL_CLS = Qwen2_5_VisionTransformerPretrainedModel
    OUTPUT_CLS = Qwen2_5_VLCausalLMOutputWithPast
else:
    raise ValueError(f"Unrecognized base model architecture, {BASE_ARCH}")


class JavisConfig(CFG_CLS):
    model_type = "javisgpt"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beats_cfg = kwargs.get("beats_cfg", {})
        self.avsync_mode = kwargs.get("avsync_mode", "merge")
        self.avgen_cfg_path = kwargs.get("avgen_cfg_path", None)

        self.calc_dummy_loss = kwargs.get("calc_dummy_loss", False)


class JavisProcessor(PROCESSOR_CLS):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template, **kwargs)
    
    def reset_image_processor(self, model_name_or_path):
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)


class JavisGPTForConditionalGeneration(PRETRAINED_MODEL_CLS, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = JavisConfig  # HACK: override config_class and model_class at config.json

    def __init__(self, config: JavisConfig):
        super().__init__(config)
        # TODO: avoid manual conversion
        if not isinstance(self.config, JavisConfig):
            self.config = JavisConfig.from_dict(config.to_dict())
            self.config.architectures = ["JavisGPTForConditionalGeneration"]
        self.visual = VISUAL_CLS._from_config(config.vision_config)

        self.beats_cfg = BEATsConfig(config.beats_cfg)
        self.beats_cfg.encoder_layerdrop = -1.0 # [modified] deepspeed layerdrop会卡住
        self.beats = BEATs(self.beats_cfg)

        self.audio_mlp = nn.Sequential(
            nn.Linear(self.beats_cfg.encoder_embed_dim, config.hidden_size*4),
            nn.GELU(approximate='tanh'),
            nn.Linear(config.hidden_size*4, config.hidden_size),
        )
        self.avsync_proj = AVSync(
            config.hidden_size, 
            config.num_attention_heads,
            mode=config.avsync_mode
        )

        if self.config.avgen_cfg_path:
            from interface.javisdit_interface import JavisDiTInterface

            self.av_generator = JavisDiTInterface(config.avgen_cfg_path)
            assert self.av_generator.cond_embed_num == AV_GEN_TOKEN_NUM, \
                f'{self.av_generator.cond_embed_num} {AV_GEN_TOKEN_NUM}'
            self.avgen_token = nn.Parameter(torch.rand((1, AV_GEN_TOKEN_NUM, config.hidden_size)) * 1e-3)
            self.avgen_cond_proj = self.av_generator.get_cond_projector(config.hidden_size)

        self.model = MODEL_CLS(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def init_avgen_token(self, method='vocab_cluster'):
        if not hasattr(self, 'avgen_token'):
            return

        if method == 'normal':
            nn.init.normal_(self.avgen_token.data, mean=0.0, std=0.02)
        elif method == 'vocab_cluster':
            input_embeddings_weight = self.get_input_embeddings().weight.detach()
            vocab_size = input_embeddings_weight.shape[0]
            cluster_size = vocab_size // AV_GEN_TOKEN_NUM
            num_embeddings_to_use = AV_GEN_TOKEN_NUM * cluster_size
            indices = torch.randperm(vocab_size)[:num_embeddings_to_use]
            embeddings_subset = input_embeddings_weight[indices]
            reshaped_embeddings = embeddings_subset.view(1, AV_GEN_TOKEN_NUM, cluster_size, -1)
            mean_embeddings = reshaped_embeddings.mean(dim=-2)
            self.avgen_token.data.copy_(mean_embeddings)
        elif method == 'vocab_tail_repeat':
            input_embeddings_weight = self.get_input_embeddings().weight.detach()
            input_embeddings_init = input_embeddings_weight[-1].view(1, 1, -1)
            input_embeddings_init = input_embeddings_init.repeat(1, AV_GEN_TOKEN_NUM, 1)
            self.avgen_token.data.copy_(input_embeddings_init)

        # TODO: maybe unused
        self.avgen_token.data[self.avgen_token.data.isnan()] = 0.0

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_audio_projector(self):
        return self.audio_mlp

    def get_avsync_projector(self):
        return self.avsync_proj

    def get_avgen_query(self):
        return self.avgen_token

    def get_avgen_cond_projector(self):
        return self.avgen_cond_proj

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        audio_grid_thw: Optional[torch.LongTensor] = None,
        audio_video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.LongTensor] = None,  # compatible with Qwen2.5-VL
        second_per_audio_grid_ts: Optional[torch.LongTensor] = None,
        second_per_audio_video_grid_ts: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        token_id_dict = {  # we treat audio_video as a new modality
            "image": IMAGE_TOKEN_INDEX, "audio": AUDIO_TOKEN_INDEX,
            "video": VIDEO_TOKEN_INDEX, "audio_video": AUDIO_VIDEO_TOKEN_INDEX
        }
        start_token_id_dict = {
            "vision": self.config.vision_start_token_id,
            "audio": self.config.audio_start_token_id,
            "audio_video": self.config.audio_video_start_token_id,
        }
        grid_thw_dict = {
            "image": image_grid_thw, "audio": audio_grid_thw,
            "video": video_grid_thw, "audio_video": audio_video_grid_thw
        }
        second_per_grid_ts_dict = {
            "image": None if image_grid_thw is None else torch.zeros((len(image_grid_thw), )), 
            "audio": second_per_audio_grid_ts,
            "video": second_per_grid_ts, 
            "audio_video": second_per_audio_video_grid_ts
        }
        mrope_position_deltas = []
        is_mm = any(g is not None for g in [image_grid_thw, video_grid_thw, audio_grid_thw, audio_video_grid_thw])
        if input_ids is not None and is_mm:
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            mm_index_dict = {modality: 0 for modality in token_id_dict.keys()}
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]

                mm_num_dict = {modality: 0 for modality in token_id_dict.keys()}
                for modality, start_token_id in start_token_id_dict.items():
                    mm_start_indices = torch.argwhere(input_ids == start_token_id).squeeze(1)
                    mm_tokens = input_ids[mm_start_indices + 1]
                    # image and audio tokens are enclosed with the same <|vision_start|> and <|vision_end|> tokens
                    if modality == 'vision':
                        mm_num_dict["image"] = (mm_tokens == token_id_dict["image"]).sum().item()
                        mm_num_dict["video"] = (mm_tokens == token_id_dict["video"]).sum().item()
                    else:
                        # mm_num_dict[modality] = len(mm_tokens)
                        mm_num_dict[modality] = (mm_tokens == token_id_dict[modality]).sum().item()
                
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                mm_remain_num_dict = {modality: num for modality, num in mm_num_dict.items()}
                for _ in range(sum(mm_num_dict.values())):
                    ed_mm_dict = {}
                    for modality, mm_remain_num in mm_remain_num_dict.items():
                        mm_token_id = token_id_dict[modality]
                        if mm_token_id in input_tokens and mm_remain_num > 0:
                            ed_mm_dict[modality] = input_tokens.index(mm_token_id, st)
                        else:
                            ed_mm_dict[modality] = len(input_tokens) + 1
                    # find the modality with smallest ed_index
                    ed_mm = min(ed_mm_dict, key=ed_mm_dict.get)
                    t, h, w = (
                        grid_thw_dict[ed_mm][mm_index_dict[ed_mm]][0],
                        grid_thw_dict[ed_mm][mm_index_dict[ed_mm]][1],
                        grid_thw_dict[ed_mm][mm_index_dict[ed_mm]][2],
                    )
                    if second_per_grid_ts_dict[ed_mm] is not None:
                        second_per_grid_t = second_per_grid_ts_dict[ed_mm][mm_index_dict[ed_mm]].item()
                    else:
                        second_per_grid_t = 1.0
                    mm_index_dict[ed_mm] += 1
                    mm_remain_num_dict[ed_mm] -= 1
                    ed = ed_mm_dict[ed_mm]

                    # audio has no spatial merging
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // (spatial_merge_size if ed_mm != 'audio' else 1),
                        w.item() // (spatial_merge_size if ed_mm != 'audio' else 1),
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    if BASE_ARCH == "Qwen2VL":
                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    elif BASE_ARCH == "Qwen2_5_VL":
                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
                        t_index = t_index * second_per_grid_t * self.config.vision_config.tokens_per_second
                        t_index = t_index.long().flatten()
                    else:
                        raise NotImplementedError(f"Unsupported MRoPE for architecture {BASE_ARCH}")

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            assert all(mm_remain_num == 0 for mm_remain_num in mm_remain_num_dict.values()), f'\n{mm_remain_num_dict}'
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas
    
    def prepare_inputs_labels_for_multimodal():
        pass

    @torch.compiler.disable
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,  # NOTE: always None here?
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,  # NOTE: always True here?
        ## MRoPE
        position_ids: Optional[torch.LongTensor] = None,  # NOTE: always None here?
        rope_deltas: Optional[torch.LongTensor] = None,  # NOTE: always None here?
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        ## Output
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,  # NOTE: always None here?
        output_hidden_states: Optional[bool] = None,
        ## MultiModality
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        audio_fbank: Optional[torch.Tensor] = None,
        audio_pad_mask: Optional[torch.Tensor] = None,
        video_pad_mask: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        audio_grid_thw: Optional[torch.LongTensor] = None,
        audio_video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.LongTensor] = None,
        second_per_audio_grid_ts: Optional[torch.LongTensor] = None,
        second_per_audio_video_grid_ts: Optional[torch.LongTensor] = None,
        target_texts: Optional[List[str]] = None,
        target_videos: Optional[torch.Tensor] = None,
        target_audios: Optional[torch.Tensor] = None,
        num_items_in_batch=None,  # TODO: unknown bug for transformers.Trainer
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        zero_pad_loss = 0.
        if inputs_embeds is None:
            # special tokens are set to -200/-300/-400/-500/-600 for image/audio/video/audio_video/av_gen
            # HACK: move X_TOKEN_INDEX back to self.config.x_token_id
            inputs_embeds = self.model.embed_tokens(torch.where(input_ids < 0, torch.tensor(0, dtype=input_ids.dtype), input_ids))

            ## NOTE: Process audio_video first. DO NOT CHANGE.
            if audio_video_grid_thw is not None:  # all audio-video pairs
                # shape(B, 6) -> [Tv, Hv, Wv, Ta, Ma, 1]
                assert (audio_video_grid_thw[:1, :3] == audio_video_grid_thw[:, :3]).all(), \
                    f'Currently does not support dynamic resolution or audio-video frame numbers, but support dynamic duration'
                # TODO: we treat audio_video as a new modality with the same tokens as video, this is highlighted at several places
                B, Tv, Ta = audio_video_grid_thw.shape[0], audio_video_grid_thw[0, 0].item(), audio_video_grid_thw[:, 3].max().item()
                if audio_grid_thw is not None:
                    Ta = max(Ta, audio_grid_thw[:, 0].max().item())
                # shape(B*T*H*W,C1)
                # split from single-modal video
                m_pixel_values_videos, pixel_values_videos = pixel_values_videos[:B].type(self.visual.dtype), pixel_values_videos[B:]
                if video_pad_mask is not None:
                    m_video_pad_mask, video_pad_mask = video_pad_mask[:B], video_pad_mask[B:]
                else:
                    m_video_pad_mask = video_pad_mask
                # shape(B*T*H/m*W/m,m*m*C2), m=2
                video_embeds = self.visual(m_pixel_values_videos, grid_thw=audio_video_grid_thw[:, :3])
                # shape(B, T, S, C)
                video_embeds = video_embeds.view(B, Tv, -1, video_embeds.shape[-1])

                # split from single-modal audio
                m_audio_fbank, audio_fbank = audio_fbank[:B], audio_fbank[B:]
                if audio_pad_mask is not None:
                    m_audio_pad_mask, audio_pad_mask = audio_pad_mask[:B], audio_pad_mask[B:]
                else:
                    m_audio_pad_mask = audio_pad_mask
                audio_embeds, m_audio_pad_mask = self.beats.extract_features(
                    fbank=m_audio_fbank, 
                    padding_mask=m_audio_pad_mask,
                    feature_only=True
                ) # [B, T*S, 768], [B, T*S] or None
                audio_embeds = audio_embeds.type(self.visual.dtype)
                audio_embeds = self.audio_mlp(audio_embeds)
                audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                # shape [B, T, S, C]
                audio_embeds = audio_embeds.view(B, Ta, -1, audio_embeds.shape[-1])
                # shape [B, T*S] -> shape[B, T]
                if m_audio_pad_mask is not None:
                    m_audio_pad_mask = m_audio_pad_mask.view(B, Ta, -1)
                
                # shape(B,T,S,C) for embeds, shape(B,T) for masks  --> shape(N, C)
                audio_video_embeds = self.avsync_proj(
                    audio_embeds, video_embeds, m_audio_pad_mask, m_video_pad_mask
                )
                n_audio_video_tokens = (input_ids == AUDIO_VIDEO_TOKEN_INDEX).sum().item()
                n_audio_video_features = audio_video_embeds.shape[0]
                if n_audio_video_tokens != n_audio_video_features:
                    raise ValueError(
                        "Audio_video features and audio_video tokens do not match: "
                        f"tokens: {n_audio_video_tokens}, features {n_audio_video_features}"
                    )
                    n_delta = n_audio_video_tokens - n_audio_video_features
                    if n_delta > 0:
                        audio_video_embeds = smart_pad(audio_video_embeds, n_delta, dim=0)
                    else:
                        audio_video_embeds = audio_video_embeds[:n_delta]

                audio_video_mask = (
                    (input_ids == AUDIO_VIDEO_TOKEN_INDEX)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                audio_video_embeds = audio_video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(audio_video_mask, audio_video_embeds)
            elif self.config.calc_dummy_loss:
                zero_pad_loss += calc_zero_loss(self.avsync_proj)
                
            if image_grid_thw is not None:  # all images
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == IMAGE_TOKEN_INDEX)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if video_grid_thw is not None:  # all videos
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == VIDEO_TOKEN_INDEX).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == VIDEO_TOKEN_INDEX)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if audio_grid_thw is not None:  # all audios
                audio_embeds, audio_pad_mask = self.beats.extract_features(
                    fbank=audio_fbank, 
                    padding_mask=audio_pad_mask,
                    feature_only=True
                ) # [B, L, 768]
                if audio_pad_mask is not None:
                    audio_embeds = audio_embeds[~audio_pad_mask]  # [N, 768]
                else:
                    audio_embeds = audio_embeds.flatten(0,1)  # [N, 768]
                audio_embeds = audio_embeds.type(self.visual.dtype)
                audio_embeds = self.audio_mlp(audio_embeds)

                audio_mask = (
                    (input_ids == AUDIO_TOKEN_INDEX)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                n_audio_tokens = (input_ids == AUDIO_TOKEN_INDEX).sum().item()
                n_audio_features = audio_embeds.shape[0]
                if n_audio_tokens != n_audio_features:
                    raise ValueError(
                        f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
                    )
                audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
            elif self.config.calc_dummy_loss:
                zero_pad_loss += calc_zero_loss(self.audio_mlp)

            if (input_ids == GEN_AUDIO_VIDEO_TOKEN_INDEX).any():
                n_avgen_tokens = (input_ids == GEN_AUDIO_VIDEO_TOKEN_INDEX).sum().item()
                assert n_avgen_tokens % AV_GEN_TOKEN_NUM == 0
                n_avgen_sample = n_avgen_tokens // AV_GEN_TOKEN_NUM
                av_gen_embeds = self.avgen_token.expand(n_avgen_sample, -1, -1)
                n_avgen_features = av_gen_embeds.shape[0] * av_gen_embeds.shape[1]
                if n_avgen_tokens != n_avgen_features:
                    raise ValueError(
                        "Audio_video generation features and audio_video tokens do not match: "
                        f"tokens: {n_avgen_tokens}, features {n_avgen_features}"
                    )
                av_gen_mask = (
                    (input_ids == GEN_AUDIO_VIDEO_TOKEN_INDEX)  # TODO: 1-bit shift
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                av_gen_embeds = av_gen_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(av_gen_mask, av_gen_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, 
                    image_grid_thw, video_grid_thw, audio_grid_thw, audio_video_grid_thw, 
                    second_per_grid_ts, second_per_audio_grid_ts, second_per_audio_video_grid_ts,
                    attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            ntp_loss = loss.detach().clone()
        
        if any(flag is not None for flag in [target_texts, target_audios, target_videos]):
            avgen_embeds = self.parse_avgen_embed(hidden_states, input_ids, GEN_AUDIO_VIDEO_TOKEN_INDEX)
            assert avgen_embeds.shape[0] == len(target_texts)
            avgen_conds: Union[torch.Tensor, List[torch.Tensor]] = self.avgen_cond_proj(avgen_embeds)
            if target_texts is not None:
                caption_loss = self.av_generator.compute_caption_loss(avgen_conds, target_texts)
                loss += caption_loss
                rank0_print(f'ntp loss: {ntp_loss.item()}; caption loss: {caption_loss.item()};', end=" ")
            if (target_videos is not None or target_audios is not None):
                assert target_videos is not None and target_audios is not None
                diffusion_loss = self.av_generator.compute_diffusion_loss(avgen_conds, target_audios, target_videos)
                loss += diffusion_loss
                rank0_print(f'diffusion loss: {diffusion_loss.item()}')
            else:
                rank0_print("")
        elif self.config.calc_dummy_loss and getattr(self, 'av_generator', None) is not None:
            zero_pad_loss += 0. * self.avgen_token.view(-1).sum()
            zero_pad_loss += calc_zero_loss(self.avgen_cond_proj)

        if loss is not None:
            loss += zero_pad_loss  # HACK: compatible for imbalanced batching

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # FIXME: temporary solution. move X_TOKEN_INDEX back to self.config.x_token_id
        input_ids[input_ids < 0] = 0  ## 0=`!`

        return OUTPUT_CLS(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        audio_fbank=None,
        audio_pad_mask=None,
        video_pad_mask=None,
        image_grid_thw=None,
        video_grid_thw=None,
        audio_grid_thw=None,
        audio_video_grid_thw=None,
        second_per_grid_ts=None,
        second_per_audio_grid_ts=None,
        second_per_audio_video_grid_ts=None,
        target_texts=None,
        target_videos=None,
        target_audios=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        # Exception 4: If input_embeds are passed then slice it through `cache_position`, to keep only the unprocessed tokens and
        # generate the first token for each sequence. Later use the generated Input ids for continuation.
        if past_key_values is not None:
            if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            elif (
                inputs_embeds is not None  # Exception 1
                or (is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1])  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values, pixel_values_videos, audio_fbank = None, None, None
            image_grid_thw, video_grid_thw = None, None
            audio_grid_thw, audio_video_grid_thw = None, None
            target_texts, target_videos, target_audios = None, None, None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "audio_fbank": audio_fbank,
                "audio_pad_mask": audio_pad_mask,
                "video_pad_mask": video_pad_mask,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "audio_grid_thw": audio_grid_thw,
                "audio_video_grid_thw": audio_video_grid_thw,
                "target_texts": target_texts,
                "target_videos": target_videos,
                "target_audios": target_audios,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
                "second_per_audio_grid_ts": second_per_audio_grid_ts,
                "second_per_audio_video_grid_ts": second_per_audio_video_grid_ts,
            }
        )
        return model_inputs

    def parse_avgen_embed(self, hidden_states, input_ids, avgen_id=GEN_AUDIO_VIDEO_TOKEN_INDEX):
        shift_input_ids = torch.cat((input_ids[:, 1:], input_ids[:, :1]), dim=1)
        avgen_embeds = hidden_states[shift_input_ids == avgen_id]
        avgen_embeds = avgen_embeds.view(-1, AV_GEN_TOKEN_NUM, self.config.hidden_size)
        return avgen_embeds

    @torch.no_grad()
    def generate_audio_video_infer(self, attention_mask, past_key_values, embed_only=False):
        batch_size, cache_length = attention_mask.shape
        device = attention_mask.device
        assert batch_size == 1
        assert cache_length == past_key_values[0][0].shape[-2] + 1
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        attention_mask_pad = torch.ones((batch_size, AV_GEN_TOKEN_NUM), device=device)
        attention_mask = torch.cat((attention_mask, attention_mask_pad), dim=1)
        # compatible with `preprocess_qwen` logic
        input_ids = torch.cat((
            torch.full((batch_size, 1), self.config.audio_video_start_token_id),
            torch.full((batch_size, AV_GEN_TOKEN_NUM), GEN_AUDIO_VIDEO_TOKEN_INDEX)
        ), dim=1).to(device)
        input_ids_bak = input_ids.clone()
        cache_position = torch.arange(cache_length, cache_length+AV_GEN_TOKEN_NUM+1).to(device)
        
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=True,
            output_hidden_states=True,
        )

        avgen_embeds = self.parse_avgen_embed(outputs.hidden_states[-1], input_ids_bak, GEN_AUDIO_VIDEO_TOKEN_INDEX)
        avgen_conds: Union[torch.Tensor, List[torch.Tensor]] = self.avgen_cond_proj(avgen_embeds)

        if embed_only:
            return avgen_conds
        else:
            audio, video = self.av_generator.sample(avgen_conds)
            return audio, video

    @torch.no_grad()
    def generate_audio_video_direct(self, avgen_embeds: Union[torch.Tensor, str], embed_only=False):
        if isinstance(avgen_embeds, torch.Tensor):
            avgen_conds: Union[torch.Tensor, List[torch.Tensor]] = self.avgen_cond_proj(avgen_embeds)
        else:
            avgen_conds, _ = self.av_generator.encode_text(avgen_embeds)

        if embed_only:
            return avgen_conds
        
        audio, video = self.av_generator.sample(avgen_conds)

        return audio, video
