# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import ast
import os
import os.path as osp
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Tuple, Union
from PIL import Image, ImageFile
from packaging import version
import numpy as np
from glob import glob
from tqdm import tqdm

import time
import random
import torch.distributed
import yaml
import math
import re
import torch
import torch.nn.functional as F

import transformers
import tokenizers
import deepspeed

from transformers import AutoConfig, PreTrainedModel
from torch.utils.data import Dataset

from javisgpt.constants import *
from javisgpt.train.llava_trainer import LLaVATrainer
from javisgpt.train.ckpt_utils import *
from javisgpt.model import JavisConfig, JavisGPTForConditionalGeneration
from javisgpt.model.qwen2_vl import Qwen2VLProcessor, Qwen2VLImageProcessor
from javisgpt.model.blocks import smart_pad
from javisgpt.utils import rank0_print, rank_print
from javisgpt.mm_utils import process_video, process_audio, process_image

import sys; sys.path.append('./javisdit')
from interface.javisdit_interface import Preprocessor

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

# import warnings
# warnings.filterwarnings('ignore')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Qwen2-VL"})
    version: Optional[str] = field(default="v0")
    beats_path: str = field(default=None)
    audio_projector_path: str = field(default=None)

    avsync_mode: str = field(default='merge')
    avsync_projector_path: str = field(default=None)

    avgen_cfg_path: str = field(default=None)
    avgen_projector_path: str = field(default=None)

    tokenizer_path: Optional[str] = field(default=None)

    all_projector_path: str = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=16)
    frame_width: Optional[int] = field(default=588)  # 588 or 448 TODO: dynamic size
    frame_height: Optional[int] = field(default=336)  # TODO: dynamic size
    frame_stride: Optional[int] = field(default=2)  # 2 for Qwen2-VL
    force_sample: Optional[bool] = field(default=True)

    audio_folder: Optional[str] = field(default=None)
    audio_sr: Optional[int] = field(default=16000)
    max_audio_length_s: Optional[float] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    training_stage: str = "audio_align"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    calc_dummy_loss: bool = field(default=False)
    verbose_logging: bool = field(default=False)
    disable_tqdm: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict[str, List[str]],
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    force_resize: bool = False,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer_vocab_size = len(tokenizer)
    model_vocab_size = model.get_input_embeddings().weight.data.shape[0]

    if num_new_tokens > 0 and (tokenizer_vocab_size > model_vocab_size or force_resize):
        rank0_print(f'model vocabulary size increased from {model_vocab_size} to {tokenizer_vocab_size}')
        model.resize_token_embeddings(tokenizer_vocab_size)

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    return num_new_tokens


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_qwen(
    sources, tokenizer: transformers.PreTrainedTokenizer, max_len=2048, system_message: str = "You are a helpful assistant.", 
    has_image: bool = False, has_video: bool = False, has_audio: bool = False, has_audio_video: bool = False, 
    image_grid_thw=None, video_grid_thw=None, audio_grid_thw=None, audio_video_grid_thw=None, merge_size=2,
    avsync_mode: str='merge', gen_audio_video: bool=False,
    is_training=True
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        image_grid_thw = image_grid_thw.prod() // merge_size**2
    if has_audio:
        tokenizer.add_tokens([DEFAULT_AUDIO_TOKEN], special_tokens=True)
        audio_grid_thw = audio_grid_thw.prod()
    if has_video:
        tokenizer.add_tokens([DEFAULT_VIDEO_TOKEN], special_tokens=True)
        video_grid_thw = video_grid_thw.prod() // merge_size**2
    if has_audio_video:
        tokenizer.add_tokens([DEFAULT_AUDIO_VIDEO_TOKEN], special_tokens=True)
        # TODO: we treat audio_video as a new modality with the same tokens as video, this is highlighted at several places
        assert audio_video_grid_thw.ndim == 1
        assert avsync_mode == 'merge'
        audio_video_grid_thw = audio_video_grid_thw[:3].prod() // merge_size**2
    if gen_audio_video:
        tokenizer.add_tokens([DEFAULT_GEN_AUDIO_VIDEO_TOKEN], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    audio_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_TOKEN)
    video_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_VIDEO_TOKEN)
    audio_video_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_VIDEO_TOKEN)
    gen_audio_video_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_GEN_AUDIO_VIDEO_TOKEN)
    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    # unmask_tokens = ["<|im_start|>", "<|im_end|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]  # TODO: add special tokens to unmask_tokens
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        assert roles[source[0]["from"]] == roles["human"], 'conversation must start from humen'
        # if roles[source[0]["from"]] != roles["human"]:
        #     source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]
            
            if has_image:
                replace_token = DEFAULT_VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN * image_grid_thw + DEFAULT_VISION_END_TOKEN
                content = content.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            
            if has_video:
                replace_token = DEFAULT_VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN * video_grid_thw + DEFAULT_VISION_END_TOKEN
                content = content.replace(DEFAULT_VIDEO_TOKEN, replace_token)
            
            if has_audio:
                replace_token = DEFAULT_AUDIO_START_TOKEN + DEFAULT_AUDIO_TOKEN * audio_grid_thw + DEFAULT_AUDIO_END_TOKEN
                content = content.replace(DEFAULT_AUDIO_TOKEN, replace_token)
            
            if has_audio_video:
                replace_token = DEFAULT_AUDIO_VIDEO_START_TOKEN + DEFAULT_AUDIO_VIDEO_TOKEN * audio_video_grid_thw + DEFAULT_AUDIO_VIDEO_END_TOKEN
                content = content.replace(DEFAULT_AUDIO_VIDEO_TOKEN, replace_token)
            
            if gen_audio_video:
                replace_token = DEFAULT_AUDIO_VIDEO_START_TOKEN + DEFAULT_GEN_AUDIO_VIDEO_TOKEN * AV_GEN_TOKEN_NUM + DEFAULT_AUDIO_VIDEO_END_TOKEN
                content = content.replace(DEFAULT_GEN_AUDIO_VIDEO_TOKEN, replace_token)

            role =  roles.get(role, role)
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
          
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            # require gradient on special tokens even in inputs
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            # reset special token for multi-modalities
            # NOTE: multi-modal data maybe occur in GPT's response in multi-turn conversation
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
                target[idx] = IGNORE_INDEX
            if encode_id == audio_token_index:
                input_id[idx] = AUDIO_TOKEN_INDEX
                target[idx] = IGNORE_INDEX
            if encode_id == video_token_index:
                input_id[idx] = VIDEO_TOKEN_INDEX
                target[idx] = IGNORE_INDEX
            if encode_id == audio_video_token_index:
                input_id[idx] = AUDIO_VIDEO_TOKEN_INDEX
                target[idx] = IGNORE_INDEX
            if encode_id == gen_audio_video_token_index:
                input_id[idx] = GEN_AUDIO_VIDEO_TOKEN_INDEX
                target[idx] = IGNORE_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    if not is_training:
        ## skip the end tag `<|im_end|>\n` for the last 2 tokens
        return input_ids[:, :-2]

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = []
        # Handle multiple JSON files specified in the data_path

        data_args.dataset_paths = [data_path]
        rank0_print(f"Loading {data_path}")
        with open(data_path, "r") as file:
            cur_data_dict = json.load(file)
            rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
            self.list_data_dict.extend(cur_data_dict)

        # debug only      
        # self.list_data_dict = self.list_data_dict
        # rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

        # JavisDiT preprocessor
        if self.data_args.avgen_cfg_path:
            self.avgen_processor = Preprocessor(
                cfg_path=self.data_args.avgen_cfg_path,
                video_folder=self.data_args.video_folder,
                audio_folder=self.data_args.audio_folder,
            )

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # return self._get_item(self.list_data_dict[i])
    
        num_base_retries = 3

        # try the current sample first
        for attempt_idx in range(1):
            try:
                sample = self._get_item(self.list_data_dict[i])
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e, self.list_data_dict[i])
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(self.list_data_dict[next_index])
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e,
                      self.list_data_dict[next_index])
                pass

        try:
            sample = self._get_item(self.list_data_dict[i])
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources: Union[List[dict], dict]) -> Dict[str, torch.Tensor]:
        single_sample = isinstance(sources, dict)
        if single_sample:
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        image_grid_thw, video_grid_thw, audio_grid_thw, audio_video_grid_thw = None, None, None, None
        second_per_grid_ts, second_per_audio_grid_ts, second_per_audio_video_grid_ts = None, None, None
        has_image, has_audio, has_video, has_audio_video, gen_audio_video = False, False, False, False, False

        # TODO: currently support multi-image but single-video/audio with fixed size

        max_length_s = self.data_args.max_audio_length_s
        
        if "gen_audio_video" in sources[0]:
            avgen_meta = sources[0]["gen_audio_video"]
            # TODO: align with `self.data_args.video_folder` and `self.data_args.audio_folder`
            target_texts, target_audios, target_videos = self.avgen_processor.load_data(avgen_meta)
            gen_audio_video = True

            avgen_task_type = avgen_meta.get('task_type', 'T2AV')
            if avgen_task_type in ['A2V', 'AI2V', 'V2A']:
                max_length_s = self.avgen_processor.duration
            elif 'Exten' in avgen_task_type:
                max_length_s = sources[0].pop('end_s')

        if "image" in sources[0]:
            image_file = sources[0]["image"]
            image_folder = self.data_args.image_folder
            if type(image_file) is list:
                # raise NotImplementedError('unsupported for multiple images with different sizes')
                image_data = [process_image(osp.join(f, image_file), self.data_args) for f in image_file]
            else:
                image_file = osp.join(image_folder, image_file)
                image_data = [process_image(image_file, self.data_args)]
            pixel_values_list, image_grid_thw_list = list(zip(*image_data))
            pixel_values = torch.cat(pixel_values_list)  # shape(M, 1176), compatible with dynamic resolution
            image_grid_thw = torch.stack(image_grid_thw_list)  # shape(B, 3)
            # TODO: what's this? move to the end of multi-modal input preparation
            # sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            has_image = True

        if "audio" in sources[0] and "video" not in sources[0]:
            audio_file = sources[0]["audio"]
            audio_folder = self.data_args.audio_folder
            audio_file = osp.join(audio_folder, audio_file)
            _, audio_fbank, audio_grid_thw, second_per_audio_grid_ts = process_audio(
                audio_file, 
                max_audio_length_s=max_length_s, 
                audio_sr=self.data_args.audio_sr
            )
            has_audio = True
        
        if "audio" in sources[0] and "video" in sources[0]:  # audio_video mode
            video_file = sources[0]["video"]
            video_folder = self.data_args.video_folder
            video_file = osp.join(video_folder, video_file)
            # TODO: we treat audio_video as a new modality with the same tokens as video, this is highlighted at several places
            pixel_values_videos, video_grid_thw, second_per_audio_video_grid_ts, duration = \
                process_video(video_file, self.data_args, return_duration=True, max_length_s=max_length_s)

            audio_file = sources[0]["audio"]
            audio_folder = self.data_args.audio_folder
            audio_file = osp.join(audio_folder, audio_file)
            # assume audio have nearly the same duration with video
            _, audio_fbank, audio_grid_thw, *remains = process_audio(
                audio_file, audio_length_s=duration, 
                audio_sr=self.data_args.audio_sr,
            )

            audio_video_grid_thw = torch.cat((video_grid_thw, audio_grid_thw), 0)  # shape(6,)
            video_grid_thw, audio_grid_thw = None, None

            has_audio_video = True

        if "video" in sources[0]:
            video_file = sources[0]["video"]
            video_folder = self.data_args.video_folder
            video_file = osp.join(video_folder, video_file)
            pixel_values_videos, video_grid_thw, second_per_grid_ts = \
                process_video(video_file, self.data_args, max_length_s=max_length_s)
            # sources[0]["conversations"][0]["value"] = f'{DEFAULT_VIDEO_TOKEN}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_VIDEO_TOKEN, "")}'
            has_video = True
            
        conversations = copy.deepcopy([e["conversations"] for e in sources])

        merge_size = self.data_args.image_processor.merge_size
        data_dict = preprocess_qwen(
            conversations, self.tokenizer, 
            has_image=has_image, has_audio=has_audio, has_video=has_video, has_audio_video=has_audio_video, gen_audio_video=gen_audio_video,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, audio_grid_thw=audio_grid_thw, audio_video_grid_thw=audio_video_grid_thw, 
            merge_size=merge_size, avsync_mode=self.data_args.avsync_mode
        )
        prompt = data_dict.get("prompt", None)

        if single_sample:
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # independent filling for modalities?
        if has_image:
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_grid_thw
        
        if has_video:
            data_dict["pixel_values_videos"] = pixel_values_videos
            data_dict["video_grid_thw"] = video_grid_thw
            data_dict["second_per_grid_ts"] = second_per_grid_ts
        
        if has_audio:
            data_dict["audio_fbank"] = audio_fbank
            data_dict["audio_grid_thw"] = audio_grid_thw
            data_dict["second_per_audio_grid_ts"] = second_per_audio_grid_ts
        
        if has_audio_video:
            data_dict["audio_fbank"] = audio_fbank
            data_dict["pixel_values_videos"] = pixel_values_videos
            data_dict["audio_video_grid_thw"] = audio_video_grid_thw
            data_dict["second_per_audio_video_grid_ts"] = second_per_audio_video_grid_ts
        
        if gen_audio_video:
            data_dict["target_texts"] = target_texts
            data_dict["target_videos"] = target_videos
            data_dict["target_audios"] = target_audios

        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = sources[0].get("id", None)

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    require_data_id: bool = False

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def pad_temporal(self, data, padding_value=0.0):
        # shape[0] must be temporal dimension
        max_len = max([x.shape[0] for x in data])
        z, pad_masks = [], torch.full((len(data), max_len), 0, dtype=torch.bool)
        for i, x in enumerate(data):
            pad_len = max_len-x.shape[0]
            if pad_len > 0:
                x = smart_pad(x, pad_len, dim=0, mode='constant', value=padding_value)
                pad_masks[i, -pad_len:] = 1
            z.append(x)
        
        return torch.stack(z), pad_masks

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # # NOTE: check corrupt files
        # return {}
        # TODO: temporal wrapper for batch feature dataset
        if isinstance(instances, list) and len(instances) == 1 and isinstance(instances[0], list):
            instances = instances[0]
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

        mm_cfg = {
            ## NOTE: Process audio_video first. DO NOT CHANGE.
            # TODO: we assume each sample contains one and only one audio-video pair
            "audio_video_grid_thw": ["audio_fbank", "pixel_values_videos", "audio_video_grid_thw", "second_per_audio_video_grid_ts"], # , "audio_pad_mask"
            # TODO: deal with variant image numbers across batch
            "image_grid_thw": ["pixel_values", "image_grid_thw"],
            # TODO: we assume each sample contains one and only one video
            # TODO: we assume each video have the same frame numbers
            "video_grid_thw": ["pixel_values_videos", "video_grid_thw", "second_per_grid_ts"],
            # TODO: we assume each sample contains one and only one audio
            "audio_grid_thw": ["audio_fbank", "audio_grid_thw", "second_per_audio_grid_ts"],
            ## generation targets
            # TODO: we assume each target video-audio pairs have the same fps and resolution
            "target_texts": ["target_texts", "target_videos", "target_audios"],
        }
        mm_dict = {}
        for kw, mm_list in mm_cfg.items():
            for instance in instances:
                if kw in instance:
                    for m in mm_list:
                        mm_dict[m] = mm_dict.get(m, []) + [instance[m]]
        for m, data in mm_dict.items():
            data = [x for x in data if x is not None]
            if len(data) == 0:
                continue
            if m == "audio_fbank":
                batch["audio_fbank"], batch["audio_pad_mask"] = self.pad_temporal(data, padding_value=0)
            elif m in ["pixel_values", "image_grid_thw"]:
                batch[m] = torch.cat(data)
            elif m == "target_texts":
                if isinstance(data[0], str):
                    batch["target_texts"] = data  # List[str]
                else:
                    batch["target_texts"] = {  # Dict[str, Tensor]
                        k: torch.cat([x[k] for x in data], 0)
                        for k in data[0].keys()
                    }
            else:
                batch[m] = torch.stack(data)

        if self.require_data_id:
            batch["data_id"] = [instance["id"] for instance in instances]

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_model_and_tokenizer(
    model_args, training_args, require_prev_vocab_size=False, 
    tokenizer_only=False, maybe_merge_lora=False, is_training=True,
) -> Union[Tuple[JavisGPTForConditionalGeneration, transformers.PreTrainedTokenizer], Tuple[JavisGPTForConditionalGeneration, transformers.PreTrainedTokenizer, int]]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_path or model_args.model_name_or_path,   # TODO: check compatibility
        cache_dir=training_args.cache_dir, 
        model_max_length=training_args.model_max_length, 
        padding_side="right" if is_training else "left"  # left for batch-infer
    )
    prev_vocab_size = len(tokenizer)
    if tokenizer_only:
        return tokenizer

    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    beats_ckpt = torch.load(model_args.beats_path, map_location='cpu')
    beats_cfg = beats_ckpt['cfg']
    setattr(cfg_pretrained, 'beats_cfg', beats_cfg)

    setattr(cfg_pretrained, 'avsync_mode', model_args.avsync_mode)
    setattr(cfg_pretrained, 'avgen_cfg_path', model_args.avgen_cfg_path)
    if is_training:
        setattr(cfg_pretrained, 'calc_dummy_loss', training_args.calc_dummy_loss)  # multi-gpu balancing
    
    model = JavisGPTForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=False,
        config=cfg_pretrained
    )
    if hasattr(model, 'beats'):
        model.beats.load_state_dict(beats_ckpt['model'])
        # freeze beats
        for name, param in model.beats.named_parameters():
            param.requires_grad = False
        model.beats.eval()
    if hasattr(model, 'av_generator'):
        model.init_avgen_token()
        model.av_generator.build_models()
        # freeze av_generator
        for name, param in model.av_generator.named_parameters():
            param.requires_grad = False
        model.av_generator.eval()
    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

    special_tokens_dict = {
        'audio_start': DEFAULT_AUDIO_START_TOKEN,
        'audio_end': DEFAULT_AUDIO_END_TOKEN,
        'audio_pad': DEFAULT_AUDIO_PAD_TOKEN,
        'audio_video_start': DEFAULT_AUDIO_VIDEO_START_TOKEN,
        'audio_video_end': DEFAULT_AUDIO_VIDEO_END_TOKEN,
        'audio_video_pad': DEFAULT_AUDIO_VIDEO_PAD_TOKEN,
        ## add new modality tokens. TODO: compatible with <|x_pad|>
        # 'audio': DEFAULT_AUDIO_TOKEN or DEFAULT_AUDIO_PAD_TOKEN,
        # 'audio_video': DEFAULT_AUDIO_VIDEO_TOKEN or DEFAULT_AUDIO_VIDEO_PAD_TOKEN,
    }
    num_new_tokens = smart_tokenizer_and_embedding_resize(
        {"additional_special_tokens": list(special_tokens_dict.values())}, 
        tokenizer, model
    )
    assert len(tokenizer) - prev_vocab_size == num_new_tokens
    rank0_print(f'{num_new_tokens} new tokens are added to tokenizer')
    for k, v in special_tokens_dict.items():
        special_token_id = tokenizer.convert_tokens_to_ids(v)
        setattr(model.config, f'{k}_token_id', special_token_id)

    proj_path_cfg = {  # MUST in this order
        "Audio": model_args.audio_projector_path,
        "AVSync": model_args.avsync_projector_path,
        "AVGen": model_args.avgen_projector_path,
        "All": model_args.all_projector_path,
    }
    for m_proj, proj_ckpt_path in proj_path_cfg.items():
        if not proj_ckpt_path or not osp.exists(proj_ckpt_path):
            continue
        if not os.path.exists(proj_ckpt_path):
            rank0_print(f"WARNING: Failed to load {m_proj} projector from empty path {proj_ckpt_path}")
        rank0_print(f"Loading {m_proj} projector...")
        projector_ckpt = torch.load(proj_ckpt_path, map_location='cpu', weights_only=True)
        projector_ckpt = {(k[17:] if k.startswith("base_model.model.") else k): v for k, v in projector_ckpt.items()}
        # projector_ckpt = {(k[11:] if k.startswith("base_model.") else k): v for k, v in projector_ckpt.items()}
        # if any(k.startswith("model.model.") for k in projector_ckpt):
        #     projector_ckpt = {(k[6:] if k.startswith("model.") else k): v for k, v in projector_ckpt.items()}
        model_ckpt = model.state_dict()
        used_keys, unused_keys = [], []
        for k, v in projector_ckpt.items():
            if k in model_ckpt:
                if model_ckpt[k].shape == v.shape:
                    used_keys.append(k)
                else:
                    rank0_print(f'Shape mismatch! {k}: ckpt {projector_ckpt[k].shape} v.s. model {model_ckpt[k].shape}')
            else:
                unused_keys.append(k)
        model.load_state_dict({k: projector_ckpt[k] for k in used_keys}, strict=False)
        rank0_print(f'{used_keys} have been successfully loaded.')
        if len(unused_keys):
            rank0_print(f'\n########## WARNING ##########\nunused keys: {unused_keys}\n########## WARNING ##########\n')

    if osp.exists(training_args.lora_weight_path):
        from peft import PeftModel
        rank0_print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, training_args.lora_weight_path, torch_device="cpu")
        if maybe_merge_lora:
            rank0_print("Merging LoRA weights...")
            model = model.merge_and_unload()

    if not is_training:
        model.eval()
        model.model.eval()

    ret = (model, tokenizer, )

    if require_prev_vocab_size:
        ret += (prev_vocab_size, )

    return ret


def train(attn_implementation=None):
    global local_rank
    # torch.multiprocessing.set_start_method('spawn')  # taken from NExT-GPT

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    setattr(data_args, 'avsync_mode', model_args.avsync_mode)
    setattr(data_args, 'avgen_cfg_path', model_args.avgen_cfg_path)

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        # rank0_print(f"evaluation_args = {vars(evaluation_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    model, tokenizer, prev_vocab_size = \
        get_model_and_tokenizer(model_args, training_args, require_prev_vocab_size=True)
    model.config.use_cache = False
    # TODO: we freeze the backbone llm by default
    model.requires_grad_(False)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(model_args.model_name_or_path)
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        if isinstance(model, PreTrainedModel):
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
        else:
            for name, param in model.named_parameters():
                if '.lora_' in name:
                    param.requires_grad_(True)

    def freeze_prev_tokens_grad(grad):
        grad[:prev_vocab_size] = 0  # 将旧词表部分的梯度置为 0
        return grad

    if training_args.training_stage == 'audio_align':
        model.get_audio_projector().requires_grad_(True)
        model.get_input_embeddings().requires_grad_(True)
        model.get_input_embeddings().weight.register_hook(freeze_prev_tokens_grad)
    elif training_args.training_stage == 'audio_video_align':
        model.get_avsync_projector().requires_grad_(True)
        model.get_input_embeddings().requires_grad_(True)
        model.get_input_embeddings().weight.register_hook(freeze_prev_tokens_grad)
    elif training_args.training_stage == 'audio_video_gen_align':
        model.get_avgen_query().requires_grad_(True)
        model.get_avgen_cond_projector().requires_grad_(True)
        # TODO: disable output gradient for avgen_align with LoRA
        # model.get_output_embeddings().requires_grad_(True)
        # model.get_output_embeddings().weight.register_hook(freeze_prev_tokens_grad)
    else: # training_args.training_stage == 'audio_video_instruct':
        if training_args.training_stage != 'av_finetune':  # mm_pretrain or mm_insttune
            model.get_audio_projector().requires_grad_(True)
        if training_args.training_stage != 'mm_pretrain':  # av_finetune or mm_insttune
            model.get_avsync_projector().requires_grad_(True)
        if getattr(model, "avgen_token", None) is not None:
            model.get_avgen_query().requires_grad_(True)
            model.get_avgen_cond_projector().requires_grad_(True)
        model.get_input_embeddings().requires_grad_(True)
        model.get_input_embeddings().weight.register_hook(freeze_prev_tokens_grad)
        model.get_output_embeddings().requires_grad_(True)
        model.get_output_embeddings().weight.register_hook(freeze_prev_tokens_grad)

    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params_list = [p[0] for p in model.named_parameters() if p[1].requires_grad]

    rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB")
    rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB")
    # rank0_print(f"Trainable parameter list: {trainable_params_list}")

    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    trainer.train()
    model.config.use_cache = True

    if training_args.lora_enable:
        # save adapters alone
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, adapter_only=True)
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank in [0, -1]:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            # torch.save(non_lora_state_dict, osp.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
