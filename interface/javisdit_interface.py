import os
import pdb
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings('ignore')
from typing import Literal, Union, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from javisdit.registry import MODELS, SCHEDULERS, build_module
from javisdit.utils.config_utils import read_config
from javisdit.utils.train_utils import VAMaskGenerator
from javisdit.utils.misc import requires_grad

from javisdit.models.prior_encoder.ImageBind import data as imagebind_data
from javisdit.datasets.read_audio import read_audio
from javisdit.datasets.utils import (
    VID_EXTENSIONS, AUD_EXTENSIONS,
    get_transforms_image, get_transforms_video, get_transforms_audio, save_sample
)
from javisdit.datasets.datasets import load_video_audio_transform
from javisdit.datasets.aspect import get_image_size, get_num_frames
from javisdit.utils.inference_utils import (
    apply_va_mask_strategy,
    collect_va_references_batch,
    prepare_multi_resolution_info,
)

from javisgpt.utils import rank0_print


class Preprocessor(object):
    def __init__(self, cfg_path, video_folder='./', audio_folder='./'):
        cfg = read_config(cfg_path)
        self.version = cfg.get('version', 'v1.0')
        self.video_folder = video_folder
        self.audio_folder = audio_folder

        self.num_frames = cfg.get('num_frames', 81)
        self.image_size = cfg.get('image_size', (480, 864))  # H, W
        self.video_fps = cfg.get('video_fps', 16)
        self.frame_interval = cfg.get('frame_interval', 1)
        assert self.frame_interval == 1, "TODO: support frame_interval > 1"
        # TODO: currently support fixed duration
        self.duration = self.num_frames / self.video_fps
        self.video_transform = get_transforms_video(
            name="resize_crop", image_size=self.image_size
        )

        self.audio_fps = cfg.get('sampling_rate', 16000)
        self.audio_transform = get_transforms_audio(
            name="mel_spec_audioldm2", cfg=cfg.get("audio_cfg")
        )

        self.load_av_feat = cfg.get('load_av_feat', False)
        self.direct_load_video_clip = cfg.get('direct_load_video_clip', True)
        self.pre_tokenize = cfg.get('pre_tokenize', False)
        if self.pre_tokenize:
            self.jav_ctx_maxlen = cfg.get('text_encoder_model_max_length', 512)
            if self.version == 'v0.1':
                from transformers import AutoTokenizer
                self.jav_tokenizer = AutoTokenizer.from_pretrained(
                    cfg.text_encoder['from_pretrained'],
                )
            elif self.version == 'v1.0':
                from javisdit.models.wan.modules.tokenizers import HuggingfaceTokenizer
                tokenizer_path = os.path.join(
                    cfg.text_encoder['from_pretrained'], cfg.text_encoder['t5_tokenizer']
                )
                self.jav_tokenizer = HuggingfaceTokenizer(
                    name=tokenizer_path, seq_len=self.jav_ctx_maxlen, clean='whitespace'
                )
            else:
                raise NotImplementedError(f'Unknown version: {self.version}')
    
    def load_data(self, meta_info: dict):
        target_text = meta_info['target_text']
        target_audio = meta_info.get('target_audio', None)
        target_video = meta_info.get('target_video', None)

        if self.pre_tokenize:
            if self.version == 'v0.1':
                target_text = self.jav_tokenizer(
                    meta_info['target_text'],
                    max_length=self.jav_ctx_maxlen,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )  # 'input_ids', 'attention_mask'
                target_text['ib_text'] = imagebind_data.load_and_transform_text(
                    [meta_info['target_text']], 'cpu'
                )
            elif self.version == 'v1.0':
                ids, mask = self.jav_tokenizer(
                    meta_info['target_text'], 
                    return_mask=True, 
                    add_special_tokens=True
                )
                target_text = {'ids': ids, 'mask': mask}
            else:
                raise ValueError(f'Unknown version: {self.version}')

        task_type = meta_info.get('task_type', 'T2AV')
        fix_start_frame = None
        if task_type != 'T2AV':
            fix_start_frame = 0
        if 'Exten' in task_type:
            fix_start_frame = int(self.video_fps * meta_info['start_s'])

        if target_video is not None:
            if isinstance(target_video, torch.Tensor):
                pass
            elif isinstance(target_video, str):
                target_video = f'{self.video_folder}/{target_video}'
                target_audio = f'{self.audio_folder}/{target_audio}'
                assert os.path.exists(target_video)
                if not os.path.exists(target_audio):
                    target_audio = target_video
                if self.load_av_feat:
                    target_audio, target_video = torch.load(target_audio), torch.load(target_video)
                else:
                    target_audio, target_video = self.load_transform_audio_video(
                        target_audio, target_video, fix_start_frame=fix_start_frame
                    )
            else:
                raise ValueError(f"Unsupported target_video type: {type(target_video)}")
                
        # TODO: AV-Temporal Masks for X-Conditional Generation
        return target_text, target_audio, target_video

    def load_transform_audio_video(self, audio_path, video_path, fix_start_frame=None):
        aframes, ainfo = read_audio(audio_path, sr=self.audio_fps)
        assert ainfo['audio_fps'] == self.audio_fps
        
        _, video, audio, _, _, _ = load_video_audio_transform(
            video_path, self.direct_load_video_clip, self.num_frames, self.frame_interval, 
            audio_path, self.audio_fps, aframes, None, 
            self.video_transform, self.audio_transform, None, False,
            fix_start_frame=fix_start_frame
        )
        video = video.permute(1, 0, 2, 3)

        return audio, video


class HierarchicalCondMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, caption_out_dim, prior_out_dim=None,
                 caption_token_num=None, prior_token_num=None):
        super().__init__()
        self.caption_token_num = caption_token_num
        self.prior_token_num = prior_token_num
        self.caption_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, caption_out_dim)
        )
        if prior_out_dim is not None:
            self.prior_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), 
                nn.GELU(approximate='tanh'),
                nn.Linear(hidden_dim, prior_out_dim)
            )
        else:
            self.prior_net = None

    def forward(self, x):
        if self.prior_net is None:
            return self.caption_net(x)
        else:
            caption_x, prior_x = torch.split(
                x, 
                [self.caption_token_num, self.prior_token_num], 
                dim=1
            )
            caption_emb = self.caption_net(caption_x)
            prior_emb = self.prior_net(prior_x)
            return caption_emb, prior_emb


class JavisDiTInterface(nn.Module):
    def __init__(self, cfg_path, num_frames=81, image_size=(512, 512)):
        super().__init__()
        cfg = read_config(cfg_path)
        self.cfg = cfg
        self.version = cfg.get('version', 'v1.0')
        self.video_fps = cfg.get('video_fps', 16)
        self.audio_fps = cfg.get('sampling_rate', 16000)
        self.load_av_feat = cfg.get('load_av_feat', False)

        self.num_frames = num_frames
        self.image_size = image_size  # (H, W)

        self.caption_embed_dim = self.cfg.get("text_encoder_output_dim", 4096)
        self.caption_embed_num = self.cfg.get("text_encoder_model_max_length", 512)
    
    def build_models(self):
        # == build text-encoder ==
        text_encoder = build_module(self.cfg.get("text_encoder", None), MODELS)
        if text_encoder is not None:
            text_encoder_output_dim = text_encoder.output_dim
            text_encoder_model_max_length = text_encoder.model_max_length
            if self.version == 'v0.1':
                text_encoder.t5.model.requires_grad_(False)
                self.text_encoder_model = text_encoder.t5.model  # a hook to move to device
            elif self.version == 'v1.0':
                text_encoder.model.requires_grad_(False)
                self.text_encoder_model = text_encoder.model   # a hook to move to device
        else:
            text_encoder_output_dim = self.cfg.get("text_encoder_output_dim", 4096)
            text_encoder_model_max_length = self.cfg.get("text_encoder_model_max_length", 512)
        self.text_encoder = text_encoder
        assert self.caption_embed_dim == text_encoder_output_dim
        assert self.caption_embed_num == text_encoder_model_max_length
        
        # == (optional) build prior-encoder ==
        prior_encoder = build_module(self.cfg.get("prior_encoder", None), MODELS)
        if prior_encoder is not None:
            prior_encoder = prior_encoder.eval().requires_grad_(False)
        self.prior_encoder = prior_encoder

        # == build vae ==
        vae = build_module(self.cfg.get("vae", None), MODELS)
        if vae is not None:
            vae = vae.eval().requires_grad_(False)
            input_size = (self.num_frames, *self.image_size)
            latent_size = vae.get_latent_size(input_size)
            vae_out_channels = vae.out_channels
        else:
            latent_size = (None, None, None)
            vae_out_channels = self.cfg.get("vae_out_channels", 4)
        self.vae = vae

        # == build audio vae ==
        audio_vae = build_module(self.cfg.get("audio_vae", None), MODELS)
        if audio_vae is not None:
            audio_vae_out_channels = audio_vae.vae_out_channels
            self.audio_vae_model = audio_vae.pipe.vae  # a hook to move to device
            self.audio_vae_vocoder = audio_vae.pipe.vocoder  # a hook to move to device
        else:
            audio_vae_out_channels = self.cfg.get('audio_vae_out_channels', 8)
        self.audio_vae = audio_vae

        # == build javisdit diffusion model ==
        if self.cfg.get("model", None) is not None:
            ckpt_path = self.cfg.model.pop('weight_init_from', '')
            model = (
                build_module(
                    self.cfg.model,
                    MODELS,
                    input_size=latent_size,
                    in_channels=vae_out_channels,
                    audio_in_channels=audio_vae_out_channels,
                    caption_channels=text_encoder_output_dim,
                    model_max_length=text_encoder_model_max_length,
                    enable_sequence_parallelism=self.cfg.get("sp_size", 1) > 1,
                    weight_init_from=ckpt_path,
                )
                .eval().requires_grad_(False)
            )
            if isinstance(ckpt_path, str) and os.path.exists(ckpt_path):
                lora_ckpt_path = os.path.join(ckpt_path, self.cfg.get("lora_dir", "lora"))
                if os.path.exists(lora_ckpt_path):
                    from peft import PeftModel
                    rank0_print('Loading and merging pretrained LoRA weights for JAV-DiT')
                    model = PeftModel.from_pretrained(model, lora_ckpt_path, is_trainable=False)
                    model = model.merge_and_unload()
            if text_encoder is not None:
                text_encoder.y_embedder = model.y_embedder
            if prior_encoder is not None:
                prior_encoder.st_prior_embedder = model.st_prior_embedder
        else:
            model = None
        self.model = model

        # == setup loss function, build scheduler ==
        self.scheduler = build_module(self.cfg.get("scheduler", None), SCHEDULERS)

        requires_grad(self, False)
        self.eval()

    @property
    def prior_embed_dim(self):
        return 1024

    @property
    def prior_embed_num(self):
        return 77

    @property
    def cond_embed_num(self):
        if self.version == 'v0.1':
            return self.caption_embed_num + self.prior_embed_num
        elif self.version == 'v1.0':
            return self.caption_embed_num
        else:
            raise ValueError(f'Unknown version: {self.version}')

    def get_cond_projector(self, input_dim, hidden_dim=None):
        hidden_dim = hidden_dim or input_dim * 4
        if self.version == 'v0.1':
            projector = HierarchicalCondMLP(
                input_dim, hidden_dim,
                caption_out_dim=self.caption_embed_dim,
                prior_out_dim=self.prior_embed_dim,
                caption_token_num=self.caption_embed_num,
                prior_token_num=self.prior_embed_num,
            )
        elif self.version == 'v1.0':
            projector = HierarchicalCondMLP(
                input_dim, hidden_dim,
                caption_out_dim=self.caption_embed_dim,
                prior_out_dim=None,
            )
        else:
            raise ValueError(f'Unknown version: {self.version}')
        return projector

    def encode_text(self, texts: Union[str, List[str]]):
        caption_embs, emb_masks = self.get_caption_embedding(texts)
        prior_embs = self.get_prior_embedding(texts)

        if prior_embs is not None:
            return (caption_embs, prior_embs), emb_masks
        else:
            return caption_embs, emb_masks

    def get_caption_embedding(self, texts: Union[str, List[str]]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        res = self.text_encoder.encode(texts)
        caption_embs, emb_masks = res['y'], res['mask']

        # HACK: remove the extra dim inherited from javisdit-v0.1
        if self.version == 'v0.1':
            caption_embs = caption_embs.squeeze(1)

        return caption_embs, emb_masks
    
    def get_prior_embedding(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if self.prior_encoder is None:
            return None
        if isinstance(texts, str):
            texts = [texts]
        prior_embs = self.prior_encoder.encode_text(texts)
        return prior_embs

    def check_input_conds(self, input_conds, is_training=True, require_mask=False) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.version == 'v0.1':
            caption_conds, prior_conds = input_conds
            if not is_training:
                caption_conds = caption_conds.unsqueeze(1)
            mask = None
        elif self.version == 'v1.0':
            caption_conds, prior_conds = input_conds, None
            mask = torch.ones(caption_conds.shape[:-1], dtype=torch.long)
            mask = mask.to(caption_conds.device)
        else:
            raise ValueError(f'Unknown version: {self.version}')
    
        if require_mask:  # TODO: check
            caption_conds = (caption_conds, mask)

        return caption_conds, prior_conds

    def compute_caption_loss(self, input_conds, target_texts, prior_scale=1.0, use_cos=True):
        caption_conds, prior_conds = self.check_input_conds(input_conds)

        with torch.no_grad():
            caption_embs, emb_masks = self.get_caption_embedding(target_texts)
            if self.version == 'v0.1':
                prior_embs = self.get_prior_embedding(target_texts)
        
        caption_conds, caption_embs = caption_conds.float(), caption_embs.float()
        if self.version == 'v0.1':
            prior_conds, prior_embs = prior_conds.float(), prior_embs.float()

        caption_loss = F.mse_loss(caption_conds, caption_embs, reduction='none')
        caption_loss = caption_loss[emb_masks > 0].mean()
        # caption_loss = ((caption_loss * emb_masks.unsqueeze(-1)).mean(-1).sum(1) / emb_masks.sum(1)).mean()
        if self.version == 'v0.1':
            prior_loss = F.mse_loss(prior_conds, prior_embs, reduction='mean')
        elif self.version == 'v1.0':
            prior_loss = 0.0

        if use_cos:
            caption_cos_loss = F.cosine_embedding_loss(
                caption_conds.flatten(0,1), 
                caption_embs.flatten(0,1), 
                torch.ones_like(caption_embs[..., 0].flatten()),
                reduction='none'
            )
            caption_loss += caption_cos_loss[emb_masks.flatten() > 0].mean()
            if self.version == 'v0.1':
                prior_loss += F.cosine_embedding_loss(
                    prior_conds.flatten(0,1), 
                    prior_embs.flatten(0,1), 
                    torch.ones_like(prior_embs[..., 0].flatten()),
                    reduction='mean'
                )

        return caption_loss + prior_scale * prior_loss

    def compute_diffusion_loss(self, input_conds, target_audios, target_videos):
        caption_conds, prior_conds = self.check_input_conds(input_conds)

        if self.version == 'v0.1':
            # Inherit `caption_conds[:, None]` from javisdit-v0.1, weird
            model_args = {'sync': True, "y": caption_conds[:, None]}
            assert prior_conds is not None
            model_args.update(self.prior_encoder.encode(prior_conds))
        elif self.version == 'v1.0':
            model_args = {"y": caption_conds}
            assert prior_conds is None

        if self.load_av_feat:
            x, ax = target_videos, target_audios
        else:
            with torch.no_grad():
                x = self.vae.encode(target_videos)  # [B, C, T, H/P, W/P]
                ax = self.audio_vae.encode_audio(target_audios)  # [B, C, T, M]
        x = {'video': x, 'audio': ax}

        mask, ax_mask = None, None  # TODO: temporal mask for x-conditional generation
        model_args.update({'x_mask': mask, 'ax_mask': ax_mask})

        # update model_args for `height`, `width`, `num_frames`, etc
        B, _, T, H, W = x['video'].shape
        dtype, device = caption_conds.dtype, caption_conds.device
        if self.version == 'v0.1':
            num_frames = T // 5 * 17  # HACK: hard-coded for v0.1
        elif self.version == 'v1.0':
            num_frames = 1 + (T - 1) * 4
        model_args.update({
            'height': torch.tensor([H * 8] * B).to(device, dtype),
            'width': torch.tensor([W * 8] * B).to(device, dtype),
            'num_frames': torch.tensor([num_frames] * B).to(device, dtype),
            'fps': torch.tensor([self.video_fps] * B).to(device, dtype),
        })
        loss_dict = self.scheduler.multimodal_training_losses(
            self.model, x, model_args, mask={'video': mask, 'audio': ax_mask}
        )
        diffusion_loss = loss_dict['loss'].mean()

        return diffusion_loss

    def forward(self, x):
        warnings.warn('Deprecated.')
        pass

    def sample(
        self, input_conds: Union[torch.Tensor, List[torch.Tensor]], 
        image_size=None, resolution=None, aspect_ratio=None, num_frames=None, fps=None,
        verbose=2
    ):
        (caption_conds, caption_mask), prior_conds = \
            self.check_input_conds(input_conds, is_training=False, require_mask=True)
        prompts = {"y": caption_conds, "mask": caption_mask, "prior_emb": prior_conds}

        batch_size = caption_conds.shape[0]
        device, dtype = caption_conds.device , caption_conds.dtype
        
        # == prepare video size ==
        image_size = image_size or self.cfg.get("image_size", None)
        if image_size is None:
            resolution = resolution or self.cfg.get("resolution", None)
            aspect_ratio = aspect_ratio or self.cfg.get("aspect_ratio", None)
            assert (
                resolution is not None and aspect_ratio is not None
            ), "resolution and aspect_ratio must be provided if image_size is not provided"
            image_size = get_image_size(resolution, aspect_ratio)
        num_frames = get_num_frames(num_frames or self.cfg.num_frames)

        # == prepare reference ==
        # reference_path = self.cfg.get("reference_path", [("", "")] * batch_size)
        # mask_strategy = self.cfg.get("mask_strategy", [""] * batch_size)

        # == prepare arguments ==
        fps = fps or self.cfg.get("video_fps", 24)
        multi_resolution = self.cfg.get("multi_resolution", None)
        neg_prompts = self.cfg.get("neg_prompt", None)
        if isinstance(neg_prompts, str):
            neg_prompts = [neg_prompts] * batch_size

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            multi_resolution, batch_size, image_size, num_frames, fps, device, dtype
        )

        # == sampling ==
        # torch.manual_seed(1024)
        input_size = (num_frames, *image_size)
        v_latent_size = self.vae.get_latent_size(input_size) 
        vz = torch.randn(batch_size, self.vae.out_channels, *v_latent_size, device=device, dtype=dtype)
        audio_length_in_s = num_frames / fps
        az, original_waveform_length = self.audio_vae.prepare_latents(audio_length_in_s, batch_size, device=device, dtype=dtype)
        # masks = apply_va_mask_strategy(vz, az, refs, ms, loop_i, align=align,  TODO: mask
        #                             v2a_t_scale=1/5*17/fps/(10.24/1024)/4)
        masks = None
        samples = self.scheduler.multimodal_sample(
            self.model,
            self.text_encoder,
            {'video': vz, 'audio': az},
            prompts,
            device=device,
            additional_args=model_args,
            progress=verbose >= 2,
            mask=masks,
            prior_encoder=self.prior_encoder,
            neg_prompts=neg_prompts,
        )
        video_samples, audio_samples = samples['video'].to(dtype), samples['audio'].to(dtype)
        
        video = self.vae.decode(video_samples, num_frames=num_frames)
        self.audio_vae.dtype = dtype
        audio = self.audio_vae.decode_audio(audio_samples, original_waveform_length=original_waveform_length)

        return audio, video

    def save(self, audio, video, save_path, save_fps=None, audio_fps=None, verbose=True):
        save_fps = save_fps or self.cfg.get("save_fps", self.cfg.video_fps // self.cfg.get("frame_interval", 1))
        audio_fps = audio_fps or self.cfg.get("audio_fps")
        save_path = save_sample(
            video, fps=save_fps, audio=audio, audio_fps=audio_fps, save_path=save_path, verbose=verbose,
        )
        return save_path


if __name__ == '__main__':
    cfg_path = 'config/javisdit.py'
    preprocessor = Preprocessor(cfg_path)
    av_generator = JavisDiTInterface(cfg_path)
    import pdb; pdb.set_trace()
    pass

