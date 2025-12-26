import os
version = "v0.1"  # JavisDiT-v0.1 (JavisDiT)

WEIGHT_ROOT = os.environ.get("WEIGHT_ROOT", "../../weights")

# Data settings
num_frames = 102  # fps=24
image_size = (240, 426)  # height, width
video_fps = 24
frame_interval = 1
direct_load_video_clip = True

# Save settings
audio_fps = 16000
save_fps = 24
multi_resolution = "OpenSora"
condition_frame_length = 5  # used for video extension conditioning
align = 5  # TODO: unknown mechanism, maybe for conditional frame alignment?

# Model settings
text_encoder_model_max_length=300
pred_onset = False
spatial_prior_len = 32
temporal_prior_len = 32
st_prior_channel = 128

model = dict(
    type="VASTDiT3-XL/2",
    weight_init_from=[],
    from_pretrained=f"{WEIGHT_ROOT}/JavisVerse/JavisDiT-v0.1-jav-240p4s",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
    # video-audio joint generation
    only_train_audio=False,
    freeze_y_embedder=True,
    freeze_video_branch=True,
    freeze_audio_branch=True,
    train_st_prior_attn=True,
    train_va_cross_attn=True,
    spatial_prior_len=spatial_prior_len,
    temporal_prior_len=temporal_prior_len,
    st_prior_channel=st_prior_channel,
    audio_patch_size=(4, 1),
    require_onset=pred_onset
)
audio_vae = dict(
    type="AudioLDM2",
    from_pretrained=f"{WEIGHT_ROOT}/pretrained/dit/audioldm2",
    init_to_device=False,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained=f"{WEIGHT_ROOT}/pretrained/dit/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained=f"{WEIGHT_ROOT}/pretrained/dit/t5-v1_1-xxl",
    model_max_length=text_encoder_model_max_length,
    # shardformer=True,
)
prior_encoder = dict(
    type="STIBPrior",
    imagebind_ckpt_path=f"{WEIGHT_ROOT}/pretrained/dit",
    from_pretrained=f"{WEIGHT_ROOT}/JavisVerse/JavisDiT-v0.1-prior",
    spatial_token_num=spatial_prior_len,
    temporal_token_num=temporal_prior_len,
    out_dim=st_prior_channel,
    apply_sampling=True,
    encode_va=False,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
    pred_onset=pred_onset
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
    num_sampling_steps=30,
    cfg_scale=7.0,
)

# Audio settings
sampling_rate = 16000
mel_bins = 64
audio_cfg = {
    "preprocessing": {
        "audio": {
            "sampling_rate": sampling_rate,
            "max_wav_value": 32768.0,
            "duration": 10.24,
        },
        "stft": {
            "filter_length": 1024,
            "hop_length": 160,
            "win_length": 1024,
        },
        "mel": {
            "n_mel_channels": mel_bins,
            "mel_fmin": 0,
            "mel_fmax": 8000,
        }
    },
    "augmentation": {
        "mixup": 0.0,
    }
}