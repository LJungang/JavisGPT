import os
version = "v0.1"  # JavisDiT-v0.1 (JavisDiT)

WEIGHT_ROOT = os.environ.get("WEIGHT_ROOT", "../../weights")

# Model settings
pred_onset = False
spatial_prior_len = 32
temporal_prior_len = 32
st_prior_channel = 128

text_encoder_model_max_length=300
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