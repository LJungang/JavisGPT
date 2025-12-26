from typing import Literal
import os
BASE_ARCH: Literal["Qwen2VL", "Qwen2_5_VL"] = \
    os.environ.get("BASE_ARCH", "Qwen2VL")

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

AV_GEN_TOKEN_NUM = int(os.environ.get("AV_GEN_TOKEN_NUM", '512'))
# for JavisDiT-v0.1
# AV_GEN_TOKEN_NUM = 300 + 77
# for JavisDiT-v1.0 (JavisDiT++)
# AV_GEN_TOKEN_NUM = 512

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -300
VIDEO_TOKEN_INDEX = -400
AUDIO_VIDEO_TOKEN_INDEX = -500
GEN_AUDIO_VIDEO_TOKEN_INDEX = -600
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_AUDIO_TOKEN = "<audio>"
DEFAULT_AUDIO_VIDEO_TOKEN = "<audio_video>"
DEFAULT_GEN_AUDIO_VIDEO_TOKEN = "<gen_audio_video>"
DEFAULT_VISION_START_TOKEN = "<|vision_start|>"
DEFAULT_VISION_END_TOKEN = "<|vision_end|>"
DEFAULT_AUDIO_START_TOKEN = "<|audio_start|>"
DEFAULT_AUDIO_END_TOKEN = "<|audio_end|>"
DEFAULT_AUDIO_PAD_TOKEN = "<|audio_pad|>"
DEFAULT_AUDIO_VIDEO_START_TOKEN = "<|audio_video_start|>"
DEFAULT_AUDIO_VIDEO_END_TOKEN = "<|audio_video_end|>"
DEFAULT_AUDIO_VIDEO_PAD_TOKEN = "<|audio_video_pad|>"
DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
