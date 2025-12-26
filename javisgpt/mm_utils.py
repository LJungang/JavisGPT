from PIL import Image
from io import BytesIO
import base64
import math
import ast
import re
import random
import numpy as np
from typing import Literal

import torch
from transformers import StoppingCriteria
from javisgpt.constants import IMAGE_TOKEN_INDEX, AUDIO_TOKEN_INDEX


try:
    import av
    from decord import VideoReader, cpu
except ImportError:
    print("Please install pyav to use video processing functions.")
from PIL import Image, ImageOps
import cv2
import librosa
import soundfile as sf
import torchaudio.compliance.kaldi as ta_kaldi


def process_audio(wav_path, audio_length_s=None, max_audio_length_s=None, audio_sr=16000):
    wav, sr = librosa.load(wav_path, sr=audio_sr, duration=max_audio_length_s or audio_length_s, offset=0)
    
    if audio_length_s:
        target_length = int(audio_length_s * sr)
        if len(wav) > target_length:
            wav = wav[:target_length]
        elif len(wav) < target_length:
            pad_length = target_length - len(wav)
            wav = np.pad(wav, (0, pad_length), mode='constant', constant_values=0.0)
    
    audio_wav = torch.from_numpy(wav)
    wavform = audio_wav.unsqueeze(0) * 2 ** 15
    # shape[T,128]
    fbank = ta_kaldi.fbank(wavform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)

    audio_grid_thw = torch.tensor((fbank.shape[0] // 16, fbank.shape[1] // 16, 1))

    second_per_audio_grid_ts = torch.tensor(len(wav) / sr / audio_grid_thw[0], )

    ret = (wav, fbank, audio_grid_thw, second_per_audio_grid_ts, )
    
    return ret


def process_image(image_file, data_args, patch_min=28, pixel_max=1920*1080):
    processor = data_args.image_processor
    image = Image.open(image_file).convert("RGB")
    w, h = image.size
    if min(w, h) < patch_min:
        w, h = max(w, patch_min), max(h, patch_min)
        image = ImageOps.pad(image, (w, h))
    if w * h > pixel_max:
        s = math.sqrt(pixel_max / (w * h))
        _w, _h = int(w * s), int(h * s)
        image = ImageOps.contain(image, (_w, _h))
    image = processor.preprocess(image, return_tensors="pt")
    pixel_values = image["pixel_values"]
    image_grid_thw = image["image_grid_thw"][0]
    return pixel_values, image_grid_thw


def process_video(video_file, data_args, backend:Literal['decord', 'pyav']='decord', 
                  return_duration=False, max_length_s=None, start_s=None, end_s=None):
    process_video_func = eval(f'process_video_with_{backend}')
    processor = data_args.image_processor
    video, _, frame_time, _, sample_fps = process_video_func(
        video_file, data_args, max_length_s, start_s=start_s, end_s=end_s
    )
    video = processor.preprocess(images=None, videos=video, return_tensors="pt")
    pixel_values_videos = video["pixel_values_videos"]
    video_grid_thw = video["video_grid_thw"][0]
    second_per_grid_ts = torch.tensor(processor.temporal_patch_size / sample_fps, )

    ret = (pixel_values_videos, video_grid_thw, second_per_grid_ts, )

    if return_duration:
        ret += (frame_time[-1], )

    return ret


def process_video_with_decord(video_file, data_args, max_length_s=None, start_s=None, end_s=None):
    width, height = data_args.frame_width, data_args.frame_height
    # TODO: currently support fixed resolution
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1, width=width, height=height)
    total_frame_num = len(vr)
    start_frame_idx = int(start_s * vr.get_avg_fps()) if start_s else 0
    total_frame_num = min(int(end_s * vr.get_avg_fps()), total_frame_num) if end_s else total_frame_num
    if max_length_s is not None and max_length_s > 0:
        assert start_s is None and end_s is None, 'Currently not supported'
        total_frame_num = min(total_frame_num, int(vr.get_avg_fps() * max_length_s))
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    frame_idx = [i for i in range(start_frame_idx, total_frame_num, avg_fps)]
    frame_time = [i/avg_fps for i in frame_idx]
    
    if data_args.frames_upbound > 0 and (len(frame_idx) > data_args.frames_upbound or data_args.force_sample):
        uniform_sampled_frames = np.linspace(start_frame_idx, total_frame_num - 1, data_args.frames_upbound, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]

    if data_args.frame_stride > 0 and len(frame_idx) % data_args.frame_stride != 0:
        num_frames = int(np.ceil(len(frame_idx) / data_args.frame_stride)) * data_args.frame_stride
        uniform_sampled_frames = np.linspace(start_frame_idx, total_frame_num - 1, num_frames, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    video = vr.get_batch(frame_idx).asnumpy()
    # frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = num_frames = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)

    sample_fps = num_frames / (total_frame_num-start_frame_idx+1) * vr.get_avg_fps()

    return video, video_time, frame_time, num_frames_to_sample, sample_fps


def process_video_with_pyav(video_file, data_args, max_length_s=None, start_s=None, end_s=None):
    container = av.open(video_file)
    # !!! This is the only difference. Using auto threading
    container.streams.video[0].thread_type = "AUTO"
    width, height = data_args.frame_width, data_args.frame_height

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time
    video_fps = total_frame_num / video_time
    start_frame_idx = int(start_s * video_fps) if start_s else 0
    total_frame_num = min(int(end_s * video_fps), total_frame_num) if end_s else total_frame_num
    if max_length_s is not None and max_length_s > 0:
        assert start_s is None and end_s is None, 'Currently not supported'
        total_frame_num = min(total_frame_num, int(video_fps * max_length_s))
        video_frames = video_frames[:total_frame_num]
        video_time = total_frame_num / video_fps
    avg_fps = round(video_fps / data_args.video_fps)
    frame_idx = [i for i in range(start_frame_idx, total_frame_num, avg_fps)]
    frame_time = [i/avg_fps for i in frame_idx]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(start_frame_idx, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/video_fps for i in frame_idx]

    if data_args.frame_stride > 0 and len(frame_idx) % data_args.frame_stride != 0:
        num_frames = int(np.ceil(len(frame_idx) / data_args.frame_stride)) * data_args.frame_stride
        uniform_sampled_frames = np.linspace(start_frame_idx, total_frame_num - 1, num_frames, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/video_fps for i in frame_idx]

    frames = []
    for i in frame_idx:
        frame = video_frames[i].to_ndarray(format="rgb24")
        if width is not None and height is not None:
            frame = cv2.resize(frame, (width, height))
        frames.append(frame)
    video = np.stack(frames)
    # frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    num_frames_to_sample = num_frames = len(frame_idx)

    sample_fps = num_frames / (total_frame_num-start_frame_idx+1) * video_fps

    return video, video_time, frame_time, num_frames_to_sample, sample_fps


# TODO: experimental
def tokenizer_image_audio_token(prompt, tokenizer=None, image_token_index=IMAGE_TOKEN_INDEX, audio_token_index=AUDIO_TOKEN_INDEX, return_tensors=None):

    input_ids = []
    while "<image>" in prompt or "<audio>" in prompt:
        index_im = prompt.find("<image>")
        index_ad = prompt.find("<audio>")

        if index_im<index_ad and index_im!=-1:
            sep = image_token_index
            prompt_chunks = prompt.split("<image>", 1) # 分成2部分
        else:
            sep = audio_token_index
            prompt_chunks = prompt.split("<audio>", 1) # 分成2部分

        input_ids.append(tokenizer(prompt_chunks[0]).input_ids + [sep])

        prompt = prompt_chunks[1]

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids



if __name__ == '__main__':
    _, audio_fbank, audio_grid_thw, *remains = \
        process_audio("../VAffusion/samples/x_cond/debug/sample_0000.wav")
    import pdb; pdb.set_trace()
    pass

# print(tokenizer_image_token("based on this <image>, describe this <audio> the answer is"))

# import pdb; pdb.set_trace()