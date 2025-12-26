import argparse
import torch
import os
import pdb

from javisgpt.constants import *
from javisgpt.train.train_audio_video import preprocess_qwen, get_model_and_tokenizer
from javisgpt.model.qwen2_vl import Qwen2VLProcessor, Qwen2VLImageProcessor
from javisgpt.utils import disable_torch_init, set_random_seed, str2bool
from javisgpt.mm_utils import process_video, process_audio, process_image

set_random_seed(42)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Input arguments
    parser.add_argument("--image_path", help="Path to the image file")
    parser.add_argument("--video_path", help="Path to the video file")
    parser.add_argument("--audio_path", help="Path to the audio file")
    parser.add_argument("--prompt", type=str, default=None) 
    parser.add_argument("--history", nargs='+', default=[]) 
    parser.add_argument("--av_generate", type=str2bool, default=False)
    parser.add_argument("--save_path_prefix", type=str, default='./debug/demo/gen_test', 
                        help="Path prefix to save the generated sounding video")
    parser.add_argument("--prefill_avgen_token", type=str2bool, default=False)
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="./pretrained_models/Qwen2-VL-7B-Instruct")
    parser.add_argument("--beats_path", type=str, default="./pretrained_models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    parser.add_argument("--audio_projector_path", type=str, default=None)
    parser.add_argument("--avsync_mode", type=str, default="merge")
    parser.add_argument("--avsync_projector_path", type=str)
    parser.add_argument("--avgen_cfg_path", type=str)
    parser.add_argument("--avgen_projector_path", type=str)
    parser.add_argument("--all_projector_path", type=str, default=None)
    parser.add_argument("--lora_weight_path", type=str, default="")
    # Model arguments
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--bf16", type=str2bool, default=False)
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--model_max_length", type=int, default=32768)
    # Process arguments
    parser.add_argument("--video_fps", type=int, default=2)
    parser.add_argument("--frames_upbound", type=int, default=16)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--frame_width", type=int, default=588)
    parser.add_argument("--frame_height", type=int, default=336)
    parser.add_argument("--force_sample", type=str2bool, default=True)
    parser.add_argument("--audio_sr", type=int, default=16000)
    parser.add_argument("--max_audio_length_s", type=float, default=30)
    parser.add_argument("--use_audio_in_video", type=str2bool, default=False)

    
    return parser.parse_args()


@torch.no_grad()
def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    
    disable_torch_init()
    
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model, tokenizer = get_model_and_tokenizer(args, args, maybe_merge_lora=True, is_training=False)
    model = model.to(device='cuda', dtype=dtype)
    image_processor = Qwen2VLImageProcessor.from_pretrained(args.model_name_or_path)
    setattr(args, 'image_processor', image_processor)

    preprocess_args, input_args = {}, {}
    if args.use_audio_in_video:
        args.audio_path = args.video_path
    if args.audio_path and args.video_path:
        pixel_values_videos, video_grid_thw, second_per_audio_video_grid_ts, duration = \
            process_video(args.video_path, args, return_duration=True, 
                            max_length_s=args.max_audio_length_s)
        # assume audio have nearly the same duration with video
        _, audio_fbank, audio_grid_thw, *remains = process_audio(
            args.audio_path, audio_length_s=duration, audio_sr=args.audio_sr
        )
        audio_video_grid_thw = torch.cat((video_grid_thw, audio_grid_thw), 0)  # shape(6,)

        sources = [[{'from': 'human', 'value': '<audio_video>\n' + args.prompt}, {'from':'gpt', 'value': ''}]]
        preprocess_args.update({'has_audio_video': True, 'audio_video_grid_thw': audio_video_grid_thw})
        input_args.update({
            'audio_fbank': audio_fbank.unsqueeze(0),
            'pixel_values_videos': pixel_values_videos.unsqueeze(0), 
            'audio_video_grid_thw': audio_video_grid_thw.unsqueeze(0),
            'second_per_audio_video_grid_ts': second_per_audio_video_grid_ts.unsqueeze(0)
        })
    elif args.audio_path and args.image_path:
        raise NotImplementedError
    elif args.audio_path:
        audio_wav, audio_fbank, audio_grid_thw, second_per_audio_grid_ts = process_audio(args.audio_path)
        sources = [[{'from': 'human', 'value': '<audio>\n' + args.prompt}, {'from':'gpt', 'value': ''}]]
        preprocess_args.update({'has_audio': True, 'audio_grid_thw': audio_grid_thw})
        input_args.update({'audio_fbank': audio_fbank.unsqueeze(0), 'audio_grid_thw': audio_grid_thw.unsqueeze(0),
                            'second_per_audio_grid_ts': second_per_audio_grid_ts.unsqueeze(0)})
    elif args.image_path:
        image = [process_image(args.image_path, args)]
        sources = [[{'from': 'human', 'value': '<image>\n' + args.prompt}, {'from':'gpt', 'value': ''}]]
        pixel_values, image_grid_thw = [torch.stack(x) for x in zip(*image)]
        preprocess_args.update({'has_image': True, 'image_grid_thw': image_grid_thw[0]})
        input_args.update({'pixel_values': pixel_values, 'image_grid_thw': image_grid_thw})
    elif args.video_path:
        pixel_values_videos, video_grid_thw, second_per_grid_ts = process_video(args.video_path, args)
        sources = [[{'from': 'human', 'value': '<video>\n' + args.prompt}, {'from':'gpt', 'value': ''}]]
        preprocess_args.update({'has_video': True, 'video_grid_thw': video_grid_thw})
        input_args.update({'pixel_values_videos': pixel_values_videos, 'video_grid_thw': video_grid_thw.unsqueeze(0),
                            'second_per_grid_ts': second_per_grid_ts.unsqueeze(0)})
    else:
        sources = [[{'from': 'human', 'value': args.prompt}, {'from':'gpt', 'value': ''}]]
    if args.av_generate:
        preprocess_args.update({'gen_audio_video': True})
        if args.prefill_avgen_token:
            sources[0][-1]['value'] += '<gen_audio_video>'

    context_round = len(args.history)
    if context_round > 0:
        assert context_round % 2 == 0
        context = []
        for r in range(0, context_round, 2):
            context.append({'from': 'human', 'value': args.history[r]})
            context.append({'from': 'gpt', 'value': args.history[r+1]})
        sources = [context + sources[0]]

    ## skip the end tag `<|im_end|>\n` for last 2 tokens
    input_ids = preprocess_qwen(sources, tokenizer=tokenizer, is_training=False, **preprocess_args)
    input_texts = tokenizer.batch_decode(torch.where(input_ids > 0, input_ids, torch.full_like(input_ids, tokenizer.encode("#")[0])))
    print("inputs:", input_texts)

    im_end_id = tokenizer.convert_tokens_to_ids([DEFAULT_IM_END_TOKEN])
    avgen_start_id = tokenizer.convert_tokens_to_ids([DEFAULT_AUDIO_VIDEO_START_TOKEN, DEFAULT_AUDIO_VIDEO_END_TOKEN])
    eos_token_id = im_end_id + avgen_start_id
    with torch.inference_mode():
        input_args = {k: v.to(model.device) for k, v in input_args.items()}
        input_args.update(dict(
            inputs=input_ids.to(model.device), 
            do_sample=True, temperature=0.1, max_new_tokens=256, top_p=0.1, num_beams=1,
            eos_token_id=eos_token_id,
        ))

        outputs = model.generate(return_dict_in_generate=True, output_hidden_states=True, **input_args)
        generated_ids = outputs.sequences
        output_texts = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:])
        print("outputs:", output_texts)

        if args.av_generate:
            assert DEFAULT_AUDIO_VIDEO_START_TOKEN in output_texts[0] + input_texts[0]
            os.makedirs(os.path.dirname(args.save_path_prefix), exist_ok=True)
            if output_texts[0].strip().endswith(DEFAULT_AUDIO_VIDEO_START_TOKEN):
                assert not args.prefill_avgen_token
                print('Generate for open-ended conversation')
                past_key_values = outputs.past_key_values
                attention_mask = generated_ids.ne(tokenizer.pad_token_id)
                audio, video = model.generate_audio_video_infer(attention_mask, past_key_values)
                output_texts[0] = output_texts[0].replace(DEFAULT_AUDIO_VIDEO_START_TOKEN, '')
            else:
                assert args.prefill_avgen_token
                print('Generate for av-generation alignment')
                hidden_states = outputs.hidden_states[0][-1]
                avgen_embeds = model.parse_avgen_embed(hidden_states, input_ids, GEN_AUDIO_VIDEO_TOKEN_INDEX)
                audio, video = model.generate_audio_video_direct(avgen_embeds)
            model.av_generator.save(audio[0], video[0], save_path=args.save_path_prefix)
    

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)

# CUDA_VISIBLE_DEVICES=4 python demo/audio_visual_demo.py --model-path /fs/fast/share/aimind_files/ycsun/pretrained_models/Qwen2-VL-7B-Instruct --audio_projector_path checkpoints/va_mllm_stage1_audio_align_audiocaps_wavcaps_cloth_merged_v1_3k_test_llava_format_debug/audio_projector.bin --prompt 'Please describe the content of this audio.' --audio_path case_data/YmcNmtHjbjew.flac
    