import argparse

from javisgpt.train.train_audio_video import get_model_and_tokenizer


def merge_lora(args):
    model, tokenizer = get_model_and_tokenizer(args, args, maybe_merge_lora=True, is_training=False)

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='./checkpoints/finetune_1')
    parser.add_argument("--model-base", type=str, default='./checkpoints/pretrain_dec_1/checkpoint-4')
    parser.add_argument("--save-model-path", type=str, default='./checkpoints/nextgpt-v1.5-7b-lora')

    args = parser.parse_args()

    merge_lora(args)