import math
import os.path as osp
import pandas as pd
from typing import Dict, Optional, Sequence, List, Tuple, Union
from copy import deepcopy

from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

from javisgpt.mm_utils import process_audio, process_video
from javisgpt.train.train_audio_video import preprocess_qwen, DataCollatorForSupervisedDataset
from javisgpt.utils import rank0_print


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class ClothoAQADataset(Dataset):
    def __init__(self, questions, tokenizer, data_args):
        self.questions = questions
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        sample = self.questions[idx]

        audio_path   = sample['audio']
        question_id  = sample['id']
        question_raw = sample['question']
        answer       = sample['answer']

        audio_path = osp.join(self.data_args.audio_folder, audio_path)
        _, audio_fbank, audio_grid_thw, second_per_audio_grid_ts = \
            process_audio(
                audio_path, 
                max_audio_length_s=self.data_args.max_audio_length_s, 
                audio_sr=self.data_args.audio_sr
            )

        question = '<audio>\n' + question_raw        
        sources = [[{'from': 'human', 'value': question}, {'from':'gpt', 'value': ''}]]
        input_ids = preprocess_qwen(
            sources, tokenizer=self.tokenizer, audio_grid_thw=audio_grid_thw,
            has_audio=True, avsync_mode=self.data_args.avsync_mode, is_training=False
        )[0]

        question_id = f'audio_{question_id}_{idx}'
        return {
            'question_id': question_id,
            'input_ids':   input_ids,
            'question':    question_raw,
            'answer':      answer,
            'audio_fbank': audio_fbank, 
            'audio_grid_thw':audio_grid_thw, 
            'second_per_audio_grid_ts':second_per_audio_grid_ts,
        }


class TUT2017Dataset(ClothoAQADataset):
    pass


class ActivityNetDataset(Dataset):
    def __init__(self, questions, tokenizer, data_args):
        self.questions = questions
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        sample = self.questions[idx]

        video_path   = sample['video']
        question_id  = sample['id']
        question_raw = sample['question']
        answer       = sample['answer']

        video_file = osp.join(self.data_args.video_folder, video_path)
        pixel_values_videos, video_grid_thw, second_per_grid_ts = process_video(video_file, self.data_args)

        question = '<video>\n' + question_raw        
        sources = [[{'from': 'human', 'value': question}, {'from':'gpt', 'value': ''}]]
        merge_size = self.data_args.image_processor.merge_size
        input_ids = preprocess_qwen(
            sources, tokenizer=self.tokenizer, video_grid_thw=video_grid_thw, merge_size=merge_size,
            has_video=True, avsync_mode=self.data_args.avsync_mode, is_training=False
        )[0]

        question_id = f'video_{question_id}_{idx}'
        return {
            'question_id': question_id,
            'input_ids':   input_ids,
            'question':    question_raw,
            'answer':      answer,
            'pixel_values_videos': pixel_values_videos, 
            'video_grid_thw': video_grid_thw, 
            'second_per_grid_ts': second_per_grid_ts,
        }


class PerceptionDataset(ActivityNetDataset):
    
    def __getitem__(self, idx):
        sample = self.questions[idx]

        video_path  = sample['video']
        question_id = sample['id']
        conversations = sample['conversations']
        question_raw = [conv['value'] for conv in conversations[0::2]]
        answer = [conv['value'] for conv in conversations[1::2]]

        video_file = osp.join(self.data_args.video_folder, video_path)
        pixel_values_videos, video_grid_thw, second_per_grid_ts = process_video(video_file, self.data_args)
        merge_size = self.data_args.image_processor.merge_size

        n_turn = len(conversations) // 2
        input_ids_list = []
        conversations[0]['value'] = '<video>\n' + conversations[0]['value']
        for i in range(n_turn):
            convs = deepcopy(conversations[:(i+1)*2])
            convs[-1]['value'] = ""

            input_ids = preprocess_qwen(
                [convs], tokenizer=self.tokenizer, video_grid_thw=video_grid_thw, merge_size=merge_size,
                has_video=True, avsync_mode=self.data_args.avsync_mode, is_training=False
            )[0]
            input_ids_list.append(input_ids)

        question_id = [f'video_{question_id}_{idx}_round{i}' for i in range(n_turn)]
        pixel_values_videos = [pixel_values_videos] * n_turn
        video_grid_thw = [video_grid_thw] * n_turn
        second_per_grid_ts = [second_per_grid_ts] * n_turn

        return {
            'question_id': question_id,
            'input_ids':   input_ids_list,
            'question':    question_raw,
            'answer':      answer,
            'pixel_values_videos': pixel_values_videos, 
            'video_grid_thw': video_grid_thw, 
            'second_per_grid_ts': second_per_grid_ts,
            'mini_batch': n_turn
        }


class MVBenchDataset(ActivityNetDataset):
    
    def __getitem__(self, idx):
        sample = self.questions[idx]

        video_path   = sample['video']
        question_id  = sample['id']
        question_raw = sample['question']
        answer       = sample['answer']
        bound        = sample['bound']

        video_file = osp.join(self.data_args.video_folder, video_path)
        pixel_values_videos, video_grid_thw, second_per_grid_ts = \
            process_video(video_file, self.data_args, start_s=bound[0], end_s=bound[1])

        question = '<video>\n' + question_raw        
        sources = [[{'from': 'human', 'value': question}, {'from':'gpt', 'value': ''}]]
        merge_size = self.data_args.image_processor.merge_size
        input_ids = preprocess_qwen(
            sources, tokenizer=self.tokenizer, video_grid_thw=video_grid_thw, merge_size=merge_size,
            has_video=True, avsync_mode=self.data_args.avsync_mode, is_training=False
        )[0]

        question_id = f'video_{question_id}_{idx}'
        return {
            'question_id': question_id,
            'input_ids':   input_ids,
            'question':    question_raw,
            'answer':      answer,
            'pixel_values_videos': pixel_values_videos, 
            'video_grid_thw': video_grid_thw, 
            'second_per_grid_ts': second_per_grid_ts,
            'meta': {k: sample[k] for k in ['task_type', 'data_type']}
        }


class AVQADataset(Dataset):
    def __init__(self, questions, tokenizer, data_args):
        self.questions = questions
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        sample = self.questions[idx]
            
        video_path    = sample['video']
        audio_path    = sample['audio']
        question_id   = sample['id']
        conversations = sample['conversations']

        question_raw  = conversations[0]['value']
        answer        = conversations[-1]['value']
        
        conversations[0]['value'] = '<audio_video>\n' + conversations[0]['value']
        conversations[-1]['value'] = ''

        video_file = osp.join(self.data_args.video_folder, video_path)
        pixel_values_videos, video_grid_thw, second_per_audio_video_grid_ts, duration = \
            process_video(video_file, self.data_args, return_duration=True, 
                            max_length_s=self.data_args.max_audio_length_s)

        audio_file = osp.join(self.data_args.audio_folder, audio_path)
        # assume audio have nearly the same duration with video
        _, audio_fbank, audio_grid_thw, *remains = process_audio(
            audio_file, audio_length_s=duration, audio_sr=self.data_args.audio_sr
        )

        audio_video_grid_thw = torch.cat((video_grid_thw, audio_grid_thw), 0)  # shape(6,)

        merge_size = self.data_args.image_processor.merge_size
        input_ids = preprocess_qwen(
            [conversations], tokenizer=self.tokenizer, 
            audio_video_grid_thw=audio_video_grid_thw, merge_size=merge_size,
            has_audio_video=True, avsync_mode=self.data_args.avsync_mode, is_training=False
        )[0]

        question_id = f'audio_video_{question_id}_{idx}'
        sample = {
            'question_id': question_id,
            'input_ids':   input_ids,
            'question':    question_raw,
            'answer':      answer,
            'audio_fbank': audio_fbank, 
            'pixel_values_videos': pixel_values_videos, 
            'audio_video_grid_thw': audio_video_grid_thw, 
            'second_per_audio_video_grid_ts': second_per_audio_video_grid_ts,
        }
        
        return sample


class MusicAVQADataset(AVQADataset):
    pass


class AVSDDataset(AVQADataset):
    
    def __getitem__(self, idx):
        sample = self.questions[idx]

        video_path  = sample['video']
        audio_path  = sample['audio']
        question_id = sample['id']
        conversations = sample['conversations']  # <audio_video> already in

        question_id = [f'video_{question_id}_{idx}_round{i}' for i in range(n_turn)]
        question_raw = [conv['value'] for conv in conversations[0::2]]
        QUESTION_TMPL = 'Answer the following questions based on this sounding video.'
        question_raw = [q.replace(QUESTION_TMPL, '').strip() for q in question_raw]
        answer = [conv['value'] for conv in conversations[1::2]]

        conversations[0]['value'] = '<audio_video>\n' + conversations[0]['value']

        video_file = osp.join(self.data_args.video_folder, video_path)
        pixel_values_videos, video_grid_thw, second_per_audio_video_grid_ts, duration = \
            process_video(video_file, self.data_args, return_duration=True, 
                            max_length_s=self.data_args.max_audio_length_s)

        audio_file = osp.join(self.data_args.audio_folder, audio_path)
        # assume audio have nearly the same duration with video
        _, audio_fbank, audio_grid_thw, *remains = process_audio(
            audio_file, audio_length_s=duration, audio_sr=self.data_args.audio_sr,
        )

        audio_video_grid_thw = torch.cat((video_grid_thw, audio_grid_thw), 0)  # shape(6,)

        merge_size = self.data_args.image_processor.merge_size

        n_turn = len(conversations) // 2
        input_ids_list = []
        for i in range(n_turn):
            convs = deepcopy(conversations[:(i+1)*2])
            convs[-1]['value'] = ""

            input_ids = preprocess_qwen(
                [convs], tokenizer=self.tokenizer, 
                audio_video_grid_thw=audio_video_grid_thw, merge_size=merge_size,
                has_audio_video=True, avsync_mode=self.data_args.avsync_mode, is_training=False
            )[0]
            input_ids_list.append(input_ids)

        audio_fbank = [audio_fbank] * n_turn
        pixel_values_videos = [pixel_values_videos] * n_turn
        audio_video_grid_thw = [audio_video_grid_thw] * n_turn
        second_per_audio_video_grid_ts = [second_per_audio_video_grid_ts] * n_turn
        
        sample = {
            'question_id': question_id,
            'input_ids':   input_ids_list,
            'question':    question_raw,
            'answer':      answer,
            'audio_fbank': audio_fbank, 
            'pixel_values_videos': pixel_values_videos, 
            'audio_video_grid_thw': audio_video_grid_thw, 
            'second_per_audio_video_grid_ts': second_per_audio_video_grid_ts,
            'mini_batch': n_turn,
        }

        return sample


class JavisBenchDataset(Dataset):
    def __init__(self, meta_file: str, tokenizer: AutoTokenizer, data_args, **kwargs):
        self.questions = pd.read_csv(meta_file)
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        sample = self.questions.iloc[idx]

        caption = sample['text']
        question = f"Make a sounding-video out of this:\n{caption}\n"
        question_id = f'JavisBench_{idx}' #sample['id']

        sources = [[{'from': 'human', 'value': question}, {'from':'gpt', 'value': '<gen_audio_video>'}]]
        input_ids = preprocess_qwen(sources, tokenizer=self.tokenizer, 
                                    gen_audio_video=True, avsync_mode=self.data_args.avsync_mode, is_training=False)[0]

        return {
            'question_id': question_id,
            'input_ids':   input_ids,
            'raw_caption': caption
        }


class DataCollatorForEvalDataset(DataCollatorForSupervisedDataset):
    
    def __call__(self, instances: Union[Sequence[Dict]]) -> Dict[str, torch.Tensor]:
        if (mini_batch := instances[0].pop('mini_batch', 0)) > 0:
            assert len(instances) == 1
            instance = instances[0]
            assert all(len(v) == mini_batch for v in instance.values()), {k: len(v) for k, v in instance.items()}
            instances = [{k: v[i] for k, v in instance.items()} for i in range(mini_batch)]
        input_ids = [instance.pop("input_ids")[:self.tokenizer.model_max_length] for instance in instances]
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch = dict(input_ids=input_ids)

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
        }
        mm_dict = {}
        for kw, mm_list in mm_cfg.items():
            for instance in instances:
                if kw in instance:
                    for m in mm_list:
                        mm_dict[m] = mm_dict.get(m, []) + [instance.pop(m)]
        for m, data in mm_dict.items():
            data = [x for x in data if x is not None]
            if len(data) == 0:
                continue
            if m == "audio_fbank":
                batch["audio_fbank"], batch["audio_pad_mask"] = self.pad_temporal(data, padding_value=0)
            elif m in ["pixel_values", "image_grid_thw"]:
                batch[m] = torch.cat(data)
            elif isinstance(data[0], torch.Tensor):
                batch[m] = torch.stack(data)
            else:
                batch[m] = data
        
        if len(instances[0]):
            assert len(set([len(ins) for ins in instances])) == 1  # all instances have the same keys
            for k in instances[0].keys():
                if k in batch:
                    continue
                batch[k] = [instance[k] for instance in instances]

        return batch


def build_dataloader(dataset, tokenizer, gt_questions, data_args=None, **kwargs):
    dataset = eval(f'{dataset}Dataset')(gt_questions, tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForEvalDataset(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset, shuffle=False, 
        batch_size=kwargs.get("batch_size", 1), 
        num_workers=kwargs.get("num_workers", 1), 
        collate_fn=data_collator
    )

    return dataset, dataloader

