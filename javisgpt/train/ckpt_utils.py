import os
import logging

import torch
import transformers

from javisgpt.utils import rank0_print


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):  # TODO: configure here
    lora_module_names = set()
    multimodal_keywords = ["visual", "beats", "lm_head", "av_generator", 
                           "audio_mlp", "avsync_proj", "avgen_cond_proj"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, torch.nn.Linear):
            # names = name.split(".")
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            lora_module_names.add(name)

    return list(lora_module_names)


def get_adapter_keys_to_match_by_stage(args):  # TODO: configure here
    training_stage = getattr(args, "training_stage", None)
    rank0_print(f"Saving weights for training stage: {training_stage}")

    if training_stage == 'audio_align':
        keys_to_match = ["embed_tokens", "audio_mlp"]
        adapter_name = 'audio_proj'
    elif training_stage == 'audio_video_align':
        keys_to_match = ["embed_tokens", "avsync_proj"]
        adapter_name = 'avsync_proj'
    elif training_stage == 'audio_video_gen_align':
        keys_to_match = ["lm_head", "avgen_token", "avgen_cond_proj"]
        adapter_name = 'avgen_proj'
    else:  # multimodal training
        keys_to_match = ["embed_tokens", "lm_head", 
                         "audio_mlp", "avsync_proj", "avgen_token", "avgen_cond_proj"]
        adapter_name = 'mm_proj_all'
    
    return adapter_name, keys_to_match


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, adapter_only: bool=False):
    """Collects the state dict and dump to disk."""
    adapter_name, keys_to_match = get_adapter_keys_to_match_by_stage(trainer.args)

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    
    def smart_save(output_dir: str, save_name: str, weight_to_save: object):
        current_folder_name = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank in [-1, 0]:
            if current_folder_name.startswith("checkpoint-"):
                save_folder = os.path.join(parent_folder, save_name)
                os.makedirs(save_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(save_folder, f"{current_folder_name}.bin"))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"{save_name}.bin"))

    if adapter_name is not None:
        # Only save Adapter
        assert len(keys_to_match) >= 0
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)
        smart_save(output_dir, adapter_name, weight_to_save)
        return
    
    if adapter_only:
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
