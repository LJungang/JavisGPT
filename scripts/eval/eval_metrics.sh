

export CUDA_VISIBLE_DEVICES="0"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

USER_ROOT="/mnt/HithinkOmniSSD/user_workspace/liukai4"
WEIGHT_ROOT="${USER_ROOT}/weights"
judge_model_name_or_path="${WEIGHT_ROOT}/pretrained/mllm/Qwen2.5-14B-Instruct"

python javisgpt/eval/eval_und.py \
    --res_dir "results/qwen25_vl/joint/gen_to_und_shuffle_v2" \
    --judge_model_name_or_path "${judge_model_name_or_path}"



