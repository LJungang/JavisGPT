## <div align="center"> JavisGPT: A Unified Multi-modal LLM for Sounding-Video Comprehension and Generation</div>

<div align="center">

[[`HomePage`](https://javisverse.github.io/JavisGPT-page/)] 
[[`Paper`](https://openreview.net/forum?id=MZoOpD9NHV)] 
[[`Model`](https://huggingface.co/collections/JavisVerse/javisgpt)] 

</div>


## TL;DR

We introduce **`JavisGPT`**, a multimodal LLM that can understand audiovisual inputs and simultaneously generate synchronized sounding videos in a unified model. 
We also curate the **`JavisInst-Omni`** dataset to facilitate instruction-tuning for comprehension and generation on sounding videos.


![framework](./assets/image/framework.jpg)


## 📰 News

- **[2025.12.26]** 🔥 We release the code of [JavisGPT](#), with the preview [JavisGPT-v0.1-7B-Instruct](https://huggingface.co/JavisVerse/JavisGPT-v0.1-7B-Instruct) checkpoint at huggingface. Feel free to play with it!

### 👉 TODO 
- [ ] Release the training/evaluation dataset and scripts.
- [ ] Derive a more powerful JavisGPT model.


## Code


### Installation

Install the necessary packages:

```bash
conda create -n javisgpt python=3.10 -y
conda activate javisgpt
pip install --upgrade pip  # Enable PEP 660 support.
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -v -e ".[train]"
cp assets/src/dynamic_modules_utils.py /path/to/python3.10/site-packages/diffusers/utils/
conda install "ffmpeg<7" -c conda-forge -y  # install ffpmeg
```

Install [JavisDiT](https://github.com/JavisVerse/JavisDiT.git) dependencies:

```bash
cd ..
git clone https://github.com/JavisVerse/JavisDiT.git
cd JavisDiT
pip install -v -e . --no-deps
cd ../JavisGPT

# # make soft links if necessary
# ln -s ../JavisDiT/javisdit javisdit
```

<!-- Install the evaluation package:
```bash
conda install -c conda-forge openjdk=8
bash javisgpt/eval/caption_evaluation_tools/coco_caption/get_stanford_models.sh
pip install vllm==0.7.3
``` -->

### Inference

We assume the data structure as:

```bash
/path/to/user/root
|-- projects
|   |   |-- JavisDiT  # downstream JAV-DiT
|   |   └-- JavisGPT  # workspace of this project
|-- weights
|   |-- pretrained
|   |   |-- dit   # pretrained weights for JavisDiT
|   |   |   |-- OpenSora-VAE-v1.2
|   |   |   |-- audioldm2
|   |   |   |-- t5-v1_1-xxl
|   |   |   |-- imagebind_huge.pth
|   |   |-- mllm  # pretrained weights for JavisGPT
|   |   |   |-- BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
|   |   |   └-- Qwen2.5-VL-7B-Instruct
|   |-- JavisVerse
|   |   |-- JavisDiT-v0.1-prior
|   |   |-- JavisDiT-v0.1-jav-240p4s
|   |   └-- JavisGPT-v0.1-7B-Instruct
|-- datasets
|   |-- JavisGPT
|   |   |-- train
|   |   |   |-- MM-PreTrain
|   |   |   |-- AV-FineTune
|   |   |   └-- JavisInst-Omni
|   |   |-- eval
|   |   |   └-- JavisUnd-Eval
```

#### 1. Prepare pretrained weights

First, download [BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt](https://github.com/microsoft/unilm/tree/master/beats) from [here](https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea) and [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), and put (or link) them into `../../weights/pretrained/mllm`.

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ../../weights/pretrained/mllm/Qwen2.5-VL-7B-Instruct
```

Then, download our [JavisGPT-v0.1-7B-Instruct](https://huggingface.co/JavisVerse/JavisGPT-v0.1-7B-Instruct) and put it into `../../weights/JavisVerse`.

```bash
huggingface-cli download JavisVerse/JavisGPT-v0.1-7B-Instruct --local-dir ../../weights/JavisVerse/JavisGPT-v0.1-7B-Instruct
```

Finally, download necessary checkpoints of the downstream JAVG model ([JavisDiT](https://github.com/JavisVerse/JavisDiT.git)) and put them into `../../weights/pretrained/dit` or `../../weights/JavisVerse`, according to path definition in `./interface/config/*.py` coordinately.

```bash
huggingface-cli download hpcai-tech/OpenSora-VAE-v1.2 --local-dir ../../weights/pretrained/dit/OpenSora-VAE-v1.2
huggingface-cli download cvssp/audioldm2 --local-dir ../../weights/pretrained/dit/audioldm2
huggingface-cli download DeepFloyd/t5-v1_1-xxl --local-dir ../../weights/pretrained/dit/t5-v1_1-xxl
huggingface-cli download JavisVerse/JavisDiT-v0.1-prior --local-dir ../../weights/JavisVerse/JavisDiT-v0.1-prior
huggingface-cli download JavisVerse/JavisDiT-v0.1-jav-240p4s --local-dir ../../weights/JavisVerse/JavisDiT-v0.1-jav-240p4s
```

#### 2. Run target inference

- **Standalone Audio/Visual Comprehension**

Use the following commands to evaluate the preserved single-modality understanding capability.

For audio comprehension:

```bash
AUDIO_PATH="assets/demos/audio/Creaking_pier.wav"
PROMPT="Is the sound caused by pressure from/against wood?"

AUDIO_PATH=${AUDIO_PATH} PROMPT=${PROMPT} bash scripts/demo/demo_audio_visual.sh
```

For video comprehension:

```bash
VIDEO_PATH="assets/demos/video/ZS9XR.mp4"
PROMPT="What happened after the person took the box? A. Ate the medicine. B. Tidied up the blanket. C. Put down the cup/glass/bottle. D. Open the computer."

VIDEO_PATH=${VIDEO_PATH} PROMPT=${PROMPT} bash scripts/demo/demo_audio_visual.sh
```

- **Joint Audio-Video Comprehension**

Use the following command to evaluate the joint audio-video comprehension capability.


```bash
VIDEO_PATH="assets/demos/audio_video/00002617.mp4"
PROMPT="How many instruments in the room did not sound from beginning to end? Answer the question using a single word."
USE_AUDIO_IN_VIDEO=True

VIDEO_PATH=${VIDEO_PATH} PROMPT=${PROMPT} USE_AUDIO_IN_VIDEO=${USE_AUDIO_IN_VIDEO} bash scripts/demo/demo_audio_visual.sh
```


- **Joint Audio-Video Generation**

Use the following command to evaluate the sounding video generation capability.

```bash
PROMPT="Build a video, ensuring the content is echoed by complementary scenes: A beautiful waterfall cascades down a steep cliff into a clear pool below. Sunlight filters through the surrounding trees, creating shimmering reflections on the falling water. The scene is calm and natural, with continuous flowing water and gentle mist rising from the base. The sound consists of steady rushing water, soft splashes, and faint ambient forest noise."
AV_GENERATE=True
SAVE_PREFIX="./results/avgen/demo"

AV_GENERATE=${AV_GENERATE} PROMPT=${PROMPT} SAVE_PREFIX=${SAVE_PREFIX} bash scripts/demo/demo_audio_visual.sh
```

The generated sample will be saved at `${SAVE_PREFIX}.mp4`, e.g., `./results/avgen/demo.mp4`.

### Training

- [ ] Coming soon.

### Evaluation

- [ ] Coming soon.

## Citation

If you find JavisGPT is useful and use it in your project, please kindly cite:
```
@inproceedings{liu2025javisgpt,
    title={JavisGPT: A Unified Multi-modal LLM for Sounding-Video Comprehension and Generation},
    author={Kai Liu and Jungang Li and Yuchong Sun and Shengqiong Wu and jianzhang gao and Daoan Zhang and Wei Zhang and Sheng Jin and Sicheng Yu and Geng Zhan and Jiayi Ji and Fan Zhou and Liang Zheng and Shuicheng YAN and Hao Fei and Tat-Seng Chua},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
}
```

