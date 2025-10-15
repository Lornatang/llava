# llava

LLaVA: Large Language and Vision Assistant. From data to deployment.

## Install

**Please note that this is only supported on Linux systems.**

**1. Clone repository**

```shell
git clone https://github.com/Lornatang/llava.git
cd llava
```

**2. Install Package**

```shell
conda create -n llava python=3.12.11 -y
conda activate llava
pip3 install --upgrade pip
pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip3 install ./flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip3 install -e .
```

## Train LLaVA-OneVision (Recommended)

## Introduction

pass

## Train LLaVA

### Introduction

The training is mainly divided into two stages. The first stage is to achieve basic alignment of cross-modal features, referred to as pre-training.
The second stage is based on feature alignment and end-to-end fine-tuning to allow the model to learn to follow diverse visual-language instructions
and generate responses that meet the requirements.

### Data Preparation

The LAION-CC-SBU with BLIP captions dataset with 558k data is used for pre-training.
- [blip_laion_cc_sbu_558k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k.json)
- [blip_laion_cc_sbu_558k_meta.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k_meta.json)
- [images.zip](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip)

This is the official mixed dataset annotation, please download it for fine-tuning.
- [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)
- [coco train2017](http://images.cocodataset.org/zips/train2017.zip)
- [gqa images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- [ocr_vqa images](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing)
- [textvqa train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- [vg part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) 
- [vg part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

Please place the downloaded files into the datasets directory according to the following format requirements.

```txt
- datasets
    - llava_pretrain
        - blip_laion_cc_sbu_558k
            - blip_laion_cc_sbu_558k.json
            - blip_laion_cc_sbu_558k_meta.json
            - images
    - llava_finetune
        - llava_v1_5_mix665k.json
        - coco
            - train2017
        - gqa
            - images
        - ocr_vqa
            - images
        - textvqa
            - train_images
        - vg
            - VG_100K
            - VG_100K_2
```

### Train LLaVA pipeline

**You can try different combinations of visual language architectures as shown below!**

We use [LLM(lmsys/vicuna-13b-v1.5)](https://huggingface.co/lmsys/vicuna-13b-v1.5) and [VIT(openai/clip-vit-large-patch14-336)](https://huggingface.co/openai/clip-vit-large-patch14-336) for the following examples, but you can also use other LLMs and VIT.

```shell
hf download openai/clip-vit-large-patch14-336 --local-dir ./results/pretrained_models/openai/clip-vit-large-patch14-336
hf download lmsys/vicuna-13b-v1.5 --local-dir ./results/pretrained_models/lmsys/vicuna-13b-v1.5
# Stage1: Pretrain (Visual Feature Alignment).
bash ./tools/stage1_pretrain.sh
# Stage2: Full Finetune (Instruction Tuning with Image-Text Pairs)
bash ./tools/stage2_finetune.sh
```

# Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): Providing the most original implementation, thanks.
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): Provide many methods that are beneficial to the implementation of this project.
- [Vicuna](https://github.com/lm-sys/FastChat): Provides many optional multimodal data processing methods.
- [Qwen](https://huggingface.co/Qwen): Provides LLM that is easy to fine-tune and has excellent performance.

## Citation

```bibtex
@misc{liu2024llavanext,
    title={LLaVA-NeXT: Improved reasoning, OCR, and world knowledge},
    url={https://llava-vl.github.io/blog/2024-01-30-llava-next/},
    author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Li, Bo and Zhang, Yuanhan and Shen, Sheng and Lee, Yong Jae},
    month={January},
    year={2024}
}

@misc{liu2023improvedllava,
      title={Improved Baselines with Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
      publisher={arXiv:2310.03744},
      year={2023},
}

@misc{liu2023llava,
      title={Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={NeurIPS},
      year={2023},
}
```
