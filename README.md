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
conda create -n llava python=3.11.13 -y
conda activate llava
pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip3 install ./flash_attn-2.8.2+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
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
            - train_val_images
        - vg
            - VG_100K
            - VG_100K_2
```

### Train LLaVA pipeline

**You can try different combinations of visual language architectures as shown below!**

- Vicuna-13B-v1.5 + CLIP-ViT-L-14-336px (default)

```shell
hf download openai/clip-vit-large-patch14-336 --local-dir ./results/pretrained_models/openai/clip-vit-large-patch14-336
hf download lmsys/vicuna-13b-v1.5 --local-dir ./results/pretrained_models/lmsys/vicuna-13b-v1.5
# Pretrain
bash ./tools/pretrain/llava-vicuna_13b_v1.5-clip_vit_large_patch14_336-blip_laion_cc_sbu_558k.sh
# Finetune
bash ./tools/finetune/llava-vicuna_13b_v1.5-clip_vit_large_patch14_336-llava_v1_5_mix665k.sh
```

- TinyVicuna-1B + CLIP-ViT-L-14-336px

```shell
hf download openai/clip-vit-large-patch14-336 --local-dir ./results/pretrained_models/openai/clip-vit-large-patch14-336
hf download Jiayi-Pan/Tiny-Vicuna-1B --local-dir ./results/pretrained_models/Jiayi-Pan/Tiny-Vicuna-1B
# Pretrain
bash ./tools/pretrain/llava-tiny_vicuna_1b-clip_vit_large_patch14_336-blip_laion_cc_sbu_558k.sh
# Finetune
bash ./tools/finetune/llava-tiny_vicuna_1b-clip_vit_large_patch14_336-llava_v1_5_mix665k.sh
```

- Qwen1.5-0.5B-Chat + CLIP-ViT-B-32-224px

```shell
hf download openai/clip-vit-base-patch32 --local-dir ./results/pretrained_models/openai/clip-vit-base-patch32
hf download Qwen/Qwen1.5-0.5B-Chat --local-dir ./results/pretrained_models/Jiayi-Pan/Tiny-Vicuna-1B
# Pretrain
bash ./tools/pretrain/llava-qwen1.5_0.5b_chat-clip_vit_base_patch32-blip_laion_cc_sbu_558k.sh
# Finetune
bash ./tools/finetune/llava-qwen1.5_0.5b_chat-clip_vit_base_patch32-llava_v1_5_mix665k.sh
```
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