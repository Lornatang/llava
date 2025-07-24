# llava

LLaVA: Large Language and Vision Assistant. From data to deployment.

## Install

**Please note that this is only supported on Linux systems.**

**1. Clone repository**

```bash
git clone https://github.com/Lornatang/llava.git
cd llava
```

**2. Install Package**

```bash
conda create -n llava python=3.11.13 -y
conda activate llava
pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip3 install flash_attn-2.8.1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip3 install -e .
```

## Train

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
- [textvqa train_val_images](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing)
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

### Pre-training

We follow the original author's training methods and training hyperparameters, and use the DeepSpeed ZeRO-3 tool to complete model pre-training.

#### 1. Manually download pretrained model weights

```bash
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir ./results/pretrained_models/openai/clip-vit-large-patch14-336
huggingface-cli download lmsys/vicuna-13b-v1.5 --local-dir ./results/pretrained_models/lmsys/vicuna-13b-v1.5
```

The following models can be downloaded for quick experimentation.

- [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Tiny-Vicuna-1B](https://huggingface.co/Jiayi-Pan/Tiny-Vicuna-1B) (From TinyLlama-1.1B-Chat-v1.0 fine-tuned)

#### 2. Run pretrain script

```bash
bash ./tools/pretrain.sh
```

More details about train please see: [pretrain.sh](./tools/pretrain.sh)

### Fine-tuning

We follow the original author's training methods and training hyperparameters, and use the DeepSpeed ZeRO-3 tool to complete model fine-tuning.

#### 1. Run finetune script

```bash
bash ./tools/finetune.sh
```

More details about train please see: [finetune.sh](./tools/finetune.sh)

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