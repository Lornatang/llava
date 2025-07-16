# llava

LLaVA: Large Language and Vision Assistant. From data to deployment.

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

Please place the downloaded files into the datasets directory according to the following format requirements.

```txt
- datasets
    - blip_laion_cc_sbu_558k
        - blip_laion_cc_sbu_558k.json
        - blip_laion_cc_sbu_558k_meta.json
        - images
            - 000000000000.jpg
            - 000000000001.jpg
            ...
```

### Pre-training

We follow the original author's training methods and training hyperparameters, and use the DeepSpeed ZeRO-2 tool to complete model pre-training.

#### 1. Manually download CLIP model weights

```bash
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./results/pretrained_models/openai/clip-vit-large-patch14 
```

#### 2. Run pre-training script

```bash
bash ./tools/pretrain.sh
```

More details about train please see: [pretrain.sh](./tools/pretrain.sh)


