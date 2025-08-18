# Evaluation

## VQAv2

1. Download the data to be evaluated.

```shell
wget http://images.cocodataset.org/zips/test2015.zip ./results/eval/vqav2/
unzip ./results/eval/vqav2/test2015.zip -d ./results/eval/vqav2/
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip ./results/eval/vqav2/
unzip ./results/eval/vqav2/v2_Questions_Test_mscoco.zip -d ./results/eval/vqav2/
```

2. Convert the original conversation format to the jsonl format supported by the evaluation model.

```shell
python ./tools/eval/convert_vqav2_to_jsonl.py -i ./results/eval/vqav2/v2_OpenEnded_mscoco_test2015_questions.json
python ./tools/eval/convert_vqav2_to_jsonl.py -i ./results/eval/vqav2/v2_OpenEnded_mscoco_test-dev2015_questions.json
```

3. Evaluate the model.

```shell
python ./tools/eval/eval_vqa_model.py \
       --model-path ./results/stage2_finetune/llava-qwen1.5_0.5b_chat-clip_vit_large_patch14_336-stage2_coco_data \
       --image-folder ./results/eval/vqav2/test2015 \
       --question-file ./results/eval/vqav2/v2_OpenEnded_mscoco_test-dev2015_questions.jsonl \
       --answer-file ./results/eval/vqav2/v2_OpenEnded_mscoco_test-dev2015_questions/qwen1.5-0.5b-chat.jsonl \
       --conv-mode qwen1_5
```