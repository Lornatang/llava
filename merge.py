# Copyright (c) AlphaBetter. All rights reserved.
import argparse
from llava.utils.checkpoint import load_pretrained


def merge_lora(args):
    tokenizer, model, image_processor, context_len = load_pretrained(args.model_path, args.model_base, device_map='cpu')

    # model = model.merge_and_unload()  # This can take several minutes on cpu
    model._hf_peft_config_loaded = False

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)