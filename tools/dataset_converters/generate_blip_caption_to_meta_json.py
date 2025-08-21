# Copyright (c) AlphaBetter. All rights reserved.
import ujson
from pathlib import Path

import torch
from PIL import Image
from transformers import BitsAndBytesConfig, Blip2Processor, Blip2ForConditionalGeneration


def generate_meta_with_blip_incremental(root_dir: Path, output_file: Path) -> None:
    # Load BLIP model
    processor = Blip2Processor.from_pretrained("./results/pretrained_models/Salesforce/blip2-opt-2.7b", use_fast=True)
    kwargs = {
        "quantization_config": BitsAndBytesConfig(
            load_in_8bit=True,
        ),
        "device_map": "auto",
        "offload_folder": "offload"
    }
    model = Blip2ForConditionalGeneration.from_pretrained("./results/pretrained_models/Salesforce/blip2-opt-2.7b", **kwargs)

    # Load existing results (if any)
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing_records = {rec["id"]: rec for rec in ujson.load(f)}
        print(f"🔄 Loaded {len(existing_records)} existing records.")
    else:
        existing_records = {}
        print("🆕 No existing output, start fresh.")

    # Iterate subfolders
    for subdir in sorted(root_dir.iterdir()):
        if not subdir.is_dir():
            continue

        for json_file in subdir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                meta = ujson.load(f)

            img_file = json_file.with_suffix(".jpg")
            if not img_file.exists():
                print(f"⚠️ Image not found for {json_file}")
                continue

            sample_id = meta.get("id") or json_file.stem

            # If already exists with blip_caption, skip
            if sample_id in existing_records and existing_records[sample_id].get("blip_caption"):
                continue

            record = {
                "id": sample_id,
                "image": str(img_file.relative_to(root_dir)),
                "caption": meta.get("caption"),
                "url": meta.get("url"),
            }

            # If record already exists (but missing blip_caption), update instead of duplicate
            if sample_id in existing_records:
                record.update(existing_records[sample_id])

            # Generate BLIP caption
            try:
                raw_image = Image.open(img_file).convert("RGB")
                inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
                output = model.generate(**inputs, max_new_tokens=50)
                blip_caption = processor.decode(output[0], skip_special_tokens=True).strip()
                print(blip_caption)
                record["blip_caption"] = blip_caption
            except Exception as e:
                print(f"❌ Failed to process {img_file}: {e}")
                record["blip_caption"] = None

            existing_records[sample_id] = record
            # Write back incrementally (避免断电/中断丢失)
            with open(output_file, "w", encoding="utf-8") as f:
                ujson.dump(list(existing_records.values()), f, ensure_ascii=False, indent=2, escape_forward_slashes=False)


    print(f"✅ Finished! Total {len(existing_records)} records saved to {output_file}")


if __name__ == "__main__":
    root_dir = Path("/mnt/data/larry/datasets/open_data/sbucaptions")  # 数据集的根目录
    output_file = Path("sbucaptions_meta.json")
    generate_meta_with_blip_incremental(root_dir, output_file)
