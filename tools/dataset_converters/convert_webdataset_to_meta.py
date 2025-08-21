# Copyright Lornatang. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import orjson


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inputs",
        type=str,
        required=True,
        help="Root directory containing subfolders to process.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--status-file",
        default="processed_folders.json",
        type=str,
        help="Path to the JSON file storing processed folder names. Defaults to ``processed_folders.json``.",
    )
    parser.add_argument(
        "--batch-size",
        default=10,
        type=int,
        help="Number of folders to process per batch. Defaults to 10.",
    )
    return parser.parse_args()


def load_processed_folders(status_file: Union[Path, str]) -> Set[Any]:
    status_file = Path(status_file)
    if status_file.exists():
        with status_file.open("rb") as f:
            return set(orjson.loads(f.read()))
    return set()


def save_processed_folders(status_file: Union[Path, str], processed: Set[Any]) -> None:
    status_file = Path(status_file)
    with status_file.open("wb") as f:
        f.write(orjson.dumps(list(processed), option=orjson.OPT_INDENT_2))


def load_existing_results(output_file: Union[Path, str]) -> List[Any]:
    output_file = Path(output_file)
    if output_file.exists() and output_file.stat().st_size > 0:
        with output_file.open("rb") as f:
            return orjson.loads(f.read())
    return []


def process_folder(folder_path: Union[Path, str]) -> List[Dict[str, Any]]:
    folder_path = Path(folder_path)
    results: List[Dict[str, Any]] = []
    json_files: List[Path] = [entry for entry in folder_path.iterdir() if entry.is_file() and entry.suffix == ".json"]
    if not json_files:
        return results

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, folder_path, json_file) for json_file in json_files]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)
    return results


def process_single_file(folder_path: Union[Path, str], json_file: Union[Path, str]) -> Optional[Dict[str, Any]]:
    try:
        folder_path = Path(folder_path)
        json_file = Path(json_file)
        prefix = json_file.stem
        image_file = Path(folder_path, f"{prefix}.jpg")
        if not image_file.exists():
            return None
        with json_file.open("rb") as f:
            data = orjson.loads(f.read())
        return {
            "id": data.get("key", ""),
            "image": f"{folder_path.name}/{image_file.name}",
            "caption": data.get("caption", ""),
            "blip_caption": data.get("blip_caption", ""),
            "url": data.get("url", "")
        }
    except Exception as e:
        with Path("error.log").open("a", encoding="utf-8") as err_f:
            err_f.write(f"Failed processing {json_file}: {str(e)}.\n")
        return None


def generate_meta_json_from_webdataset(
        root_dir: Union[Path, str],
        output_file: Union[Path, str],
        status_file: Union[Path, str] = "processed_folders.json",
        batch_size: int = 10,
        max_workers: int = 8
) -> None:
    root_dir, output_file, status_file = Path(root_dir), Path(output_file), Path(status_file)
    processed: Set[Any] = load_processed_folders(status_file)
    all_results: List[Any] = load_existing_results(output_file)
    print(f"Already processed {len(processed)} folders, current total results: {len(all_results)}.")

    all_folders: List[Path] = [entry for entry in root_dir.iterdir() if entry.is_dir() and entry.name not in processed]
    total_folders = len(all_folders)
    print(f"Found {total_folders} folders to process, starting batch processing...")

    for i in range(0, total_folders, batch_size):
        batch = all_folders[i:i + batch_size]
        batch_start = time.time()
        for folder in batch:
            folder_name = folder.name
            print(f"Processing folder {folder_name}...")
            folder_results = process_folder(folder)
            if folder_results:
                all_results.extend(folder_results)
                print(f"Folder {folder_name} completed, added {len(folder_results)} records.")
            processed.add(folder_name)
            save_processed_folders(status_file, processed)

        with output_file.open("wb") as f:
            f.write(orjson.dumps(all_results, option=orjson.OPT_INDENT_2))

        batch_end = time.time()
        processed_count = len(processed)
        progress = processed_count / total_folders * 100
        print(f"Batch {i // batch_size + 1} finished, time {batch_end - batch_start:.2f}s | "
              f"Progress: {processed_count}/{total_folders} ({progress:.1f}%).\n")

    print(f"All folders processed! Total records: {len(all_results)} saved to {output_file}.")


def main() -> None:
    opts = get_opts()

    generate_meta_json_from_webdataset(
        opts.inputs,
        opts.output,
        opts.status_file,
        batch_size=opts.batch_size,
    )


if __name__ == "__main__":
    main()
