# Dataset Converters

## SBU Captions

In order to quickly download this dataset, we refer to the [img2dataset](https://github.com/rom1504/img2dataset) library.

[SBU Captions]([https://www.cs.rice.edu/~vo9/sbucaptions/sbu-captions]) is a large-scale dataset that contains 860K image-text pairs as well as many other meta-attributes to increase the usability to train various models. This dataset is one of the key benchmark datasets.

### Download the metadata

```shel
wget https://www.cs.rice.edu/~vo9/sbucaptions/sbu-captions-all.tar.gz -O ./datasets/sbu-captions-all.tar.gz
tar -xvzf ./datasets/sbu-captions-all.tar.gz -C ./datasets/
```

### Download the images with img2dataset

```shell
img2dataset --url_list ./datasets/sbu-captions-all.json --input_format "json" --url_col "image_urls" --caption_col "captions" --output_format webdataset --output_folder ./datasets/sbucaptions --processes_count 16 --thread_count 64 --image_size 256
```

### Generating image-text alignment data based on [BLIP2](https://arxiv.org/abs/2301.12597v3)

```shell
pip install orjson
python ./tools/dataset_converters/generate_blip_caption_to_json.py -i ./datasets/sbucaptions
python ./tools/dataset_converters/convert_webdataset_to_meta.py -i ./datasets/sbucaptions -o ./datasets/sbucaptions/sbucaptions_meta.json --max-workers 64
```
