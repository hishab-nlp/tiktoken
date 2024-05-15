# Tiktoken Modified
This fork version is modified version of tiktoken for training from scratch, encoding support for Bangla language and so on.

## Installation
Install the dependencies using `pip install tiktoken blobfile loguru`

## Train
To train tiktoken from scratch check the scripts and update the pretokenized pattern and then run:

```bash
cd scripts
python train_bpe.py \
--data_path "../data" \
--vocab_size 500 \
--output_file "../data/mytokenizer.model" \
--visualise "simple"
```

