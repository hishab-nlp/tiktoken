# Tiktoken Modified
This fork version is modified version of tiktoken for training from scratch, encoding support for Bangla language and so on.

## Installation
Install the dependencies using `pip install -r requirements.txt`

## Train
To train tiktoken from scratch

```bash
cd scripts
python train_bpe.py \
--data_path "path/mytextfiles" \
--vocab_size 1000 \
--pretokenizer_pattern "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w\p{N}]?\w+|\p{N}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" \
--output_file "path/mytokenizer.model" \
--visualise "simple"
```

