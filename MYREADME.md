# Tiktoken Modified
This fork version is modified version of tiktoken for training from scratch, encoding support for Bangla language and so on.

## Installation
Install the dependencies using `pip install tiktoken blobfile loguru`

### Install from source
Installing from source requires rust compiler and rust setup tools.
- Install rust:

    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    . "$HOME/.cargo/env"
    ```
- Install rust setup tools by `pip install setuptools-rust`
- Now install from source: `pip install -e .`

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

