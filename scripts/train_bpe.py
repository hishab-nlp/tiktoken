import glob
from tqdm import tqdm
from typing import List
from loguru import logger
from argparse import ArgumentParser
from tiktoken.load import dump_tiktoken_bpe
from tiktoken._educational import bpe_train

args = ArgumentParser()
args.add_argument("--data_path", type=str, help="text files root path, files extension must be .txt")
args.add_argument("--vocab_size", type=int, default=1000, help="vocab size")
args.add_argument("--output_file", type=str, help="output file name with path")
args.add_argument("--visualise", type=str, help="visualize parameter to visualize token while training. choose from ['color', 'colour', 'simple']")
parser = args.parse_args()

# this pattern is working fine for Bangla and English text
pretokenized_pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w\p{N}]?\w+|\p{N}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

def load_text(files: List[str]):
    texts = ""
    for file in tqdm(files):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            texts += text + '\n'
    return texts

def main():
    files = glob.glob(parser.data_path + "/**/*.txt", recursive=True)
    logger.info(f"total text files found: {len(files)}")
    logger.info("Loading all text into memory.......")
    data = load_text(files)
    logger.info(f"first 100 chars from data: {data[:100]}")
    logger.info("Text loading in memory done!")
    logger.info("Training start..............")
    mergeable_ranks = bpe_train(
        data=data,
        vocab_size=parser.vocab_size,
        pat_str=pretokenized_pattern,
        visualise=parser.visualise
    )
    logger.info("Training has been completed!")

    dump_tiktoken_bpe(
        mergeable_ranks,
        parser.output_file
    )
    logger.info(f"tokenizer dumped at {parser.output_file}")


if __name__ == "__main__":
    main()