import glob
import time
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
args.add_argument("--resume", type=str, default="no")
args.add_argument("--word_select_threshold", type=float, default=0.98)
args.add_argument("--checkpoint_path", type=str, default="checkpoints")
args.add_argument("--save_every_n_steps", type=int, default=2048)

parser = args.parse_args()

pretokenized_pattern = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[\r\n\t]+|[^\w\p{N}\p{L}\s]+|\s+[^\w\p{N}\p{L}\s]+|\s*\n|\s*\t|\s*\r|(?<=\p{N})(?=\p{L})|(?<=\p{L})(?=\p{N})|\s{2,}(?=\s\w)|\s*\w+|\s*\p{N}+|\s*\p{N}{1,3}"""

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
    if parser.resume=="no":
        data = load_text(files)
    else:
        data = ""
    logger.info(f"first 100 chars from data: {data[:100]}")
    logger.info("Text loading in memory done!")
    logger.info("Training start..............")
    start_time = time.time()

    mergeable_ranks = bpe_train(
        data=data,
        vocab_size=parser.vocab_size,
        pat_str=pretokenized_pattern,
        visualise=parser.visualise,
        resume = parser.resume,
        word_select_threshold=parser.word_select_threshold,
        checkpoint_path=parser.checkpoint_path,
        save_every_n_steps=parser.save_every_n_steps
    )
    logger.info("Training has been completed!")
    logger.info(f"total training time: {(time.time() - start_time) / 60} min")

    dump_tiktoken_bpe(
        mergeable_ranks,
        parser.output_file
    )
    logger.info(f"tokenizer dumped at {parser.output_file}")


if __name__ == "__main__":
    main()