import regex
from typing import List
import multiprocessing as mp

def convert_word_to_bytes(word: str) -> List[bytes]:
    return [bytes([b]) for b in word.encode("utf-8")]

def convert_data_to_bytes(data: str, pat_str: str) -> List[List[bytes]]:
    words = regex.findall(pat_str, data)
    print(words[:100])
    pool = mp.Pool(mp.cpu_count())
    words_bytes = pool.map(convert_word_to_bytes, words)
    pool.close()
    pool.join()
    return words_bytes

# Example usage
if __name__ == "__main__":
    with open('data/sample_text.txt') as f:
        data = f.read()

    # data = """To make the processing of converting words to lists of bytes faster using multiprocessing, you can split the data into chunks and process each chunk in parallel. Here's an updated version of the code that uses the multiprocessing module to achieve this:"""
    # pat_str = (
    #     r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    # )
    pat_str = (
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w\p{N}]?\w+|\p{N}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        # r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w\p{N}]?\w+|\p{N}+| ?[^\s\w\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
    
    words = convert_data_to_bytes(data, pat_str)
    print(words[:10])  # Print first 10 results for verification
