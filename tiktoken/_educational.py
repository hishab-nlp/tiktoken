"""This is an educational implementation of the byte pair encoding algorithm."""
import collections
from typing import Optional

import regex

import tiktoken

from tqdm.auto import tqdm
from collections import Counter
from itertools import chain
import pickle
import os

import numpy as np

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data,path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def count_and_multiply(tuples_list, multipliers):
    # Count the frequency of each tuple
    frequency_counter = Counter(tuples_list)
    
    # Multiply the frequency with the corresponding multiplier
    result = {tup: count * multipliers[i] for i, (tup, count) in enumerate(frequency_counter.items())}
    
    return result

class SimpleBytePairEncoding:
    def __init__(self, *, pat_str: str, mergeable_ranks: dict[bytes, int]) -> None:
        """Creates an Encoding object."""
        # A regex pattern string that is used to split the input text
        self.pat_str = pat_str
        # A dictionary mapping token bytes to their ranks. The ranks correspond to merge priority
        self.mergeable_ranks = mergeable_ranks

        self._decoder = {token: token_bytes for token_bytes, token in mergeable_ranks.items()}
        self._pat = regex.compile(pat_str)

    def encode(self, text: str, visualise: Optional[str] = "colour") -> list[int]:
        """Encodes a string into tokens.

        >>> enc.encode("hello world")
        [388, 372]
        """
        # Use the regex to split the text into (approximately) words
        words = self._pat.findall(text)
        tokens = []
        for word in words:
            # Turn each word into tokens, using the byte pair encoding algorithm
            word_bytes = word.encode("utf-8")
            word_tokens = bpe_encode(self.mergeable_ranks, word_bytes, visualise=visualise)
            tokens.extend(word_tokens)
        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decodes a list of tokens into bytes.

        >>> enc.decode_bytes([388, 372])
        b'hello world'
        """
        return b"".join(self._decoder[token] for token in tokens)

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of tokens into a string.

        Decoded bytes are not guaranteed to be valid UTF-8. In that case, we replace
        the invalid bytes with the replacement character "�".

        >>> enc.decode([388, 372])
        'hello world'
        """
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

    def decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]:
        """Decodes a list of tokens into a list of bytes.

        Useful for visualising how a string is tokenised.

        >>> enc.decode_tokens_bytes([388, 372])
        [b'hello', b' world']
        """
        return [self._decoder[token] for token in tokens]

    @staticmethod
    def train(training_data: str, vocab_size: int, pat_str: str):
        """Train a BPE tokeniser on some data!"""
        mergeable_ranks = bpe_train(data=training_data, vocab_size=vocab_size, pat_str=pat_str)
        return SimpleBytePairEncoding(pat_str=pat_str, mergeable_ranks=mergeable_ranks)

    @staticmethod
    def from_tiktoken(encoding):
        if isinstance(encoding, str):
            encoding = tiktoken.get_encoding(encoding)
        return SimpleBytePairEncoding(
            pat_str=encoding._pat_str, mergeable_ranks=encoding._mergeable_ranks
        )


def bpe_encode(
    mergeable_ranks: dict[bytes, int], input: bytes, visualise: Optional[str] = "colour"
) -> list[int]:
    parts = [bytes([b]) for b in input]
    while True:
        # See the intermediate merges play out!
        if visualise:
            if visualise in ["colour", "color"]:
                visualise_tokens(parts)
            elif visualise == "simple":
                print(parts)

        # Iterate over all pairs and find the pair we want to merge the most
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # If there were no pairs we could merge, we're done!
        if min_rank is None:
            break
        assert min_idx is not None

        # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]

    if visualise:
        print()

    tokens = [mergeable_ranks[part] for part in parts]
    return tokens



def bpe_train(
    data: str, vocab_size: int, pat_str: str, visualise: Optional[str] = "colour", resume: str = "no",word_select_threshold: float = 0.97, checkpoint_path: str = "checkpoints", save_every_n_steps: int = 2048
) -> dict[bytes, int]:
    
    os.makedirs(checkpoint_path,exist_ok=True)
    
    if resume=="no":
        # First, add tokens for each individual byte value
        if vocab_size < 2**8:
            raise ValueError("vocab_size must be at least 256, so we can encode all bytes")
        ranks = {}
        for i in range(2**8):
            ranks[bytes([i])] = i
        
        save_pkl(ranks,os.path.join(checkpoint_path,"ranks.pkl"))

        # Splinter up our data into lists of bytes
        # data = "Hello world"
        # words = [
        #     [b'H', b'e', b'l', b'l', b'o'],
        #     [b' ', b'w', b'o', b'r', b'l', b'd']
        # ]

        #############
        words_unique = collections.Counter()
        words = regex.findall(pat_str, data)
        
        for word in tqdm(words):
            words_unique[tuple([bytes([b]) for b in word.encode("utf-8")])]+= 1
        
        words = [list(w) for w in list(words_unique.keys())]
        words_multiplier = list(words_unique.values())
        #############
        
        save_pkl(words,os.path.join(checkpoint_path,"words_init.pkl"))
        
        save_pkl(words_multiplier,os.path.join(checkpoint_path,"words_multiplier_init.pkl"))
        
        
        words_multiplier_sorted=sorted(words_multiplier,reverse=True)
        words_multiplier_sorted_cumsum = np.cumsum(words_multiplier_sorted)
        words_multiplier_sorted_cumsum_norm=words_multiplier_sorted_cumsum/words_multiplier_sorted_cumsum[-1]
        targeted_idx = len([i for i in words_multiplier_sorted_cumsum_norm if i<word_select_threshold])
        threshold_frequency = words_multiplier_sorted[targeted_idx]
        
        words = [i for c,i in enumerate(words) if words_multiplier[c]>threshold_frequency]
        words_multiplier = [i for i in words_multiplier if i>threshold_frequency]
       
        save_pkl(words,os.path.join(checkpoint_path,"words.pkl"))
        save_pkl(words_multiplier,os.path.join(checkpoint_path,"words_multiplier.pkl"))
        
        change_index = collections.Counter()
        all_pairs_ = []
        all_multipliers_ = []
        
        save_pkl(change_index,os.path.join(checkpoint_path,"change_index.pkl"))
        save_pkl(all_pairs_,os.path.join(checkpoint_path,"all_pairs_.pkl"))
        save_pkl(all_multipliers_,os.path.join(checkpoint_path,"all_multipliers_.pkl"))
        
    else:
        
        ranks=load_pkl(os.path.join(checkpoint_path,"ranks.pkl"))

        words=load_pkl(os.path.join(checkpoint_path,"words.pkl"))
        
        words_multiplier=load_pkl(os.path.join(checkpoint_path,"words_multiplier.pkl"))
   
        change_index=load_pkl(os.path.join(checkpoint_path,"change_index.pkl"))

        all_pairs_=load_pkl(os.path.join(checkpoint_path,"all_pairs_.pkl"))

        all_multipliers_=load_pkl(os.path.join(checkpoint_path,"all_multipliers_.pkl"))
  
        
    # Now, use our data to figure out which merges we should make
    
    print("Total Words: ",len(words))
    
    
    try:
        for _ in tqdm(range(len(ranks),vocab_size)):
            for c,piece in enumerate(words):
                if len(change_index)==0:
                    all_pairs_.append(list(zip(piece[:-1], piece[1:])))
                    all_multipliers_.append([words_multiplier[c]]*(len(piece)-1))
                else:
                    if change_index[c]==-1:
                        all_pairs_[c]=list(zip(piece[:-1], piece[1:]))
                        all_multipliers_[c]=[words_multiplier[c]]*(len(piece)-1)

                                            
            all_pairs=list(chain.from_iterable(all_pairs_))
            all_multipliers=list(chain.from_iterable(all_multipliers_))
            
            stats = count_and_multiply(all_pairs, all_multipliers)
            
            most_common_pair = max(stats, key=lambda x: stats[x])
            token_bytes = most_common_pair[0] + most_common_pair[1]
            
            # if ranks.get(token_bytes,None) is not None:
            token = len(ranks)
            # Add the new token!
            ranks[token_bytes] = token
            
            

            # Now merge that most common pair in all the words. That is, update our training data
            # to reflect our decision to make that pair into a new token.
            
            change_index = collections.Counter()
            for cc,word in enumerate(words):
                parent = b']^['+b"[^]".join(word)+b']^['
                child = b"[^]".join(most_common_pair)

                flag=0
                if child in parent:
                    if b"[^]"+child+b"[^]" in parent:
                        parent = parent.replace(b"[^]"+child+b"[^]",b"[^]"+token_bytes+b"[^]")
                        flag=1
                        
                    if b"]^["+child+b"[^]" in parent:
                        parent = parent.replace(b"]^["+child+b"[^]",b"]^["+token_bytes+b"[^]")
                        flag=1
                    
                    if b"[^]"+child+b"]^[" in parent:
                        parent = parent.replace(b"[^]"+child+b"]^[",b"[^]"+token_bytes+b"]^[")
                        flag=1
                    
                    if b"]^["+child+b"]^[" in parent:
                        parent = parent.replace(b"]^["+child+b"]^[",b"]^["+token_bytes+b"]^[")
                        flag=1
                    
                if flag==1:
                    words[cc]=parent[3:-3].split(b"[^]")
                    change_index[cc]=-1

            # See the intermediate merges play out!
            if visualise:
                print(f"The current most common pair is {most_common_pair[0]} + {most_common_pair[1]}")
                try:
                    print(f"So we made {token_bytes} our {len(ranks)}th token")
                except:
                    print(f"So we made {token_bytes.decode()} our {len(ranks)}th token")
                
                if visualise in ["colour", "color"]:
                    print("Now the first fifty words in our training data look like:")
                    visualise_tokens([token for word in words[:50] for token in word])
                elif visualise == "simple":
                    print("Now the first twenty words in our training data look like:")
                    for word in words[:20]:
                        print(word)
                print("\n")
                

            if len(ranks)%save_every_n_steps==0:
                
                save_pkl(ranks,os.path.join(checkpoint_path,"ranks.pkl"))
            
                save_pkl(words,os.path.join(checkpoint_path,"words.pkl"))

                save_pkl(words_multiplier,os.path.join(checkpoint_path,"words_multiplier.pkl"))
                
                save_pkl(change_index,os.path.join(checkpoint_path,"change_index.pkl"))

                save_pkl(all_pairs_,os.path.join(checkpoint_path,"all_pairs_.pkl"))

                save_pkl(all_multipliers_,os.path.join(checkpoint_path,"all_multipliers_.pkl"))

    except Exception as e:
        print(e) 
        
    ranks = dict(zip(list(ranks.keys()),[i for i in range(len(list(ranks.keys())))]))  
         
    return ranks


def visualise_tokens(token_values: list[bytes]) -> None:
    background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]
    # If token boundaries do not occur at unicode character boundaries, it's unclear how best to
    # visualise the token. Here, we'll just use the unicode replacement character to represent some
    # fraction of a character.
    unicode_token_values = [x.decode("utf-8", errors="replace") for x in token_values]

    running_length = 0
    last_color = None
    for token in unicode_token_values:
        color = background[running_length % len(background)]
        if color == last_color:
            color = background[(running_length + 1) % len(background)]
            assert color != last_color
        last_color = color
        running_length += len(token)
        print(color + token, end="")
    print("\u001b[0m")


def train_simple_encoding():
    gpt2_pattern = (
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    with open(__file__, "r") as f:
        data = f.read()

    enc = SimpleBytePairEncoding.train(data, vocab_size=600, pat_str=gpt2_pattern)

    print("This is the sequence of merges performed in order to encode 'hello world':")
    tokens = enc.encode("hello world")
    assert enc.decode(tokens) == "hello world"
    assert enc.decode_bytes(tokens) == b"hello world"
    assert enc.decode_tokens_bytes(tokens) == [b"hello", b" world"]

    return enc
