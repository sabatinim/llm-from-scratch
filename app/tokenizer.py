from importlib.metadata import version
import tiktoken

import re
from typing import List



###
# Functions
###
def read_dataset():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        return raw_text


def tokenize(text: str):
    regex = r'([,.:;?_!"()\']|--|\s)'

    tokens = re.split(regex, text)

    tokens_no_whitespaces = [r.strip() for r in tokens if r.strip()]

    return tokens_no_whitespaces


##
# Simple Tokenizer implementation
##
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.to_encode = {word: idx for idx, word in enumerate(vocab)}
        self.to_decode = {idx: word for idx, word in enumerate(vocab)}

    def encode(self, text: str) -> List[int]:
        tokens = tokenize(text=text)

        return [self.to_encode[token] for token in tokens]

    def decode(self, ids: List[int])->str:
        text =  " ".join(self.to_decode[id] for id in ids)
        result = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return result


class SimpleTokenizerV2:
    def __init__(self, vocab):
        vocab.extend(["<|endoftext|>", "<|unk|>"])
        self.to_encode = {word: idx for idx, word in enumerate(vocab)}
        self.to_decode = {idx: word for idx, word in enumerate(vocab)}

    def encode(self, text: str) -> List[int]:
        tokens = tokenize(text=text)

        preprocessed = [token if token in self.to_encode else "<|unk|>" for token in tokens ]

        return [self.to_encode[token] for token in preprocessed]

    def decode(self, ids: List[int])->str:
        text =  " ".join(self.to_decode[id] for id in ids)
        result = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return result

class SimpleTokenizerV3:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
    
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    
    def decode(self, ids: List[int])->str:
        return self.tokenizer.decode(ids)