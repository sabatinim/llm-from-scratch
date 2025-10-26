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


###
# Tests
###

# Not considering whitespaces and punctuation
def test_simple_re():
    text = "Hello, world. This, is a test"
    result = re.split(r'(\s)', text)
    assert result == ['Hello,', ' ', 'world.', ' ',
                      'This,', ' ', 'is', ' ', 'a', ' ', 'test']

# Here remove whitespaces


def test_white_spaces_re():
    text = "Hello, world. This, is a test"
    result = re.split(r'([,.] | \s)', text)
    result = [r.strip() for r in result if r.strip()]
    assert result == ['Hello', ',', 'world', '.', 'This', ',', 'is a test']

# Final tokenizer


def test_final_tokenizer():
    text = "Hello, world. This--, is a test"
    result = tokenize(text)
    assert result == ['Hello', ',', 'world', '.',
                      'This', '--', ',', 'is', 'a', 'test']


def test_count_token():
    text = read_dataset()
    result = tokenize(text)
    print(result[:100])
    assert len(result) == 4690

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

def test_simple_tokenizer_encode():
    tokenizer = SimpleTokenizerV1(vocab=tokenize(read_dataset()))

    encoded = tokenizer.encode(
        """It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.""")
    assert encoded == [4409, 4680, 4681, 4647, 3870, 4606, 4508, 4673, 4321, 4223, 4673, 4689, 4478, 4688, 2923, 4383, 4641, 1968, 2180, 4688]

def test_simple_tokenizer_decode():
    tokenizer = SimpleTokenizerV1(vocab=tokenize(read_dataset()))
    ids = [4409, 4680, 4681, 4647, 3870, 4606, 4508, 4673, 4321, 4223, 4673, 4689, 4478, 4688, 2923, 4383, 4641, 1968, 2180, 4688]
    decoded = tokenizer.decode(ids)
    
    assert decoded == """It' s the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
