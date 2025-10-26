import re

###
## Functions
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
## Tests
###

# Not considering whitespaces and punctuation
def test_simple_re():
    text = "Hello, world. This, is a test"
    result = re.split(r'(\s)', text)
    assert result == ['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test'] 

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
    assert result == ['Hello', ',', 'world', '.', 'This', '--', ',', 'is', 'a', 'test']

def test_count_token():
    text = read_dataset()
    result = tokenize(text)
    print(result[:100])
    assert len(result) == 4690