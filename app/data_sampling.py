import torch
from torch.utils.data import Dataset, DataLoader
from app.tokenizer import SimpleTokenizerV3

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

        
def create_dataloader(txt, batch_size, max_length, stride, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = SimpleTokenizerV3()
    dataset = GPTDatasetV1(txt=txt, tokenizer=tokenizer, max_length=max_length, stride=stride)
    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last,
                        num_workers=num_workers)
    return loader
