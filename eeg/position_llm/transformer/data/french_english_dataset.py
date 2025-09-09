from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from tokenizer.tokenizer import Tokenizer

class DummyDataset(Dataset):
    def __init__(self, train=True):
        self.train = train

    def __len__(self):
        return 100
    
    def __getitem__(self, idx):
        return torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4])

class FrenchEnglishDataset(Dataset):
    def __init__(self, translation_csv: Path, tokenizer: Tokenizer, train=True):

        self.lines = []
        self.train = train
        with open(translation_csv, "r") as file:
            file.readline()
            for line in tqdm(file):
                self.lines.append(line.strip().split("\t"))

        self.tokenizer = tokenizer

    def __len__(self):
        if self.train:
            return int(len(self.lines) * 1.0)
        else:
            return int(len(self.lines) * 0.1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.lines[idx][0][0] == "\"":
            line = self.lines[idx][0].split("\",")
        else:
            line = self.lines[idx][0].split(",")

        english_sentence, french_sentence = line[0], line[1]

        french_encoded = self.tokenizer.encode(french_sentence)
        english_encoded = self.tokenizer.encode(english_sentence)

        return french_encoded, english_encoded

    def collate_fn(batch):
        inputs = [torch.tensor(b[0]) for b in batch]
        targets = [torch.tensor(b[1]) for b in batch]

        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

        return inputs, targets
