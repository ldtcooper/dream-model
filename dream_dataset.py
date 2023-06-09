import torch
from torch.utils.data import Dataset


class DreamsDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()

        self.input_ids = []
        self.attn_masks = []
        self.sot_token = "<|startoftext|>"
        self.eot_token = "<|endoftext|>"

        for row in data:
            encodings_dict = tokenizer(
                f'{self.sot_token}{row}{self.eot_token}', truncation=True, max_length=1024, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(
                encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
