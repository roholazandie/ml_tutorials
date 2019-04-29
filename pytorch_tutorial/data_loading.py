import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


if __name__ == "__main__":
    input_size = 5
    data_size = 100
    batch_size = 10
    random_dataset = RandomDataset(input_size, data_size)
    rand_loader = DataLoader(random_dataset, batch_size=batch_size, shuffle=True, num_workers=3)

    for data in rand_loader:
        print(data)
