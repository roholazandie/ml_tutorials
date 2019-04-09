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


class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output


if __name__ == "__main__":
    # Parameters and DataLoaders
    input_size = 5
    output_size = 2

    batch_size = 30
    data_size = 100

    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                             batch_size=batch_size, shuffle=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #the core of this tutorial: using nn.DataParallel to use multiple gpus
    model = Model(input_size, output_size)
    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), "GPU's")
        model = nn.DataParallel(model)

    model.to(device)

    #run the model
    for data in rand_loader:
        input_ = data.to(device)
        output = model(input_)
        print("input size", input_.size(), "output size", output.size())
