import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
'''

In PyTorch, the learnable parameters (i.e. weights and biases)
of an torch.nn.Module model are contained in the model’s parameters 
(accessed with model.parameters()). A state_dict is simply a Python dictionary object
that maps each layer to its parameter tensor.
Note that only layers with learnable parameters (convolutional layers, linear layers, etc.) 
and registered buffers (batchnorm’s running_mean) have entries in the model’s state_dict. 
Optimizer objects (torch.optim) also have a state_dict,
which contains information about the optimizer’s state, as well as the hyperparameters used.
'''

# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# print model's state dict
print("Model's state_dict")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# print optimizer's state dict
print("Optimizer's state_dict")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


#############
#save model to disk
def save_load_state_dict(model):
    torch.save(model.state_dict(), "../data/model.pt")

    model = TheModelClass() # a new instance
    model.load_state_dict(torch.load("../data/model.pt"))
    #you must call model.eval() to set dropout and batch normalization layers
    # to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model.eval()

def save_load_entire_model(model):
    '''
    This save/load process uses the most intuitive syntax and involves the least amount of code.
    Saving a model in this way will save the entire module using Python’s pickle module.
    The disadvantage of this approach is that the serialized data is bound to
    the specific classes and the exact directory structure used when the model is saved.
    '''
    torch.save(model, "../data/model.pt")
    # Model class must be defined somewhere
    model = torch.load("../data/model.pt")
    model.eval()

def _correct_repeated_sentences(text):
    import re
    from itertools import combinations

    split_text = re.split(r' *[\?\.\!][\'"\)\]]* *', text)
    matches = list(re.finditer(r' *[\?\.\!][\'"\)\]]* *', text))

    drop = []
    for i, j in combinations(range(len(split_text)), 2):
        if split_text[j] and split_text[j] in split_text[i]:
            drop.append(j)
    drop = set(drop)
    drop = sorted(drop, reverse=True)

    for d in drop:
        split_text.pop(d)
        matches.pop(d)

    original_text = ''

    for s, m in zip(split_text, matches):
        original_text += s + m.group()
    if len(split_text) > len(matches):
        original_text += split_text[-1]
    return original_text


if __name__ == "__main__":
    model = TheModelClass()
    #save_load_state_dict(model)
    #save_load_entire_model(model)
    res = _correct_repeated_sentences("I like it. I like it!")
    print(res)