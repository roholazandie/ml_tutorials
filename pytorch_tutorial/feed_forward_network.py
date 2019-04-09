import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5) #nSamples x nChannels x Height x Width
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y = Wx+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        # Max pooling over 2x2 window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


if __name__ == "__main__":
    net = Net()
    print(net)

    params = list(net.parameters())
    #print(params)

    input_tensor = torch.randn(1, 1, 32, 32)
    output = net(input_tensor)
    print(output)

    #Zero the gradient buffers of all parameters and backprops with random gradients:
    #net.zero_grad()
    #output.backward(torch.randn(1, 10))

    target = torch.randn(10)  # a dummy target, for example
    mse_loss = nn.MSELoss()

    loss = mse_loss(output, target)
    print(loss)

    print(loss.grad_fn) #MSE
    print(loss.grad_fn.next_functions[0][0]) #Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) #Relu


    net.zero_grad()
    print(net.conv1.bias.grad)
    loss.backward()
    print(net.conv1.bias.grad) #conv1.bias.grad after backward



    # stochastic gradient descent
    # weight = weight - learning_rate * gradient

    learning_rate = 0.01
    # for f in net.parameters():
    #     f.data.sub_(f.grad.data * learning_rate)

    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    #training loop
    optimizer.zero_grad()
    output = net(input_tensor)
    loss = mse_loss(output, target)
    loss.backward()
    optimizer.step()