import torch



def create_matrices():
    x = torch.empty(5, 3)

    print(x)

    x = torch.rand(4, 6)

    print(x)

    x = torch.zeros(3, 4, dtype=torch.long)

    print(x)

    # construct a tensor from data (array)
    x1 = torch.tensor([4, 5.9])
    print(x1)

    x2 = x.new_ones(3, 3, dtype=torch.double)
    print(x2)

    z = torch.rand_like(x2, dtype=torch.float)  # creates a random matrix with the shape of x2
    print(z)
    print(z.size())


def operators():
    x = torch.rand((3, 4), dtype=torch.float32)
    y = torch.rand((3, 4), dtype=torch.float32)

    print(x + y)

    z = torch.rand((4, 5), dtype=torch.float32)

    print(torch.mm(x, z))  # matrix multiplication

    #providing output matrix
    result = torch.empty((3, 4))
    torch.add(x, y, out=result)
    print(result)


def indexing():
    x = torch.rand((3, 4), dtype=torch.float32)
    print(x)

    print(x[1, :])
    print(x[:, 0])
    print(x[1:3, 1:3])


def resize_reshape():
    x = torch.rand((3, 4), dtype=torch.float32)
    y = x.view(12)
    print(y)

    z = x.view((-1, 2))
    print(z)

    z1 = x.view((2, -1))
    print(z1)


def get_the_values():
    x = torch.rand(1, dtype=torch.float32)

    print(x)

    print(x.item())

    print(x[0])


def conversion_with_numpy():
    x = torch.rand((3, 4), dtype=torch.float32)
    a = x.numpy()

    print(x)
    print(a)


    x1 = torch.from_numpy(a)
    print(x1)


if __name__ == "__main__":
    #create_matrices()
    #operators()
    #indexing()
    #resize_reshape()
    #get_the_values()
    conversion_with_numpy()
