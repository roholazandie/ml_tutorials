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


def batch_matrix_multiplication():
    batch1 = torch.rand((10, 4, 5))
    batch2 = torch.rand((10, 5, 3))

    result = torch.bmm(batch1, batch2)
    print(result)
    print(result.size())


def any():
    x = torch.ones(10, 10, dtype=torch.uint8)
    res = torch.stack([x, x]).any(dim=-1)
    print(res)


def repeat():
    x = torch.rand((3,1))
    v = x.repeat(3, 2)
    e = x.expand(3, 2)
    print(v)
    print(e)


def scatter():
    #x = torch.rand(2, 5)
    #print(x)
    x = torch.tensor(1)
    v = torch.zeros(3, 5).scatter_(0, torch.tensor([[0, 1, 2, 0, 0],
                                                    [2, 0, 0, 1, 2]]), x)
    print(v)


def split():
    a = torch.randn(50, 80)
    res = a.split(40, 1) # split to two randn(50, 40)
    print(res)

    res = a.split(10, 0) # split to 5 rand(10, 80)
    print(res)


def unsqueeze():
    padding_mask = torch.tensor([0,0,0,1,1,1,1,1,1], dtype=torch.float32)
    res = padding_mask#.unsqueeze(1).unsqueeze(2)
    mat = torch.ones(9, 9, dtype=torch.float32)
    t = mat + res
    print(res)

def gather():
    #https://stackoverflow.com/a/54706716/2736889
    index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    src = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    result = src.gather(0, index)

    print(result)

if __name__ == "__main__":
    #create_matrices()
    #operators()
    #indexing()
    #resize_reshape()
    #get_the_values()
    #conversion_with_numpy()
    #batch_matrix_multiplication()
    #any()
    #repeat()
    #scatter()
    #split()
    #unsqueeze()
    gather()