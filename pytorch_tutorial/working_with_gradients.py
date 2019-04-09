import torch



# out = 1/4(sum(3*(x+2)^2))
x = torch.ones(2, 2,  dtype=torch.float32, requires_grad=True)

z = 3 * (x + 2) * (x + 2)

out = z.mean()

print(out.grad_fn)

#take gradient
out.backward()

#check gradient
# d(out)/dx
print(x.grad)
