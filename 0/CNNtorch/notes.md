# PyTorch Introduction

- `Tensors`: _fancy optimized multidimensional matrices_
- `Automatic Differentiation`: "_lemme do the math_"

Python or C++? I mean ...

- GPU check: `nvidia-smi`

Everything is a tensor. It's basically a `np.ndarray`. 

We can specify: 
- the `dtype` (integer), 
- the `device` (cpu or cuda), 
- the `layout` (dense tensors = strided or sparse = sparse_coo)

GPU will try to allocate stuff in contiguous cells! beware if stuff is sparse !

We can't sum tensors from different GPUs ! beware of cuda:n where n is the GPU id. check also `CUDA_VISIBLE_DEVICES` environment variable.

_REMEMBER TO DEALLOCATE THE MEMORY !!!!!!_ ahh, interesting.

- Context manager: `with ...cuda('1'):` keyword !

- Tensor Broadcasting : `tensor * 3` is like every element of the tensor is multiplied by 3. `tensor1 = torch.rand(1,1,5)` and `tensor*tensor1` is _broadcasting._

- Tensor Dot Product : check if inner dimensions are matching ! and then do `tensor @ tensor1`.

- Tensor Division and Sum: guess what. `tensor \ tensor1` and `tensor + tensor1`.

We can use standard python indexing `[:k]` and conditions filter.

RESHAPING !!! squeeze, unsqueeze, transpose, permute ... 

REDUCING !!! sum, prod, amin, amax ...

## Autograd, the PyTorch Troy's horse.

We don't want to compute gradients by hand. Machines are our slaves. Long live the machines.

Take for example `f(x,y) = sin(xy)`.

We could have two nodes, x and y, another node that is multiplication between them, and another node which is the sine. It's a computational graph. Then we go backwards, derivating each operation! So, from top, we'd have cosine, x or y (depending on partial derivative variable), and then to the root we'd have grad(x) or grad(y). 

What's the matter with in-place operations??? We don't want to modify the entries of the initial tensor. Inefficient in the derivation, and we'd lose a part of the graph, the nodes x and y ! because we're doing stuff in place. we lose branches, so we can't reach some gradients for some variables.

Any tensor, or any operation, have the gradient function, for the backward. It stores useful stuff for the computation of the gradient. Set attribute `requires_grad`. 

Wanna truncate the gradient? use `detach` method. It removes the tensor from the graph, making it a leaf. 

At inference time we don't need gradients, no need for computational graph, we waste performance. Set `torch.no_grad` if you don't need gradient !

## torch.nn

Relies on Module class: `nn.Module`. Wanna do a custom module? overwrite the `forward` function, not `__call__` function. And remember to call `super().__init__()`. Use nn.`ParameterList()`, otherwise parameters won't be visible from outside !

`nn.functional` provides basic operators like all loss functions. import it as F or something.

torch.optim offers optimization algos like `sgd = torch.optim.SGD(model.parameters, lr=0.01)`. Have you used `ParameterList()`? otherwise won't work. then:

`loss.backward() ->
sgd.step() ->
sgd.zero_grad()` (important otherwise you accumulate stuff)

BEWARE !!!! Modules have train/eval mode. Change it if you don't want your sweet parameters to be retrained and forever changed.

## Datasets and Loaders

We want the fantastic `__getitem()__` function. And `torch.data.utils.DataLoader`

## Model Serialization and Logging

We can save just the parameters and not the entire model. And other fancy stuff. We can even save the model every tot epochs. Noice. May be costly tho?

We can monitor stuff with `tensorboard`. Somehow. 
