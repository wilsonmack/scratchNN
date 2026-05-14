# scratchNN
Building a neural network + training infra from scratch

My goal here is to build a neural network and related autograd architecture from scratch to deepen my understanding. I'm going to build these things relying on first principles.

I'll avoid using most libraries aside from the most basic linear algebra packages (probably in numpy) when it comes to implementation.

Ultimately what I want to do is:

# Project Plan
1. Build a custom gradient engine, analagous to the pytorch autograd
    a. Implement an "enriched" tensor structure first, which would contain an underlying data value (the tensor, likely a numpy matrix) plus necessary metadata for backprop (grad, linkages to other tensors in computational graph)
    b. Integrate the forward pass part of the autodifferentiation
    c. Implement backpropagation, expecting iteration on the two structures built above as we go...
2. Build a NN building API that supports key operations
3. (Optional) Potentially add some visualizations, like graph generation, etc...
4. Train something! At this stage I'd imagine a very simple binary classifier (ex. cats vs. dogs)

# Ideas
* For testing - will use Pytorch as a source of truth for handling unit tests (I can be confident Pytorch produces a *correct* result, up to rounding/fp diffs). For this, I'd want to use pytest to build the test

# Refs
https://pytorch.org/blog/overview-of-pytorch-autograd-engine/
https://blog.ezyang.com/2019/05/pytorch-internals/
https://en.wikipedia.org/wiki/Backpropagation