# scratchNN
Building a neural network + training infra from scratch

My goal here is to build a neural network and related autograd architecture from scratch to deepen my understanding. I'm going to build these things relying on first principles, rather than looking at examples of what is already being done (like reading through the pytorch source code).

I'll avoid using most libraries aside from the most basic linear algebra packages to prove my understanding of the math under the hood.

Ultimately what I want to do is:

# Project Goals
1. Build a custom gradient engine, analagous to the pytorch autograd
2. Build a NN building API that supports key operations
3.

# Refs
https://pytorch.org/blog/overview-of-pytorch-autograd-engine/
https://blog.ezyang.com/2019/05/pytorch-internals/
https://en.wikipedia.org/wiki/Backpropagation