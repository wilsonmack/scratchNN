# Autodifferentiation module for the neural network library.
# This module implements the core autodifferentiation functionality,
# implementing the enriched-tensor class that contains the necessary data for tracking
# gradients and supporting backprop. As an extension, I'll include some support for 
# computational graph viz.

# As a first stage - I'll build this class as a generalized tool for tracking the gradients
# in generic math operations (addition, multiplication, some special functions) and then extend
# it to support the specific operations I desire for building a NN library.

import numpy as np

class DiffableTensor:
