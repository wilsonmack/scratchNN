# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: scratchnn
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Grad Tracking
# Here I'm going to run some basic experiments to see how I can build computational graphs for scalar-valued functions, ultimately I'll extend this into vectors/tensors when I move onto building my autodifferentiation engine tool, but for now this will be my focus to get a grasp on the underlying math (and what I'll need to track).
#
# # Ultimately all I need to do here is to build a system that lets me track these in the fwd pass:
#  * Make an enriched "Number" class, that supports our basic operations under the hood (maybe default as a float for now)
#  * Whenever an operation happens, it should return a new "Number" object that has linkages that place it in the graph
#  * It should be able to leverage the graph that is built in the fwd pass to run a backwards propagation, updating the gradients of each number
#
# # Test functions:
# 1. z = x + y
# 2. z = x * y
# 3. z = x**2
# 4. z = x**2 + y
# 5. z = 4 * x**2
# 6. z = 4 * x\*\*2 + y\*\*2
# 3. z = log(x + y)

# %%
from __future__ import annotations

# %%
from multiprocessing import Value


class Function(StrEnum):
    NOOP = ""
    ADD = "add"
    MUL = "mul"
    LOG = "log"
    EXP = "exp"


class CompNode:
    def __init__(self, fn=Function):
        self._bwd = self._setup_bwd(fn)

    def _setup_bwd(self, fn: Function):
        match fn:
            case Function.NOOP:

            case Function.ADD:
                
            case Function.MUL:
            
            case Function.LOG:
            
            case Function.EXP:
            
            case _:
                raise ValueError(f"Unknown function type \'{fn}\' in computational graph!")
                

class Var:
    def __init__(self, val, fn=Function, parents=(), tag=""):
        self.val = val
        self.fn = fn
        self._bwd = lambda: None
        self.grad = 0
        self.parents = parents
        self.tag = tag

    def enforce_var(self, other) -> Var:
        if not isinstance(other, Var):
            other = Var(other)
        return other

    def __add__(self, other: Var) -> Var:
        other = self.enforce_var(other)
        result = self.val + other.val
        new_var = Var(result, fn=Function.ADD, parents=(self, other))
        def _add_bwd():
            self.grad = new_var.grad
            other.grad = new_var.grad
        self._bwd = _add_bwd
        return new_var

    def __mul__(self, other):
        other = self.enforce_var(other)
        result = self.val * other.val
        new_var = Var(result, fn=Function.MUL, parents=(self, other))
        def _mul_bwd():
            self.grad = new_var.grad * other.val
            other.grad = new_var.grad * self.val
        self._bwd = _mul_bwd
        return new_var

    def __repr__(self):
        return " ".join(["Var", self.tag, ":", f"(val={self.val}, grad={self.grad})"])
    
    def backward(self, val=1.0):
        # Typically we first toposort

        # Then we call backward on each node in the reversed
        # computational graph



# %%
x = Var(1, tag="x")
y = Var(2, tag="y")
z = x + y
L = z * 3.0
z.tag = "z"
# print([v for v in z.chained])
L.backward(1.0)

# %%
# dL/dz = 3.0

# dL/dx = dL/dL * dL/dz * dz/dx
#       = 1.0 * z.grad * (1.0)

# %%
x = Var(1, tag="x")
y = Var(2, tag="y")
z = x * y
z.tag = "z"
# print([v for v in z.chained])
z.backward(1.0)
print([v for v in z.chained])
