import math, allo
import numpy as np
from allo._mlir import ir as mlir_ir
StringAttr = mlir_ir.StringAttr
from allo.ir.types import float32, int32, Int, index
from allo.autoscheduler.passes import dataflow_optimization_pass
from allo.autoscheduler.config import AutoschedulerConfig
from allo.autoscheduler.dfg import DFG
from gurobipy import GurobiError
from allo.customize import Partition as partition
from allo.ir.utils import MockBuffer

# --------------------------------------------------------------------------------
# Problem size: 1 head of 32 × 32 with 64-wide model dimension
# (easy to synthesize quickly; raise L/D/H for bigger experiments)
# --------------------------------------------------------------------------------
H, L, D = 8, 64, 1024
P = H // 4 # parallel heads
h_d:int32 = D // H
Ty = float32
MIN_FLOAT32 = -3.402823466e+38  # minimum float32 value

TILE_SIZE_SOFTMAX = 16

import math
import numpy as np
float32_scalar = float32    

def flash_attention(Q: Ty[L, D], K: Ty[L, D], V: Ty[L, D])->Ty[L, D]:
    Z: Ty[L, D] = 0.0
    for i in allo.grid(L, name = "i"):
        Z_row = flash_attention_column(Q, K, V, i)
        for j in allo.grid(D, name = "j"):
            Z[i, j] = Z_row[j]
    return Z

def flash_attention_column(Q: Ty[L, D], K: Ty[L, D], V: Ty[L, D], V_row: index)->Ty[D]:
    Z_row: Ty[D]
    for i in allo.grid(D, name = "i"): # iterate over the columns of one row
            x_i = dot_product(Q, K, i)
            Z_tmp = online_softmax(x_i, i, V, V_row)
            Z_row[i] = Z_tmp
    return Z_row

def dot_product(Q: Ty[L, D], K: Ty[L, D], start_pos: index)->Ty:
    #this kernel computes the dot product of qkT vectors
    total: Ty = 0.0
    for j in allo.grid(D, name = "i"):
        total += Q[start_pos,j] * K[start_pos,j]
    return total

def online_softmax(qkT: Ty, start_pos: index, V: Ty[L, D], V_row: index) -> Ty:
    local_max_curr: Ty = MIN_FLOAT32
    local_max_prev: Ty = MIN_FLOAT32
    exp_prev: Ty = 0.0
    exp_curr: Ty = 0.0
    Z_prev: Ty = 0.0
    Z_curr: Ty = 0.0
    if qkT > local_max_prev:
        local_max_curr = qkT
    exp_curr = allo.exp(local_max_prev - local_max_curr)*exp_prev + allo.exp(qkT - local_max_curr)
    Z_curr = Z_prev*(exp_prev/exp_curr)*allo.exp(local_max_prev - local_max_curr) + (allo.exp(qkT - local_max_curr)/exp_curr)*V[V_row, start_pos]

    #update the previous values
    local_max_prev = local_max_curr
    exp_prev = exp_curr
    Z_prev = Z_curr

    return Z_curr

def test_flash_attention():
    Q = np.random.rand(L, D).astype(np.float32)
    K = np.random.rand(L, D).astype(np.float32)
    V = np.random.rand(L, D).astype(np.float32)
    #Z = flash_attention(Q, K, V)

    s1 = allo.customize(dot_product)
    s2 = allo.customize(online_softmax)
    s3 = allo.customize(flash_attention)

    s3.compose([s1, s2])
    #mod = s3.build(target = "vitis_hls", mode = "csyn", project = "flash_attention.prj")()
    mod = s1.build(target = "vitis_hls", mode = "csyn", project = "dot_prodcut.prj")()

    #mod = s3.build(target = "vitis_hls", mode = "csyn", project = "flash_attention.prj")()


if __name__ == "__main__":
    test_flash_attention()

