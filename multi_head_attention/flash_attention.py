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
# P = H // 4 # parallel heads
# h_d:int32 = D // H
Ty = float32
MIN_FLOAT32 = -3.402823466e+38  # minimum float32 value

TILE_SIZE_SOFTMAX = 16

import math
import numpy as np
float32_scalar = float32    


def flash_attention_column(Q: Ty[L, D], K: Ty[L, D], V: Ty[L, D], V_row: int32)->Ty[D]:
    Z_row: Ty[D]
    for i in allo.grid(D, name = "i"): # iterate over the columns of one row
        index: int32 = i
        x_i = dot_product(Q, K, index)
        Z_tmp = online_softmax(x_i, index, V, V_row)
        Z_row[i] = Z_tmp
    return Z_row

def flash_attention(Q: Ty[L, D], K: Ty[L, D], V: Ty[L, D])->Ty[L, D]:
    Z: Ty[L, D] = 0.0
    Q_sliced: Ty[L, D]
    K_sliced: Ty[L, D]
    V_sliced: Ty[L, D]
    for i in allo.grid(L, name = "i"):
        x: int32 = i
        for ii, jj in allo.grid(L, D, name="loop_slicing"):
            Q_sliced[ii, jj] = Q[ii, jj]
            K_sliced[ii, jj] = K[ii, jj]
            V_sliced[ii, jj] = V[ii, jj]
        Z_row = flash_attention_column(Q_sliced, K_sliced, V_sliced, x)
        for j in allo.grid(D, name = "j"):
            Z[i, j] = Z_row[j]
    return Z



def dot_product(Q: Ty[L, D], K: Ty[L, D], start_pos: int32)->Ty:
    #this kernel computes the dot product of qkT vectors
    total: Ty = 0.0
    for j in allo.grid(D, name = "i"):
        total += Q[start_pos,j] * K[start_pos,j]
    return total

def online_softmax(qkT: Ty, start_pos: int32, V: Ty[L, D], V_row: int32) -> Ty:
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

def np_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def sdp(Q, K, V, H, D):
    context = np.zeros(Q.shape)
    h_d = D // H
    for i in range(H):
        # split Q, K, V
        Q_h = Q[:, i * h_d : (i + 1) * h_d]
        K_h = K[:, i * h_d : (i + 1) * h_d]
        V_h = V[:, i * h_d : (i + 1) * h_d]
        # compute attention
        attention = np.matmul(Q_h, K_h.T)
        Y = np_softmax(attention)
        context_i = np.matmul(Y, V_h)
        context[:, i * h_d : (i + 1) * h_d] = context_i
    return context

    
def test_flash_attention():
    Q = np.random.rand(L, D).astype(np.float32)
    K = np.random.rand(L, D).astype(np.float32)
    V = np.random.rand(L, D).astype(np.float32)
    my_solution = np.zeros(Q.shape)
    solution = sdp(Q, K, V, H, D)
    #Z = flash_attention(Q, K, V)

    # s1 = allo.customize(dot_product)
    # s2 = allo.customize(online_softmax)
    # s3 = allo.customize(flash_attention_column)
    s4 = allo.customize(flash_attention)

    # s3.compose([s1, s2])
    # s4.compose([s3])

    # mod = s4.build()
    # my_sol = mod(Q, K, V)

    #mod = s3.build(target = "vitis_hls", mode = "csyn", project = "flash_attention.prj")()
    #mod = s2.build(target = "vitis_hls", mode = "csyn", project = "online_softmax.prj")()
    mod = s4.build(target = "vitis_hls", mode = "csyn", project = "flash_attention.prj")()
    #rint("finished csynth")
    # mod = s4.build(target = "vitis_hls", mode = "sw_emu", project = "sw_flash_attention.prj")(Q, K, V, my_solution)
    np.testing.assert_allclose(my_sol, solution, atol=1e-5)
    print("everything passed!")
    #mod = s3.build(target = "vitis_hls", mode = "csyn", project = "flash_attention.prj")()


if __name__ == "__main__":
    test_flash_attention()

