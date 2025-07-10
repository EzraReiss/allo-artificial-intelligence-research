import math, allo
import numpy as np
from allo._mlir import ir as mlir_ir
StringAttr = mlir_ir.StringAttr
from allo.ir.types import float32, int32
from allo.autoscheduler.passes import dataflow_optimization_pass
from allo.autoscheduler.config import AutoschedulerConfig
from allo.autoscheduler.dfg import DFG
from gurobipy import GurobiError
from allo.customize import Partition as partition
from allo.ir.utils import MockBuffer

TILE_SIZE = 16
L = 64
Ty = float32
MIN_FLOAT32:Ty = -3.402823466e+38  # minimum float32 value

def softmax_top(QK_in: Ty[L, L]) -> Ty[L, L]:     # TEMP for exponentials
    QK_out: Ty[L, L] = 0.0
    max_vals: Ty[L] = MIN_FLOAT32
    rows_total: Ty[L] = 0.0
    invs: Ty[L] = 0.0
    # exp_buf: Ty[L, L] = 0.0
    # exp_buf_copy: Ty[L, L] = 0.0
    i_arr: int32[L*L]
    for i in allo.grid(L*L, name = "i"):
        i_arr[i] = i

    for i_outer in allo.grid(L//TILE_SIZE, name = "i_outer"):
        for i_inner in allo.grid(TILE_SIZE, name = "i_inner"):
            max_vals[i_outer*TILE_SIZE + i_inner] = softmax_p1(QK_in, i_arr[i_outer*TILE_SIZE + i_inner])
            exp_buf = softmax_p2(QK_in, max_vals, i_arr[i_outer*TILE_SIZE + i_inner])
            invs[i_outer*TILE_SIZE + i_inner] = softmax_p3(exp_buf, rows_total, i_arr[i_outer*TILE_SIZE + i_inner])
            softmax_p4(QK_out, exp_buf, invs[i_outer*TILE_SIZE + i_inner], i_arr[i_outer*TILE_SIZE + i_inner])
    return QK_out

def softmax_p1(QK_in: Ty[L, L], i_pos: int32) -> Ty:
    local_max: Ty = MIN_FLOAT32
    for j1 in allo.grid(L, name = "j1"):
        v:Ty = QK_in[i_pos, j1]
        m:Ty = local_max             
        local_max = v if v > m else m  
    return local_max

def softmax_p2(QK_in: Ty[L, L], max_vals: Ty[L], i_pos: int32) ->  Ty[L]:
    local_max: Ty = max_vals[i_pos]
    exp_buf: Ty[L] = 0.0
    for j2 in allo.grid(L, name = "j2"):
        e:Ty = allo.exp(QK_in[i_pos, j2] - local_max)
        exp_buf[j2] = e
    return exp_buf

def softmax_p3(exp_buf: Ty[L], rows_total: Ty[L], i_pos: int32) -> Ty:
    for j3 in allo.grid(L, name = "j3"):
        rows_total[i_pos] += exp_buf[j3]
    inv:Ty = 1.0 / rows_total[i_pos] #this does not catch a division by zero
    return inv

def softmax_p4(QK_out: Ty[L, L], exp_buf_copy: Ty[L], invs: Ty, i_pos: int32):
    for j4 in allo.grid(L, name = "j4"):
        QK_out[i_pos, j4] = exp_buf_copy[j4] * invs


def test_function_equivalence():
    QK_in = np.random.rand(L, L).astype(np.float32)
    QK_out_base = np.zeros((L, L), dtype=np.float32)
    QK_out_opt = np.zeros((L, L), dtype=np.float32)
    base_sch = allo.customize(softmax_v1)
    opt_sch = allo.customize(softmax_v1)


    base_mod = base_sch.build(target="vitis_hls", mode="sw_emu", project="sw_softmax_top_opt.prj")(QK_in, QK_out_base)
    opt_mod = opt_sch.build(target="vitis_hls", mode="sw_emu", project="sw_softmax_top_opt.prj")(QK_in, QK_out_opt)
    np.testing.assert_allclose(QK_out_opt, QK_out_base,  rtol=1e-5, atol=1e-5)
    print("passed functional simulation 1")

def test_softmax_top():
    # create the schedules
    s1 = allo.customize(softmax_p1)
    s2 = allo.customize(softmax_p2)
    s3 = allo.customize(softmax_p3)
    s4 = allo.customize(softmax_p4)
    top_sch = allo.customize(softmax_top)

    # # create fifo streams from the inner loop to softmax_p4
    # top_sch.to(top_sch.exp_buf, "softmax_p4")
    # top_sch.to(top_sch.invs, "softmax_p4")

    # partitions
    # top_sch.partition(top_sch.QK_in, partition_type=partition.Cyclic, dim=1, factor = 4)
    # top_sch.partition(top_sch.max_vals, partition_type=partition.Cyclic, dim=1, factor = 4)
    # top_sch.partition(top_sch.exp_buf, partition_type=partition.Cyclic, dim=0, factor = 16)
    # top_sch.partition(top_sch.rows_total, partition_type=partition.Cyclic, dim=1, factor = 4)
    # top_sch.partition(top_sch.invs, partition_type=partition.Cyclic, dim=1, factor = 16)
    # top_sch.partition(top_sch.QK_out, partition_type=partition.Cyclic, dim=1, factor = 16)

    # schedule the lower level functions
    schedule_softmax_p1(s1)
    schedule_softmax_p2(s2)
    schedule_softmax_p3(s3)
    schedule_softmax_p4(s4)

    # compose the lower level schedules into top_sch
    top_sch.compose([s1, s2, s3, s4])

    # unfold the inner loop and apply dataflow to the outer loop
    top_sch.dataflow(top_sch.get_loops("softmax_top")["i_outer"]["i_inner"])
    top_sch.unroll("i_outer", factor = 4)
    #top_sch.unfold("i_outer", [0]) #unfold the outer loop
    #top_sch.dataflow("softmax_top")
    # build the top level schedule
    mod = top_sch.build(target="vitis_hls", mode="csyn", project="softmax_top.prj")()


def schedule_softmax_p1(s):
    ### partitions are just here for bookkeeping ###
    # s.partition(s.QK_in, partition_type=partition.Cyclic, dim=1, factor = 4)
    # s.partition(s.max_vals, partition_type=partition.Cyclic, dim=1, factor = 4)
    s.pipeline("j1")
    

def schedule_softmax_p2(s):
    ### partitions are just here for bookkeeping ###
    # s.partition(s.QK_in, partition_type=partition.Cyclic, dim=1, factor = 4)
    #s.partition(s.exp_buf, partition_type=partition.Cyclic, dim=0, factor = 16)
    s.pipeline("j2")


def schedule_softmax_p3(s):
    ### partitions are just here for bookkeeping ###
    # we partition on all dimensions of exp_buf to unroll the inner and outer loop
    # s.partition(s.exp_buf, partition_type=partition.Cyclic, dim=0, factor = 16)
    # s.partition(s.rows_total, partition_type=partition.Cyclic, dim=1, factor = 4)
    # s.partition(s.invs, partition_type=partition.Cyclic, dim=1, factor = 16)
    s.pipeline("j3")


def schedule_softmax_p4(s):
    ### partitions are just here for bookkeeping ###
    # s.partition(s.exp_buf, partition_type=partition.Cyclic, dim=0, factor = 16)
    # s.partition(s.QK_out, partition_type=partition.Cyclic, dim=1, factor = 16)
    # s.partition(s.invs, partition_type=partition.Cyclic, dim=1, factor = 16)
    # s.reorder("j4", "i4")
    s.unroll("j4", factor = 16)
    s.pipeline("j4")


def schedule_base():
    s1 = allo.customize(softmax_p1)
    s2 = allo.customize(softmax_p2)
    s3 = allo.customize(softmax_p3)
    s4 = allo.customize(softmax_p4)
    top_sch = allo.customize(softmax_top)
    top_sch.compose([s1, s2, s3, s4])
    return top_sch


def test_base():
    top_sch = schedule_base()
    mod = top_sch.build(target="vitis_hls", mode="csyn", project="softmax_top_base.prj")()



if __name__ == "__main__":
    #test_base()
    test_softmax_top()
    #test_function_equivalence()