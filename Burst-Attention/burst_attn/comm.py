import torch
import bmtrain as bmt
import torch.distributed as dist
from bmtrain.distributed.ops import ncclSend, ncclRecv
from bmtrain.nccl import commCount, groupEnd, groupStart, allReduce, commRank
import os
from .log_helper import get_logger

_logger = get_logger(__file__, level="WARNING")

def replicate(tensor):
    res = torch.empty_like(tensor)
    res.copy_(tensor)
    return res

def broadcast(tensor, src, group=None):
    if is_bmt_enable():
        return bmt.distributed.broadcast(tensor, src)
    else:
        dist.broadcast(tensor, src, group)
        return tensor


def broadcast_list(tensor_list, src, group=None):
    res = []
    for tensor in tensor_list:
        res.append(broadcast(tensor, src, group))
    return res


def log_rank0(*args, **kwargs):
    if get_rank() == 0:
        _logger.info(*args, **kwargs)


def is_bmt_enable():
    return bmt.init.is_initialized()


def _ring(tensor):
    """Sync send-recv interface"""
    return ring_send_recv(tensor, bmt.rank(), bmt.config["comm"])


def ring_send_recv(tensor, comm):
    """Sync send-recv interface"""

    rank = get_rank()
    count = get_world_size(comm)
    next_rank = (rank + 1) % count
    prev_rank = (rank - 1 + count) % count
    res = torch.empty_like(tensor, device="cuda", dtype=tensor.dtype)
    if is_bmt_enable():
        groupStart()
        ncclSend(tensor.storage(), next_rank, comm)
        ncclRecv(res.storage(), prev_rank, comm)
        groupEnd()
    else:
        send_op = dist.P2POp(dist.isend, tensor, next_rank, group=comm)
        recv_op = dist.P2POp(dist.irecv, res, prev_rank, group=comm)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()
    return res


def all_reduce(t, group=None):
    if not is_bmt_enable():
        t = dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    else:
        group = bmt.config["comm"] if not group else group
        allReduce(t.storage(), t.storage(), "sum", group)

def get_local_world_size():
    return bmt.config["local_world_size"] if is_bmt_enable() else int(os.environ.get("LOCAL_WORLD_SIZE", 1))

def get_world_size(c=None):
    if not is_bmt_enable():
        return dist.get_world_size(c)
    else:
        c = bmt.config["comm"] if not c else c
        return commCount(c)


class ops_wrapper:
    def __init__(self, op, tensor, *args, **kwargs):
        self.op = op
        self.tensor = tensor
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.op(self.tensor.storage(), *self.args, **self.kwargs)


def get_rank(group=None):
    if is_bmt_enable():
        group = bmt.config["comm"] if not group else group
        return commRank(group)
    else:
        return dist.get_rank(group)


class Ring:
    def __init__(self, process_group, local_group=[None, None], dq=False):
        if is_bmt_enable():
            if process_group:
                self.comm = process_group
            else:
                self.comm = bmt.config["comm"]
            self.backend = "bmtrain"
        else:
            self.comm = process_group
            self.backend = "torch"
        self._wait_inter = False
        self.world_size = get_world_size(process_group)
        self.rank = get_rank(process_group)
        self.event = torch.cuda.Event(enable_timing=False)
        self.local_group = local_group[0]
        self.local_group2 = local_group[1]
        self.intra_rank = get_rank(self.local_group)
        self.inter_rank = get_rank(self.local_group2)
        self.intra_size = get_world_size(self.local_group)
        self.inter_size = get_world_size(self.local_group2)
        self.double_ring = True if local_group[0] or self.inter_size == 1 or self.intra_size == 1 else False

        if self.double_ring and is_bmt_enable():
            if dq:
                self.intra_stream = bmt.config["sp_stream3"]
                self.inter_stream = bmt.config["sp_stream4"]
            else:
                self.intra_stream = bmt.config["sp_stream"]
                self.inter_stream = bmt.config["sp_stream2"]
        elif is_bmt_enable():
            if dq:
                self.intra_stream = bmt.config['sp_stream2']
                self.inter_stream = bmt.config['sp_stream2']
            else:
                self.intra_stream = bmt.config['sp_stream']
                self.inter_stream = bmt.config['sp_stream']

        self.buffer_list = []

        self.reqs = []
        self.ops = []
        self._inter_ops = []
        self._inter_reqs = []

    def _make_ring_ops(self, src_tensors, dst_tensors, group):
        # no handle is connected with Ring object
        comm = self.comm if group is None else group
        rank = get_rank(comm)
        count = get_world_size(comm)
        next_rank = (rank + 1) % count
        prev_rank = (rank - 1 + count) % count
        ops = []
        if self.backend == "torch" and comm is not None:
            next_rank = torch.distributed.get_global_rank(comm, next_rank)
            prev_rank = torch.distributed.get_global_rank(comm, prev_rank)
        for src_tensor, dst_tensor in zip(src_tensors, dst_tensors):
            if self.backend == "torch":
                send_op = dist.P2POp(dist.isend, src_tensor, next_rank, group=comm)
                recv_op = dist.P2POp(dist.irecv, dst_tensor, prev_rank, group=comm)
            else:
                send_op = ops_wrapper(ncclSend, src_tensor, next_rank, comm)
                recv_op = ops_wrapper(ncclRecv, dst_tensor, prev_rank, comm)
            if rank % 2 == 0:
                ops.append(send_op)
                ops.append(recv_op)
            else:
                ops.append(recv_op)
                ops.append(send_op)
        return ops

    def ring_send_recv(self, *tensor_list):
        res_list = [torch.empty_like(t) for t in tensor_list]
        self.ops += self._make_ring_ops(tensor_list, res_list, None)

    def check_buffer(self, buffer_list, tensor_list):
        flag = 1
        if len(buffer_list) != len(tensor_list):
            return 0
        for b, d in zip(buffer_list, tensor_list):
            if b.size() != d.size() or b.dtype != d.dtype:
                flag = 0
        return flag

    def double_ring_send_recv_q(self, tensor_list, dest_list, r=0):
        if not self.double_ring:
            self._ring_send_recv_base(tensor_list, dest_list)
        else:
            if r % self.intra_size == 1 and r != 1:
                if not self.check_buffer(self.buffer_list, tensor_list):
                    self.buffer_list = [torch.empty_like(t) for t in tensor_list]
                    send_buffer = [torch.empty_like(t) for t in tensor_list]
                    for i in range(len(tensor_list)):
                        send_buffer[i].copy_(tensor_list[i])
                    self._inter_ops += self._make_ring_ops(
                        send_buffer, self.buffer_list, self.local_group2
                    )
                else:
                    self.wait(True)
                    add_list = [t + b for t, b in zip(tensor_list, self.buffer_list)]
                    self._inter_ops += self._make_ring_ops(
                        add_list, self.buffer_list, self.local_group2
                    )
                if r // self.intra_size != self.inter_size:
                    for i in range(len(dest_list)):
                        dest_list[i].zero_()
                else:
                    self.commit()
                    self.wait(True)
                    self._ring_send_recv_base(
                        self.buffer_list, dest_list, self.local_group
                    )
                    self.commit()
                    self.wait()
            else:
                self._ring_send_recv_base(tensor_list, dest_list, self.local_group)


    def double_ring_send_recv(self, tensor_list, dest_list, r=0):
        if not self.double_ring:
            self._ring_send_recv_base(tensor_list, dest_list)
        else:
            if r % self.intra_size == 1 and r // self.intra_size != self.inter_size - 1:
                if not self.check_buffer(self.buffer_list, tensor_list):
                    self.buffer_list = [torch.empty_like(t) for t in tensor_list]
                send_buffer = [torch.empty_like(t) for t in tensor_list]
                for i in range(len(tensor_list)):
                    send_buffer[i].copy_(tensor_list[i])
                self._inter_ops += self._make_ring_ops(
                    send_buffer, self.buffer_list, self.local_group2
                )
            if r % self.intra_size == 0 and r != 0:
                if not self.check_buffer(self.buffer_list, dest_list):
                    self.buffer_list = [replicate(t) for t in tensor_list]
                    self._dest_list = dest_list

                assert (
                    len(dest_list) == len(self.buffer_list)
                ), f"len(dest_list)={len(dest_list)} len(self.buffer_list)={len(self.buffer_list)}"
                for i in range(len(dest_list)):
                    dest_list[i], self.buffer_list[i] = (
                        self.buffer_list[i],
                        dest_list[i],
                    )
                self._wait_inter = True

            else:
                self._ring_send_recv_base(tensor_list, dest_list, self.local_group)

    def _ring_send_recv_base(self, tensor_list, dest_list, group=None):
        self.ops += self._make_ring_ops(tensor_list, dest_list, group)

    def _single_p2p_call(self, ops):
        if self.rank % 2 == 0:
            ops[0]()
            ops[1]()
        else:
            ops[1]()
            ops[0]()

    def _commit_ops(self, ops, stream=None):
        if self.backend == "torch":
            reqs = dist.batch_isend_irecv(ops)
        else:
            current_stream = torch.cuda.current_stream()
            current_stream.record_event(self.event)
            with torch.cuda.stream(stream):
                self.event.synchronize()
                for op in ops:
                    op.tensor.record_stream(stream)
                groupStart()
                for op in ops:
                    op()
                groupEnd()
                stream.record_event(self.event)
            reqs = [None]
        return reqs

    def commit(self):
        if len(self.ops) > 0:
            stream = (
                self.intra_stream
                if self.backend != "torch" and "sp_stream" in bmt.config
                else None
            )
            self.reqs = self._commit_ops(self.ops, stream)
            self.ops = []
        if len(self._inter_ops) > 0:
            self._inter_reqs += self._commit_ops(
                self._inter_ops,
                self.inter_stream if self.backend != "torch" else None,
            )
            self._inter_ops = []

    def wait(self, force_wait_inter=False):
        if self._wait_inter or force_wait_inter:
            self._wait_base(True)
            self._wait_inter = False
        else:
            self._wait_base()

    def _wait_base(self, wait_inter_comm=False):
        if self.backend == "torch":
            reqs = self.reqs if not wait_inter_comm else self._inter_reqs
            for req in reqs:
                req.wait()
        else:
            if not wait_inter_comm:
                torch.cuda.current_stream().wait_stream(self.intra_stream)
            else:
                torch.cuda.current_stream().wait_stream(self.inter_stream)
        if wait_inter_comm:
            self._inter_reqs = []
        else:
            self.reqs = []


def print_rank(*args, **kwargs):
    if is_bmt_enable():
        bmt.print_rank(*args, **kwargs)
    else:

        def torch_print_rank(*args, **kwargs):
            if dist.get_rank() == 0:
                print(*args, **kwargs)

        torch_print_rank(*args, **kwargs)


def synchronize():
    if is_bmt_enable():
        bmt.synchronize()
    elif dist.is_initialized():
        dist.barrier()
    else:
        raise ValueError("Init comm first")


def gather_obj(obj):
    if is_bmt_enable():
        return bmt.store.allgather_objects(obj)
    elif dist.is_initialized():
        res = [None] * dist.get_world_size()
        dist.all_gather_object(res, obj)
        torch.distributed.barrier()
        return res
    else:
        raise ValueError("Init comm first")
