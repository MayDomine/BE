from .. import nccl
from .shape import SHAPES
from ..global_var import config
from ..utils import round_up, print_rank
from .utils import format_size
import torch

def all_reduce():
    current_stream = torch.cuda.current_stream()
    for shape in SHAPES:
        global_size = round_up(shape, config['world_size'] * 2)

        partition_tensor = torch.empty( global_size // 2, dtype=torch.half, device="cuda" )
        global_tensor = torch.empty( global_size // 2, dtype=torch.half, device="cuda" )
        
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        current_stream.record_event(start_evt)
        nccl.allReduce(partition_tensor.storage(), global_tensor.storage(),"sum", config['comm'])
        current_stream.record_event(end_evt)
        current_stream.synchronize()
        time_usage = start_evt.elapsed_time(end_evt)

        bw = global_size / 1024 / 1024 / 1024 * 1000 / time_usage * 2
        print_rank("All reduce:\tsize {}\ttime: {:4.3f}\tbw: {:2.6f} GB/s".format(format_size(global_size), time_usage, bw))

