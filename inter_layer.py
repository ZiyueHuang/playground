import collections
import contextlib
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl

import utils


torch.manual_seed(123)


class save_on_cpu_overlap(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self):
        copy_stream = torch.cuda.Stream()
        current_stream = torch.cuda.current_stream()
        pack_event_queue = utils.FreeEventQueue()
        unpack_event_queue = utils.FreeEventQueue()

        def pack_to_cpu(tensor):
            utils._deque_event_and_synchronize(pack_event_queue)
            copy_stream.wait_stream(current_stream)
            with torch.cuda.stream(copy_stream):
                packed = tensor.to("cpu", non_blocking=True)
            tensor.record_stream(copy_stream)
            utils._enque_event(pack_event_queue)
            return (tensor.device, packed)

        def unpack_from_cpu(packed):
            utils._deque_event_and_synchronize(unpack_event_queue)
            device, tensor = packed
            with torch.cuda.stream(copy_stream):
                unpacked = tensor.to(device, non_blocking=True)
            current_stream.wait_stream(copy_stream)
            unpacked.record_stream(current_stream)
            utils._enque_event(unpack_event_queue)
            return unpacked

        super().__init__(pack_to_cpu, unpack_from_cpu)


class Model(nn.Module):
    def __init__(
        self,
        num_layers,
        dim,
        use_checkpoint=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        wrp = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.REENTRANT,
            preserve_rng_state=False,
        ) if use_checkpoint else (lambda x: x)

        for i in range(self.num_layers):
            layer = wrp(nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim)))
            self.layers.append(layer)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--use_checkpoint', action='store_true')
parser.add_argument('--offload_type', type=str, default='none')
parser.add_argument('--check', action='store_true')
args = parser.parse_args()


dim = 4096
bsz = 2048

inputs = torch.randn(bsz, dim).cuda().requires_grad_()
out_grad = torch.ones(bsz, dim).cuda()

model = Model(25, dim, args.use_checkpoint).cuda()

if args.check:
    context_obj = save_on_cpu_overlap() if args.offload_type == 'cpu_overlap' else contextlib.nullcontext()
    with context_obj:
        out = model(inputs)
    out.backward(out_grad)
    torch.cuda.synchronize()
    torch.save(out.detach(), './out_{}.pt'.format(str(args.offload_type)))
    torch.save(inputs.grad.detach(), './grad_{}.pt'.format(str(args.offload_type)))

for i in range(3):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    if args.offload_type == 'cpu_overlap':
        context_obj = save_on_cpu_overlap()
    elif args.offload_type == 'cpu':
        context_obj = torch.autograd.graph.save_on_cpu(pin_memory=True)
    else:
        context_obj = contextlib.nullcontext()

    with utils.MemoryDelta() as mem:
        with context_obj:
            out = model(inputs)
    if i == 0:
        print("mem.delta() in MB: ", mem.delta()/1024/1024)
    out.backward(out_grad)
    end_event.record()
    torch.cuda.synchronize()
    print("elapsed time: ", start_event.elapsed_time(end_event))

print('reserved_bytes.all.peak in MB: ', torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024)
