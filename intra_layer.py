import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
from torch.utils.checkpoint import noop_context_fn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

import collections
from functools import partial
import contextlib

import modeling_chatglm
import configuration_chatglm

import utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_checkpoint', action='store_true')
parser.add_argument('--use_half', action='store_true')
parser.add_argument('--offload_type', type=str, default='none')
parser.add_argument('--seq_length', type=int, default=4096)
parser.add_argument('--check', action='store_true')
args = parser.parse_args()

torch.manual_seed(123)

copy_stream = torch.cuda.Stream()
current_stream = torch.cuda.current_stream()
pack_event_queue = utils.FreeEventQueue()
unpack_event_queue = utils.FreeEventQueue()


def _detach_to_cpu(x):
    if isinstance(x, torch.Tensor) and x.device.type == "cuda":
        #return x.detach()  # offload-to-HBM
        utils._deque_event_and_synchronize(pack_event_queue)
        tensor = x.detach()
        copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(copy_stream):
            packed = tensor.to("cpu", non_blocking=True)
        tensor.record_stream(copy_stream)
        utils._enque_event(pack_event_queue)
        return packed

    return x


def _to_cuda(x):
    if isinstance(x, torch.Tensor) and x.device.type == "cpu":
        utils._deque_event_and_synchronize(unpack_event_queue)
        with torch.cuda.stream(copy_stream):
            unpacked = x.to("cuda:0", non_blocking=True)
        current_stream.wait_stream(copy_stream)
        unpacked.record_stream(current_stream)
        utils._enque_event(unpack_event_queue)
        return unpacked
    return x


def _get_default_policy(allow_list=None):
    _default_allow_list = [
        #'mm.default',  # maybe useful for fp32, or offload-to-HBM for fp16
        '_scaled_dot_product_efficient_attention.default',
    ]
    if allow_list is None:
        allow_list = _default_allow_list

    def _default_policy(func_name, *args, **kwargs):
        return func_name in allow_list

    return _default_policy


"""https://github.com/pytorch/pytorch/issues/70135#issuecomment-1542439983"""
def get_selective_offloading_checkpoint_modes():
    policy_fn = _get_default_policy()
    cpu_storage = []

    class CachingMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            if policy_fn(func.__name__, *args, **kwargs):
                out = func(*args, **kwargs)
                # Detach and move tensors to cpu
                out_detached_cpu = tree_map(_detach_to_cpu, out)
                cpu_storage.append(out_detached_cpu)
                return out
            return func(*args, **kwargs)

    class CachedMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            if policy_fn(func.__name__, *args, **kwargs):
                # Detach and move tensors back to cuda
                out = tree_map(_to_cuda, cpu_storage.pop(0))
                return out
            return func(*args, **kwargs)

    return CachingMode(), CachedMode()


class Model(torch.nn.Module):
    def __init__(self, config, device=None):
        super(Model, self).__init__()
        self.config = config
        ctx_fn = noop_context_fn if args.offload_type == 'none' else get_selective_offloading_checkpoint_modes

        def build_layer(layer_number):
            wrp = partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                context_fn=ctx_fn,
                determinism_check="none",
            ) if args.use_checkpoint else (lambda x : x)
            return wrp(modeling_chatglm.GLMBlock(config, layer_number, device=device))

        self.layers = torch.nn.ModuleList([build_layer(i) for i in range(3)])

    def forward(self, hidden_states):
        pos_embed = hidden_states.new_zeros((self.config.seq_length, 1, 32, 2))
        for i in range(3):
            hidden_states, _ = self.layers[i](hidden_states, None, pos_embed)
        return hidden_states



config = configuration_chatglm.ChatGLMConfig()
config.seq_length = args.seq_length
config.rmsnorm = False  # empty init results to NAN

bsz = 2
dtype = torch.half if args.use_half else torch.float32
inputs = (1./4096 * torch.randn(config.seq_length, bsz, config.hidden_size)).to(dtype).cuda().requires_grad_()
out_grad = (1./(bsz*4096) * torch.ones(config.seq_length, bsz, config.hidden_size)).to(dtype).cuda()

model = Model(config).half().cuda() if args.use_half else Model(config).cuda()

if args.check:
    out = model(inputs)
    out.backward(out_grad)
    torch.cuda.synchronize()
    torch.save(out.detach(), './out_{}.pt'.format(str(args.use_checkpoint)))
    torch.save(inputs.grad.detach(), './grad_{}.pt'.format(str(args.use_checkpoint)))


for i in range(3):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with utils.MemoryDelta() as mem:
        out = model(inputs)
    if i == 0:
        print("mem.delta() MB", mem.delta()/1024/1024)
    out.backward(out_grad)
    end_event.record()
    torch.cuda.synchronize()
    print("elapsed time: ", start_event.elapsed_time(end_event))
