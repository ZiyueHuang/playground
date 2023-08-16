import collections
from contextlib import ContextDecorator
import torch
import torch.nn as nn
import torch.nn.functional as F


class FreeEventQueue:
    """https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_limiter_utils.py"""
    def __init__(self, max_num=2):
        self._queue = collections.deque()
        self._max_num_inflight_copy = max_num

    def enqueue(self, free_event):
        self._queue.append(free_event)

    def dequeue_if_needed(self):
        if len(self._queue) >= self._max_num_inflight_copy:
            return self._dequeue()
        return None

    def _dequeue(self):
        if self._queue:
            event = self._queue.popleft()
            return event
        return None


def _deque_event_and_synchronize(queue):
    event = queue.dequeue_if_needed()
    if event:
        event.synchronize()

def _enque_event(queue):
    free_event = torch.cuda.Event()
    free_event.record()
    queue.enqueue(free_event)


class MemoryDelta(ContextDecorator):
    def __init__(self):
        self.active_memory_enter = 0
        self.active_memory_exit = 0

    def __enter__(self):
        self.active_memory_enter = torch.cuda.memory_stats()["active_bytes.all.current"]
        return self

    def __exit__(self, *exc):
        self.active_memory_exit = torch.cuda.memory_stats()["active_bytes.all.peak"]

    def delta(self):
        return self.active_memory_exit - self.active_memory_enter


def check_correctness_inter():
    tensor_pairs = [('grad_none.pt', 'grad_cpu_overlap.pt'), ('out_none.pt', 'out_cpu_overlap.pt')]
    for t1_name, t2_name in tensor_pairs:
        torch.testing.assert_close(torch.load(t1_name), torch.load(t2_name))
        print('pass check: ', t1_name, t2_name)


def check_correctness_intra():
    tensor_pairs = [('grad_True.pt', 'grad_False.pt'), ('out_True.pt', 'out_False.pt')]
    for t1_name, t2_name in tensor_pairs:
        torch.testing.assert_close(torch.load(t1_name), torch.load(t2_name), rtol=1e-5, atol=1e-4)
        print('pass check: ', t1_name, t2_name)
