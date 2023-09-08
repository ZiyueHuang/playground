import logging
from typing import Optional

import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def rms_norm_fw(X, Y, W, V, stride, N, eps, BLOCK_SIZE_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    # Move to this row
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(x_var + eps)

    y = x * rstd
    tl.store(V + row, rstd)

    mask = cols < N
    w = tl.load(W + cols, mask=mask, other=1.0)
    y = y * w

    y_ptrs = Y + row * stride + cols
    tl.store(y_ptrs, y, mask=mask)


# Backward pass (DX + partial DW)
@triton.jit
def rms_norm_bwd_dx_fused(
    DX, DY, DW,
    X, W, V,
    Lock, stride, N,
    # META-parameters
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # position of elements processed by this program
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    # offset data pointers to start at the row of interest
    x_ptrs = X + row * stride + cols
    dy_ptrs = DY + row * stride + cols

    # load data to SRAM
    x = tl.load(x_ptrs, mask=mask, other=0)
    dy = tl.load(dy_ptrs, mask=mask, other=0)
    rstd = tl.load(V + row)

    # compute dx
    xhat = x * rstd

    w = tl.load(W + cols, mask=mask, other=0)
    wdy = w * dy


    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    dx = (wdy - xhat * mean1) * rstd

    # write-back dx
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N  # re-materialize the mask to save registers
    dx_ptrs = DX + row * stride + cols
    tl.store(dx_ptrs, dx, mask=mask)

    # accumulate partial sums for dw
    partial_dw = (dy * xhat).to(w.dtype)

    # offset locks and weight/bias gradient pointer
    # each kernel instance accumulates partial sums for
    # DW into one of GROUP_SIZE_M independent buffers
    # these buffers stay in the L2, which allow this kernel
    # to be fast
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M

    # - wait for a lock on the accumulated dw/db
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)

    dw_ptrs = DW + lock_id * N + cols

    if count == 0:
        # first store doesn't accumulate
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(dw_ptrs, mask=mask, other=0.)

    tl.store(dw_ptrs, partial_dw, mask=mask)

    # release lock
    tl.atomic_xchg(Lock, 0)


# A fused Forward-Backward pass (Y + DX + partial DW)
@triton.jit
def rms_norm_bwd_dx_y_fused(
    DX, DY, DW,
    X, W, V, Y,
    Lock, stride, N,
    # META-parameters
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # position of elements processed by this program
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    # offset data pointers to start at the row of interest
    x_ptrs = X + row * stride + cols
    dy_ptrs = DY + row * stride + cols
    y_ptrs = Y + row * stride + cols

    # load data to SRAM
    x = tl.load(x_ptrs, mask=mask, other=0)
    dy = tl.load(dy_ptrs, mask=mask, other=0)
    rstd = tl.load(V + row)

    # compute dx
    xhat = x * rstd
    tl.store(y_ptrs, xhat, mask=mask)

    w = tl.load(W + cols, mask=mask, other=0)
    wdy = w * dy


    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    dx = (wdy - xhat * mean1) * rstd

    # write-back dx
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N  # re-materialize the mask to save registers
    dx_ptrs = DX + row * stride + cols
    tl.store(dx_ptrs, dx, mask=mask)

    # accumulate partial sums for dw
    partial_dw = (dy * xhat).to(w.dtype)

    # offset locks and weight/bias gradient pointer
    # each kernel instance accumulates partial sums for
    # DW into one of GROUP_SIZE_M independent buffers
    # these buffers stay in the L2, which allow this kernel
    # to be fast
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M

    # - wait for a lock on the accumulated dw/db
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)

    dw_ptrs = DW + lock_id * N + cols

    if count == 0:
        # first store doesn't accumulate
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(dw_ptrs, mask=mask, other=0.)

    tl.store(dw_ptrs, partial_dw, mask=mask)

    # release lock
    tl.atomic_xchg(Lock, 0)


# Backward pass (total DW)
@triton.jit
def rms_norm_bwd_dw(
    DW, FINAL_DW,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cols = cols < N

    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        offs = rows[:, None] * N + cols[None, :]
        mask_rm = rows < M

        dw += tl.load(DW + offs, mask=mask_rm[:, None] & mask_cols[None, :], other=0.0)

    sum_dw = tl.sum(dw, axis=0)

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cols = cols < N

    tl.store(FINAL_DW + cols, sum_dw, mask=mask_cols)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        # catch eps being too small if the tensors are fp16
        if x.dtype == torch.float16:
            eps = max(eps, 1.6e-5)

        # allocate output
        y = torch.empty_like(x)

        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        # allocate std, used in the backward pass
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError("This RMS norm doesn't support feature dim >= 64KB.")

        if not x_arg.is_contiguous() or not y.is_contiguous():
            x_arg = x_arg.contiguous()
            y = y.contiguous()

        num_warps = min(max(BLOCK_SIZE_N // 256, 1), 16)

        # enqueue kernel
        rms_norm_fw[(M,)](
            x_arg, y, weight, rstd,
            x_arg.stride(0),
            N,
            eps,
            num_warps=num_warps,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )

        ctx.save_for_backward(x, rstd, weight)
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.num_warps = num_warps

        return y.reshape_as(x)

    @staticmethod
    def backward(
        ctx, dy
    ):
        x, rstd, weight = ctx.saved_tensors

        x = x.reshape(-1, x.size(-1))
        M, N = x.size()

        # heuristics for amount of parallel reduction stream for DG/DB
        GROUP_SIZE_M = 32
        if N <= 8192:
            GROUP_SIZE_M = 64
        if N <= 4096:
            GROUP_SIZE_M = 96
        if N <= 2048:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256

        if dy.dtype == torch.float32:
            GROUP_SIZE_M = GROUP_SIZE_M // 2

        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device="cuda")
        t_args = {"dtype": x.dtype, "device": x.device}
        _dw = torch.empty((GROUP_SIZE_M, x.size(-1)), **t_args)
        dw = torch.empty((x.size(-1),), **t_args)
        dy = dy.contiguous()
        dx = torch.empty_like(dy)

        assert (
            dy.numel() == x.numel()
        ), "Something is wrong in the backward graph, possibly because of an inplace operation after the rmsnorm"

        num_warps = min(max(ctx.BLOCK_SIZE_N // 256, 1), 16)

        rms_norm_bwd_dx_fused[(M,)](
            dx, dy, _dw, x,
            weight,
            rstd,
            locks,
            x.stride(0),
            N,
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE_N,
            num_warps=num_warps
        )

        def grid(meta):
            return [triton.cdiv(N, meta["BLOCK_SIZE_N"])]

        rms_norm_bwd_dw[grid](
            _dw, dw,
            GROUP_SIZE_M,
            N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=64
        )

        dx = dx.reshape_as(dy)
        return dx, dw, None


class FusedRMSNorm(nn.Module):

    def __init__(self, normalized_shape, device='cuda', dtype=torch.float16, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        self.epsilon = eps

    def forward(self, x):
        return _RMSNorm.apply(x, self.weight, self.epsilon)

    def init_weights(self, *args, **kwargs):
        with torch.no_grad():
            if self.weight is not None:
                self.weight.fill_(1.0)
