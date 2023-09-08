import torch

import triton
import triton.language as tl



BLOCK_M = 32
BLOCK_N = 64


_configs = [
    triton.Config({}, num_warps=1),
    triton.Config({}, num_warps=2),
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
    triton.Config({}, num_warps=16),
]


@triton.autotune(
    configs=_configs,
    key=["M", "N", "is_fp16"],
)
@triton.jit
def k_silu_mul_fw(
    Y, X,
    N_c,
    M, N,
    is_fp16: tl.constexpr,  # autotune
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    rows = row_id * BLOCK_M + tl.arange(0, BLOCK_M)

    col_id = tl.program_id(axis=1)
    cols = col_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # pointers starting point
    x1_ptrs = X + rows[:, None] * N + cols[None, :]
    x2_ptrs = X + rows[:, None] * N + cols[None, :] + N_c
    y_ptrs = Y + rows[:, None] * N_c + cols[None, :]

    block_mask = (rows[:, None] < M) & (cols[None, :] < N_c)
    x1 = tl.load(x1_ptrs, mask=block_mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptrs, mask=block_mask, other=0.0).to(tl.float32)

    x = x2 * x1 * tl.sigmoid(x1)
    tl.store(y_ptrs, x, mask=block_mask)  # output


@triton.autotune(
    configs=_configs,
    key=["M", "N", "is_fp16"],
)
@triton.jit
def k_silu_mul_bw(
    GRAD_OUT, X, GRAD_IN,
    N_c,
    M, N,
    is_fp16: tl.constexpr,  # autotune
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    rows = row_id * BLOCK_M + tl.arange(0, BLOCK_M)

    col_id = tl.program_id(axis=1)
    cols = col_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # pointers starting point
    x1_ptrs = X + rows[:, None] * N + cols[None, :]
    x2_ptrs = X + rows[:, None] * N + cols[None, :] + N_c
    grad_out_ptr = GRAD_OUT + rows[:, None] * N_c + cols[None, :]
    grad_x1_ptr = GRAD_IN + rows[:, None] * N + cols[None, :]
    grad_x2_ptr = GRAD_IN + rows[:, None] * N + cols[None, :] + N_c

    block_mask = (rows[:, None] < M) & (cols[None, :] < N_c)
    x1 = tl.load(x1_ptrs, mask=block_mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptrs, mask=block_mask, other=0.0).to(tl.float32)
    g_out = tl.load(grad_out_ptr, mask=block_mask, other=0.0).to(tl.float32)

    x3 = x1 * tl.sigmoid(x1)
    dx3 = g_out * x2
    dx2 = g_out * x3
    # x4 = x2 * x3
    sigm = tl.sigmoid(x1)
    dx1 = (dx3 * sigm * (1. + x1 * (1. - sigm)))

    tl.store(grad_x1_ptr, dx1, mask=block_mask)
    tl.store(grad_x2_ptr, dx2, mask=block_mask)


@triton.autotune(
    configs=_configs,
    key=["M", "N", "is_fp16"],
)
@triton.jit
def k_silu_mul_bw_y(
    GRAD_OUT, X, GRAD_IN, Y,
    N_c,
    M, N,
    is_fp16: tl.constexpr,  # autotune
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    rows = row_id * BLOCK_M + tl.arange(0, BLOCK_M)

    col_id = tl.program_id(axis=1)
    cols = col_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # pointers starting point
    x1_ptrs = X + rows[:, None] * N + cols[None, :]
    x2_ptrs = X + rows[:, None] * N + cols[None, :] + N_c

    grad_out_ptr = GRAD_OUT + rows[:, None] * N_c + cols[None, :]
    grad_x1_ptr = GRAD_IN + rows[:, None] * N + cols[None, :]
    grad_x2_ptr = GRAD_IN + rows[:, None] * N + cols[None, :] + N_c

    block_mask = (rows[:, None] < M) & (cols[None, :] < N_c)
    x1 = tl.load(x1_ptrs, mask=block_mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptrs, mask=block_mask, other=0.0).to(tl.float32)
    g_out = tl.load(grad_out_ptr, mask=block_mask, other=0.0).to(tl.float32)

    x3 = x1 * tl.sigmoid(x1)
    dx3 = g_out * x2
    dx2 = g_out * x3

    x4 = x2 * x3
    y_ptrs = Y + rows[:, None] * N_c + cols[None, :]
    tl.store(y_ptrs, x4, mask=block_mask)

    sigm = tl.sigmoid(x1)
    dx1 = (dx3 * sigm * (1. + x1 * (1. - sigm)))

    tl.store(grad_x1_ptr, dx1, mask=block_mask)
    tl.store(grad_x2_ptr, dx2, mask=block_mask)



class _silu_mul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        B, S, N = x.shape
        x_ = x.reshape(-1, x.shape[-1]).contiguous()

        M, N_c = B * S, N // 2
        y = torch.empty((M, N_c), device=x.device, dtype=x.dtype)

        def grid(meta):
            return (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(N_c, meta["BLOCK_N"]),
            )

        k_silu_mul_fw[grid](
            y, x_,
            N_c,
            M, N,
            x.dtype == torch.float16,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(x)

        return y.reshape(B, S, N_c)

    @staticmethod
    def backward(
        ctx, grad_out
    ):
        (x,) = ctx.saved_tensors

        B, S, N = x.shape
        x_ = x.reshape(-1, x.shape[-1]).contiguous()

        M, N_c = B * S, N // 2
        grad_out_ = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grad_in = torch.empty((M, N), device=x.device, dtype=x.dtype)

        def grid(meta):
            return (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(N_c, meta["BLOCK_N"]),
            )

        k_silu_mul_bw[grid](
            grad_out_, x_, grad_in,
            N_c,
            M, N,
            x.dtype == torch.float16,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        return grad_in.reshape(B, S, N)


def silu_mul(
    x: torch.Tensor,
):
    return _silu_mul.apply(x)
