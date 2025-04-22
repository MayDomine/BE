import torch
from .lao import _flash_attn_forward, _flash_attn_backward
from flash_attn.flash_attn_interface import (
    _flash_attn_forward as _flash_attn_forward_cuda,
    _flash_attn_varlen_forward as _flash_attn_varlen_forward_cuda,
)
from flash_attn.flash_attn_interface import (
    _flash_attn_backward as _flash_attn_backward_cuda,
    _flash_attn_varlen_backward as _flash_attn_varlen_backward_cuda,
)
import inspect


@torch.compile
def triton_scale_out(acc_o, m_i, lse_i):
    o_scale = torch.exp(m_i - lse_i)
    o_scale = o_scale.unsqueeze(-1).transpose(1, 2)
    acc_o = acc_o * o_scale
    return acc_o


@torch.compile
def cuda_scale_out_lse_helper(
    o,
    lse,
    o_i,
    lse_i,
):
    o_i = o_i.to(torch.float32)
    lse_i = lse_i.transpose(-2, -1).unsqueeze(dim=-1)
    o = o - torch.sigmoid(lse_i - lse) * (o - o_i)
    lse = lse - torch.nn.functional.logsigmoid(lse - lse_i)
    return o, lse


def record_stream(*tensorlist):
    for t in tensorlist:
        t.record_stream(torch.cuda.current_stream())
    return tensorlist


def inter_normal_attn(q, k, v, m_i, lse_i, acc_o, softmax_scale=1.0, mask_bias=None):
    m_i = m_i.to(q.dtype) if m_i is not None else None
    qk = q @ k.transpose(-2, -1) * softmax_scale
    if mask_bias is not None:
        qk = torch.masked_fill(
            qk,
            not mask_bias,
            torch.scalar_tensor(float("-10000"), device=qk.device, dtype=qk.dtype),
        )

    m_ij = torch.max(qk, dim=-1, keepdim=True)[0]
    if m_i is not None:
        m_ij = torch.maximum(m_ij, m_i)
    p = torch.exp(qk - m_ij)
    if mask_bias is not None:
        p = torch.masked_fill(
            p,
            not mask_bias,
            torch.scalar_tensor(float("0"), device=qk.device, dtype=qk.dtype),
        )
    l_ij = torch.sum(p, dim=-1, keepdim=True)
    if acc_o is not None:
        acc_o_scale = torch.exp(m_i - m_ij)
        pv = (p @ v).to(dtype=torch.float32)
        acc_o = pv + acc_o_scale * acc_o
    else:
        acc_o = (p @ v).to(dtype=torch.float32)

    if lse_i is None:
        lse_i = torch.log(l_ij + 1e-5) + m_ij
    else:
        lse_i = torch.log(torch.exp(lse_i - m_ij) + l_ij + 1e-5) + m_ij
    return acc_o, m_ij, lse_i


def inter_normal_attn_backward(
    do, q, k, v, delta, lse, d_q, d_k, d_v, softmax_scale, mask_bias
):
    # ensure q,k,v with shape [b,n,s,d]
    qk = q @ k.transpose(-2, -1) * softmax_scale
    if mask_bias is not None:
        qk = torch.masked_fill(
            qk,
            not mask_bias,
            torch.scalar_tensor(float("-10000"), device=qk.device, dtype=qk.dtype),
        )
    p = torch.exp(qk - lse)
    if mask_bias is not None:
        p = torch.masked_fill(
            p,
            not mask_bias,
            torch.scalar_tensor(float("0"), device=qk.device, dtype=qk.dtype),
        )
    d_v += p.transpose(-2, -1) @ do
    d_p = do @ v.transpose(-2, -1)
    softmax_scale = softmax_scale
    d_s = p * (d_p - delta) * softmax_scale
    d_q[:] = d_s @ k
    d_k += d_s.transpose(-2, -1) @ q


def inter_flash_attn_triton(
    q, k, v, m_i, lse_i, acc_o, softmax_scale=1.0, mask_bias=None
):
    b, s, n, d = q.shape
    if m_i is None:
        m_i = (
            -torch.ones((b, n, s), dtype=torch.float32, device="cuda") * torch.inf
        ).contiguous()
    if lse_i is None:
        lse_i = (
            -torch.ones((b, n, s), dtype=torch.float32, device="cuda") * torch.inf
        ).contiguous()
    if acc_o is None:
        acc_o = torch.zeros((b, s, n, d), dtype=torch.float32, device="cuda")
    acc_o, lse_i, m_ij, _ = _flash_attn_forward(
        q,
        k,
        v,
        m_i,
        lse_i,
        acc_o.to(dtype=torch.float32),
        causal=False,
        bias=mask_bias,
        softmax_scale=softmax_scale,
    )
    return acc_o, m_ij, lse_i


def inter_flash_attn_backward_triton(
    do, q, k, v, delta, lse, dq, dk, dv, softmax_scale, mask_bias
):
    _flash_attn_backward(
        do,
        q,
        k,
        v,
        delta,
        lse,
        dq,
        dk,
        dv,
        softmax_scale=softmax_scale,
        bias=mask_bias,
    )


def inter_flash_cuda_fwd(q, k, v, o, lse, softmax_scale=1.0, causal=False, sliding_window=None, cu_seqlen=None):
    if sliding_window:
        window_size = sliding_window
    else:
        window_size = (-1, -1) 
    if cu_seqlen is None:
        o_i, _, _, _, _, lse_i, _, _ = _flash_attn_forward_cuda(
            q,
            k,
            v,
            0.0,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=None,
            return_softmax=False,
        )
    else:
        o_i, _, _, _, _, lse_i, _, _ = _flash_attn_varlen_forward_cuda(
            q,
            k,
            v,
            cu_seqlen,
            cu_seqlen,
            q.shape[1],
            k.shape[1],
            0.0,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=None,
            return_softmax=False,
            block_table={},
        )
    if o is None:
        o = o_i.to(torch.float32)
        lse = lse_i.transpose(-2, -1).unsqueeze(dim=-1).contiguous()
    else:
        if q.shape[1] == k.shape[1] // 2:
            half_seqlen = o.shape[1] // 2

            o[:, half_seqlen:], lse[:, half_seqlen:] = cuda_scale_out_lse_helper(
                o[:, half_seqlen:], lse[:, half_seqlen:], o_i, lse_i
            )
        elif lse.shape[1] == lse_i.shape[2] + 1:
            o[:, 1:], lse[:, 1:] = cuda_scale_out_lse_helper(
                o[:, 1:], lse[:, 1:], o_i, lse_i
            )
        else:
            o, lse = cuda_scale_out_lse_helper(o, lse, o_i, lse_i)
    return o, lse


def inter_flash_cuda_bwd(
    do,
    q,
    k,
    v,
    o,
    lse,
    dq,
    dk,
    dv,
    softmax_scale,
    causal=False,
    deterministic=False,
    sliding_window=None,
    cu_seqlen=None
):
    if len(o.shape) == 3:
        # use sum(o_i * gradoutput) as delta and pass a empty out to flash backward
        # this feature requires a build of this PR: https://github.com/Dao-AILab/flash-attention/pull/905
        delta = o
        o = torch.empty_like(q)
    else:
        delta = None
    
    if sliding_window:
        window_size = sliding_window
    else:
        window_size = (-1, -1) 
    if cu_seqlen is not None:
        cu_seqlen_args = (cu_seqlen, cu_seqlen, q.shape[0], k.shape[0])
    else:
        cu_seqlen_args = ()
    bwd_func = _flash_attn_backward_cuda if cu_seqlen is None else _flash_attn_varlen_backward_cuda
    if delta is not None:
        assert (
            delta.shape[2] >= 128
        ), "optimize_bwd_comm is not supported for 128 or less sub-sequence length"
        res = bwd_func(
            do,
            q,
            k,
            v,
            o,
            lse,
            dq,
            dk,
            dv,
            *cu_seqlen_args,
            0.0,
            softmax_scale,
            causal,
            window_size,
            None,
            deterministic,  # determin
            None,
            softmax_d=delta,
        )
    else:
        res = bwd_func(
            do,
            q,
            k,
            v,
            o,
            lse,
            dq,
            dk,
            dv,
            *cu_seqlen_args,
            0.0,
            softmax_scale,
            causal,
            window_size,
            None,
            deterministic,
            None,
            )
    return res
