# B70_TARGET_FP8
"""B70 Fase S runtime helper: draft MTP LM head INT4 g128 sym.

Escribido por patch_draft_lmhead_int4.py dentro del contenedor. Provee la
cuantizacion one-time del lm_head fp16 compartido a GPTQ INT4 g128 sym y el
ruteo de las 4 pasadas del draft por ``int4_gemm_w4a16``. El target queda
fp16 (lossless).
"""
from __future__ import annotations

import os

import torch


def quantize_lmhead_to_int4(weight: torch.Tensor, group_size: int = 128):
    """Quantiza un lm_head fp16 [N, K] a GPTQ INT4 g128 sym.

    Returns (qweight, scales, qzeros, group_size):
      qweight: int32 [K//8, N] en layout NT (strides[-2] == 1), nibbles
               secuenciales LSB-first, valor almacenado = q + 8 (q in [-8, 7])
      scales:  fp16 [K//group_size, N]
      qzeros:  int8 tensor([8])  -> rama simetrica de int4_gemm_w4a16
    """
    device = weight.device
    N, K = weight.shape
    num_groups = K // group_size
    chunk = 4096
    shifts = torch.tensor(
        [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device
    )
    parts = []
    scale_parts = []
    for i in range(0, N, chunk):
        wc = weight[i : i + chunk].float()  # [c, K] fp32 (chunked: no 5 GB temp)
        wg = wc.view(wc.shape[0], num_groups, group_size)
        # B70_DRAFT_INT4_MSE=1: per-group clip factor that minimizes the squared error (same sym int4 format; opt-in)
        maxabs = wg.abs().amax(dim=-1).clamp_min(1e-8)
        scale = maxabs / 7.0
        if __import__('os').environ.get('B70_DRAFT_INT4_MSE', '0') == '1':
            _b70_best = None
            for _b70_s in (1.0, 0.975, 0.95, 0.925, 0.9, 0.875, 0.85, 0.825, 0.8, 0.775, 0.75, 0.725, 0.7):
                _b70_sc = maxabs * (_b70_s / 7.0)
                _b70_q = (wg / _b70_sc.unsqueeze(-1)).round().clamp(-8, 7)
                _b70_err = ((_b70_q * _b70_sc.unsqueeze(-1) - wg) ** 2).sum(dim=-1)
                if _b70_best is None:
                    _b70_best, scale = _b70_err, _b70_sc
                else:
                    _b70_m = _b70_err < _b70_best
                    _b70_best = torch.where(_b70_m, _b70_err, _b70_best)
                    scale = torch.where(_b70_m, _b70_sc, scale)
            del _b70_best, _b70_sc, _b70_q, _b70_err
        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)
        stored = q + 8  # 0..15
        qv = stored.view(wc.shape[0], num_groups, group_size // 8, 8)
        packed = (qv << shifts).sum(dim=-1).to(torch.int32).reshape(
            wc.shape[0], K // 8
        )
        parts.append(packed)
        scale_parts.append(scale.half())
    qweight_contig = torch.cat(parts, dim=0)  # [N, K//8] int32
    scales_contig = torch.cat(scale_parts, dim=0)  # [N, g] fp16
    # Layout NT requerido por la op (strides[-2] == 1) + scales contiguas
    qweight = qweight_contig.t()  # [K//8, N], strides (1, K//8)
    scales = scales_contig.t().contiguous()  # [g, N]
    qzeros = torch.tensor([8], dtype=torch.int8, device=device)
    return qweight, scales, qzeros, group_size


def int4_lmhead_logits(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Logits [.., vocab] via int4_gemm_w4a16 (mismo formato que el cuerpo)."""
    flat = x.reshape(-1, x.shape[-1])
    logits = torch.ops._xpu_C.int4_gemm_w4a16(
        flat, qweight, None, scales, qzeros, group_size, None
    )
    return logits.reshape(*x.shape[:-1], qweight.shape[1])


@torch.no_grad()
def build_draft_lmhead_int4(model) -> None:
    """Cuantiza el lm_head fp16 compartido del draft (one-time, no-op si no
    hay env gate o si ya se construyo). Almacena en model._b70_lmhead_int4."""
    if os.environ.get("B70_DRAFT_LMHEAD_INT4") != "1":
        return
    if getattr(model, "_b70_lmhead_int4_tp_blocked", False):
        return
    parallel_config = getattr(
        getattr(model, "vllm_config", None), "parallel_config", None
    )
    tp_size = getattr(parallel_config, "tensor_parallel_size", None)
    if tp_size is None:
        model._b70_lmhead_int4_tp_blocked = True
        print("[B70] draft LM head INT4: TP unknown; skipping "
              "(fail-closed, issue #9)", flush=True)
        return
    if tp_size > 1:
        model._b70_lmhead_int4_tp_blocked = True
        print("[B70] draft LM head INT4: TP>1 (tp=%d) detected; skipping "
              "(C1/TP1 only, issue #9)" % tp_size, flush=True)
        return
    if getattr(model, "_b70_lmhead_int4", None) is not None:
        return
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    _pre = getattr(head, "_b70_draft_int4", None)
    if _pre is not None:
        model._b70_lmhead_int4 = tuple(_pre[:4])
        if _pre[4] is not None:
            model._b70_draft_vocab_ids = _pre[4]
            model._b70_draft_vocab_full = _pre[5]
        print("[B70] draft LM head INT4: using prebuilt head", flush=True)
        return
    if weight is None:
        print("[B70] draft LM head INT4: lm_head.weight no disponible; "
              "draft sigue por fp16", flush=True)
        return
    print("[B70] draft LM head INT4: cuantizando lm_head fp16 "
          f"{tuple(weight.shape)} -> INT4 g128 sym (one-time)", flush=True)
    # B70_DRAFT_VOCAB
    _vf = os.environ.get("B70_DRAFT_VOCAB")
    _w = weight.detach()
    if _vf:
        import json as _json
        _ids = torch.tensor(sorted(_json.load(open(_vf))), dtype=torch.long, device=_w.device)
        model._b70_draft_vocab_ids = _ids
        model._b70_draft_vocab_full = _w.shape[0]
        _w = _w.index_select(0, _ids).contiguous()
        print(f"[B70] draft vocab: {_ids.numel()} / {weight.shape[0]} rows", flush=True)
    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(_w)
    model._b70_lmhead_int4 = (qweight, scales, qzeros, group_size)
    fp16_bytes = weight.numel() * weight.element_size()
    int4_bytes = qweight.numel() * qweight.element_size() + (
        scales.numel() * scales.element_size()
    )
    print(f"[B70] draft LM head INT4: listo. {fp16_bytes/1e9:.2f} GB fp16 -> "
          f"{int4_bytes/1e9:.2f} GB INT4 (ahorro "
          f"{(fp16_bytes - int4_bytes)/1e6:.1f} MB/lectura)", flush=True)


def draft_lmhead_int4_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """Logits del draft via la copia INT4 (4 pasadas/paso -> 0.66 GB c/u)."""
    qweight, scales, qzeros, group_size = model._b70_lmhead_int4
    logits = int4_lmhead_logits(
        hidden_states, qweight, scales, qzeros, group_size
    )
    _ids = getattr(model, "_b70_draft_vocab_ids", None)
    if _ids is not None:  # B70_DRAFT_VOCAB
        full = logits.new_full((*logits.shape[:-1], model._b70_draft_vocab_full), float("-inf"))
        full.index_copy_(full.dim() - 1, _ids, logits)
        logits = full
    org = getattr(getattr(model, "logits_processor", None), "org_vocab_size", None)
    if org is not None and logits.shape[-1] > org:
        logits = logits[..., :org]
    return logits
