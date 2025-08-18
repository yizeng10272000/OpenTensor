# codes/utils/gf2.py

import numpy as np
import torch
from typing import Optional

def to_bin(x, threshold=None):
    """
    Reduce any numpy/torch tensor to {0,1} (bitwise modulo 2).
    Floating-point numbers are first binarized (rounded or thresholded), while integers are directly & 1.
    """
    if isinstance(x, torch.Tensor):
        if x.dtype.is_floating_point:
            x = torch.round(x) if threshold is None else (x > threshold)
        return (x.to(torch.uint8) & 1)
    else:
        a = np.array(x, copy=False)
        if np.issubdtype(a.dtype, np.floating):
            a = np.rint(a) if threshold is None else (a > threshold)
        return (a.astype(np.uint8) & 1)


def add_mod2(a, b):
    """
    GF(2) addition, corresponding to the XOR operation.
    Supports numpy arrays or torch tensors.
    """
    a_bin = to_bin(a)
    b_bin = to_bin(b)
    if isinstance(a_bin, torch.Tensor) and isinstance(b_bin, torch.Tensor):
        return a_bin ^ b_bin
    else:
        return np.bitwise_xor(a_bin, b_bin)


def outer_mod2(a, b, c=None):
    """
    GF(2) outer product:
    - If c=None, returns the 2D outer product a ⊗ b
    - If c is provided, returns the 3D tensor outer product a ⊗ b ⊗ c
    The input vectors a, b, c can be numpy or torch tensors.
    """
    a_bin = to_bin(a).reshape(-1)
    b_bin = to_bin(b).reshape(-1)
    
    if c is None:
        if isinstance(a_bin, torch.Tensor) and isinstance(b_bin, torch.Tensor):
            return (a_bin.unsqueeze(1) & b_bin.unsqueeze(0)).to(torch.uint8)
        else:
            return np.outer(a_bin, b_bin) & 1
    else:
        c_bin = to_bin(c).reshape(-1)
        if isinstance(a_bin, torch.Tensor):
            # PyTorch third-order outer product
            return (a_bin[:, None, None] & b_bin[None, :, None] & c_bin[None, None, :]).to(torch.uint8)
        else:
            # NumPy third-order outer product
            return np.einsum('i,j,k->ijk', a_bin, b_bin, c_bin) & 1


def _to_bin_nd(x, threshold: Optional[float] = None):
    """
    Reduce any numpy/torch tensor to {0,1} (bitwise mod 2).
    """
    return to_bin(x, threshold=threshold)


@torch.no_grad()
def matrix_rank_mod2(M, threshold: Optional[float] = None) -> int:
    """
    Computes the rank of a matrix over GF(2).
    Supports numpy.ndarray and torch.Tensor (2D).
    """
    A = _to_bin_nd(M, threshold=threshold)
    if isinstance(A, torch.Tensor):
        if A.ndim != 2:
            raise ValueError("matrix_rank_mod2: input must be 2D")
        A = A.clone()
        m, n = A.shape
        r = 0
        for c in range(n):
            if r >= m:
                break
            piv = torch.nonzero(A[r:, c], as_tuple=False)
            if piv.numel() == 0:
                continue
            p = (r + piv[0, 0]).item()
            if p != r:
                A[[r, p]] = A[[p, r]]
            rows = torch.nonzero(A[:, c], as_tuple=False).squeeze(1)
            rows = rows[rows != r]
            if rows.numel() > 0:
                A[rows] ^= A[r]
            r += 1
        return int(r)
    else:
        if A.ndim != 2:
            raise ValueError("matrix_rank_mod2: input must be 2D")
        A = A.copy()
        m, n = A.shape
        r, c = 0, 0
        while r < m and c < n:
            pivot_rel = np.argmax(A[r:, c])
            if A[r + pivot_rel, c] == 0:
                c += 1
                continue
            p = r + pivot_rel
            if p != r:
                A[[r, p]] = A[[p, r]]
            idx = np.where(A[:, c] == 1)[0]
            idx = idx[idx != r]
            if idx.size:
                A[idx] ^= A[r]
            r += 1
            c += 1
        return int(r)


def terminate_rank_approx_gf2(tensor, axis: int = -1, threshold: Optional[float] = None) -> int:
    """
    Compute the "approximate termination penalty" for a rank-three tensor:
    Slice the tensor along an axis into a set of matrices, calculate the GF(2) rank of each slice and accumulate them.
    """
    X = _to_bin_nd(tensor, threshold=threshold)
    if isinstance(X, torch.Tensor):
        if X.ndim < 2:
            return 0
        axis_ = axis if axis >= 0 else X.ndim + axis
        perm = [i for i in range(X.ndim) if i != axis_] + [axis_]
        Xp = X.permute(*perm).contiguous()
        total = 0
        for k in range(Xp.shape[-1]):
            Mk = Xp[..., k]
            if Mk.ndim != 2:
                Mk = Mk.reshape(Mk.shape[0], -1)
            total += matrix_rank_mod2(Mk)
        return int(total)
    else:
        A = np.asarray(X)
        if A.ndim < 2:
            return 0
        axis_ = axis if axis >= 0 else A.ndim + axis
        A = np.moveaxis(A, axis_, -1)
        total = 0
        for k in range(A.shape[-1]):
            Mk = A[..., k]
            if Mk.ndim != 2:
                Mk = Mk.reshape(Mk.shape[0], -1)
            total += matrix_rank_mod2(Mk)
        return int(total)


# ----------------------------
# Added rank_gf2 alias, compatible with Trainer.py
rank_gf2 = matrix_rank_mod2
