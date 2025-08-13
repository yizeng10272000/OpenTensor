# codes/utils/gf2.py

import numpy as np
import torch

def _to_bin_nd(x, threshold=None):
    """
    把任意 numpy / torch 张量规约到 {0,1}（按位 mod 2）。
    浮点数会先二值化（round 或阈值），整型会直接 & 1。
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


@torch.no_grad()
def matrix_rank_mod2(M, threshold: float | None = None) -> int:
    """
    计算 GF(2) 上的矩阵秩。
    支持 numpy.ndarray 和 torch.Tensor（二维）。
    """
    A = _to_bin_nd(M, threshold=threshold)
    if isinstance(A, torch.Tensor):
        # Torch 版按列做高斯消元（XOR），只用行交换 + 异或，无比例缩放
        if A.ndim != 2:
            raise ValueError("matrix_rank_mod2: input must be 2D")
        A = A.clone()  # 就地修改的拷贝
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
        # NumPy 版
        if A.ndim != 2:
            raise ValueError("matrix_rank_mod2: input must be 2D")
        A = A.copy()
        m, n = A.shape
        r = 0
        c = 0
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


def terminate_rank_approx_gf2(tensor, axis: int = -1, threshold: float | None = None) -> int:
    """
    计算三阶张量的 “近似终止惩罚”：把张量沿着某个轴切成一组矩阵，
    对每个切片求 GF(2) 秩并累加（与原来实数域版本逐切片求秩再相加一致）。
    默认对最后一维做切片：tensor[..., k]。

    返回：∑_k rank_mod2(tensor_slice_k)
    """
    X = _to_bin_nd(tensor, threshold=threshold)
    # 统一到 numpy / torch 决策
    if isinstance(X, torch.Tensor):
        if X.ndim < 2:
            return 0
        axis_ = axis if axis >= 0 else X.ndim + axis
        # 把要切的轴移到最后，便于迭代
        perm = [i for i in range(X.ndim) if i != axis_] + [axis_]
        Xp = X.permute(*perm).contiguous()
        n_slices = Xp.shape[-1]
        total = 0
        for k in range(n_slices):
            Mk = Xp[..., k]  # 取一个矩阵切片
            if Mk.ndim != 2:
                # 若更高阶（很少见），把前面所有维展平成行
                Mk = Mk.reshape(Mk.shape[0], -1)
            total += matrix_rank_mod2(Mk)
        return int(total)
    else:
        A = np.asarray(X)
        if A.ndim < 2:
            return 0
        axis_ = axis if axis >= 0 else A.ndim + axis
        A = np.moveaxis(A, axis_, -1)
        n_slices = A.shape[-1]
        total = 0
        for k in range(n_slices):
            Mk = A[..., k]
            if Mk.ndim != 2:
                Mk = Mk.reshape(Mk.shape[0], -1)
            total += matrix_rank_mod2(Mk)
        return int(total)
