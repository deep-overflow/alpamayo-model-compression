"""OSSCAR-style local pruning with weight reconstruction, for the Tyr baseline.

`local_prune_core` is ported from Tyr-the-Pruner (github.com/AMD-AGI/Tyr-the-Pruner,
Apache-2.0, src/local_pruner.py) with the distributed plumbing removed; the math is
unchanged: second-order group removal on the input side of o_proj / down_proj with
H = sum_tokens x x^T, followed by least-squares reconstruction of the surviving
groups (W_kept = inv(H[kept]) @ G[kept], G = H @ W). o_proj groups are Q heads
(group_size = head_dim, one removal round, mha_update_iter=1 upstream); down_proj
groups are single channels (16 removal rounds, mlp_update_iter=16 upstream).

plans/2026-08-20_tyr-baseline.md.
"""

import numpy as np
import torch


class HessianHook:
    """Accumulate H = sum_tokens x x^T (fp32) at a linear module's input."""

    def __init__(self, module):
        d = module.in_features
        self.H = torch.zeros((d, d), device=module.weight.device, dtype=torch.float32)
        self.n_tokens = 0
        self.handle = module.register_forward_pre_hook(self._hook)

    @torch.no_grad()
    def _hook(self, _module, args):
        x = args[0].reshape(-1, args[0].shape[-1]).float()  # (T, d)
        self.H += x.t() @ x
        self.n_tokens += x.shape[0]

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def prune_levels(module, H, num_groups, keep_counts, update_iter, damp=1e-2):
    """Prune `module.weight` to each keep count, with reconstruction.

    Mirrors LocalPruner.prune(): dead-input zeroing, Hessian damping (`damp` x the
    mean diagonal; upstream 1e-2), G = H @ B, then one local_prune_core solve per level. Returns
    {keep_count: (out, in) float32 tensor}; the module itself is not modified.
    """
    W = module.weight.data.clone().float()          # (out, in)
    B = W.t().contiguous()                          # (in, out)
    H = H.clone()
    dead = torch.diag(H) == 0
    B[dead, :] = 0
    # upstream damping is 1e-2 of the mean diagonal; on vision-dominated prefill H is
    # numerically rank-deficient (cond ~1e34-1e38, hundreds of ~0 eigenvalues) and
    # 1e-2 lets inv(H) blow the reconstruction up 2-12x (outputs/tyr_hdiag.json),
    # so the damping is a parameter: 1.0 keeps the kept-weight change at 10-16%.
    H += torch.eye(B.shape[0], device=H.device) * torch.mean(torch.diag(H)) * damp
    G = H @ B
    out = {}
    for keep in keep_counts:
        if keep >= num_groups:
            B_sol = B.clone()                       # sparsity 0.0 branch upstream
        else:
            B_sol, _ = local_prune_core(B.clone(), H, G, num_groups, keep, update_iter)
        out[keep] = B_sol.t().contiguous()
    return out


@torch.no_grad()
def local_prune_core(W, H, G, num_total_groups: int, num_groups_to_remain: int,
                     update_iter: int = 1):
    """Ported from Tyr-the-Pruner src/local_pruner.py (Apache-2.0), math unchanged."""
    device = W.device
    cin, cout = W.shape
    group_size = int(cin / num_total_groups)

    H_inv = torch.linalg.inv(H)
    W_g = W.reshape(num_total_groups, group_size, cout)
    group_abs_sum = torch.sum(torch.abs(W_g), dim=(1, 2))
    pruned_group_mask = (group_abs_sum <= 1e-12)
    num_already_zero = int(pruned_group_mask.sum().item())

    if num_already_zero > 0:
        zero_idx = torch.cat([
            torch.arange(g * group_size, (g + 1) * group_size, device=device)
            for g in torch.nonzero(pruned_group_mask, as_tuple=False).flatten()
        ])
        H_inv[zero_idx, :] = 0
        H_inv[:, zero_idx] = 0
        W = H_inv @ G
        if (num_total_groups - num_groups_to_remain - num_already_zero) <= 0:
            W[zero_idx, :] = 0
            return W, 0.0

    remaining_to_prune = int(num_total_groups - num_groups_to_remain
                             - int(pruned_group_mask.sum().item()))
    if remaining_to_prune <= 0:
        kept_mask = (~torch.repeat_interleave(pruned_group_mask, group_size))
        W_kept = torch.zeros_like(W)
        W_kept[kept_mask, :] = torch.linalg.inv(H[kept_mask][:, kept_mask]) @ G[kept_mask, :]
        prune_loss = torch.sum(-W_kept * G + 0.5 * W_kept * (H @ W_kept)).detach().item()
        return W_kept, prune_loss

    update_rounds = max(int(min(update_iter, remaining_to_prune)), 1)
    base, extra = divmod(remaining_to_prune, update_rounds)
    groups_to_prune_each_round = torch.full((update_rounds,), base, dtype=torch.int,
                                            device=device)
    if extra > 0:
        groups_to_prune_each_round[:extra] += 1

    for round_id in range(update_rounds):
        if group_size > 1:
            obj_mat = torch.zeros_like(W)
            for g in range(num_total_groups):
                if pruned_group_mask[g]:
                    continue
                sl = slice(g * group_size, (g + 1) * group_size)
                H_block = torch.linalg.inv(H_inv[sl, sl])  # ~= H[sl, sl]
                obj_mat[sl, :] = (H_block @ W[sl, :] / 2.0) + G[sl, :]
        else:
            diag_Hinv = torch.diag(H_inv)
            safe_den = (pruned_group_mask.to(W.dtype) + diag_Hinv).clamp_min(1e-12)
            obj_mat = (1.0 / safe_den)[:, None] * (W / 2.0) + G

        obj_val = (W * obj_mat).reshape(num_total_groups, group_size, cout).sum(dim=(1, 2))
        obj_val_masked = obj_val + 1e20 * pruned_group_mask.to(obj_val.dtype)

        sorted_groups = torch.argsort(obj_val_masked)
        k = int(groups_to_prune_each_round[round_id].item())
        pick_groups = sorted_groups[:k]
        pick_idx = torch.cat([
            torch.arange(g * group_size, (g + 1) * group_size, device=device)
            for g in pick_groups
        ])

        Hinv_block_inv = torch.linalg.inv(H_inv[pick_idx][:, pick_idx])  # ~= H[pick_idx, pick_idx]
        W -= H_inv[:, pick_idx] @ Hinv_block_inv @ W[pick_idx, :]
        W[pick_idx, :] = 0
        H_inv -= H_inv[:, pick_idx] @ Hinv_block_inv @ H_inv[pick_idx, :]
        H_inv[pick_idx, :] = 0
        H_inv[:, pick_idx] = 0
        pruned_group_mask[pick_groups] = True

    W_pruned = torch.zeros_like(W)
    kept_mask = (~torch.repeat_interleave(pruned_group_mask, repeats=group_size))
    W_pruned[kept_mask, :] = torch.linalg.inv(H[kept_mask][:, kept_mask]) @ G[kept_mask, :]
    prune_loss = torch.sum(-W_pruned * G + 0.5 * W_pruned * (H @ W_pruned)).detach().item()
    return W_pruned, prune_loss


def level_keeps(total_groups, base_cut, step, num_levels):
    """{level: keep_count} for levels centered on base_cut, cut = base_cut + l*step.

    Levels whose cut leaves the [1, total_groups-1] range are dropped, mirroring the
    upstream min_level/max_level clamping.
    """
    half = num_levels // 2
    out = {}
    for lv in range(-half, half + 1):
        cut = base_cut + lv * step
        if 0 < cut < total_groups:
            out[lv] = total_groups - cut
    return out


def dual_scores(imp):
    """dual = max(rank I_traj, rank I_CoC) per layer, the same call chain make_slim uses
    for dual_u40_v2 (run_cocsafe.rank_norm), so level-0 cuts reproduce it bit for bit."""
    from run_cocsafe import rank_norm
    sq = np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"]))
    sm = np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"]))
    return sq, sm


def cut_lowest(scores_row, cut):
    """Units removed at a cut count: mask_lib.select_mask_ratios' rule (np.argsort, lowest)."""
    return np.argsort(scores_row)[:cut]


@torch.no_grad()
def reconstruct_levels(module, H, keep_sets, damp=1e-2):
    """OSSCAR reconstruction for PRESCRIBED kept input columns (no selection).

    Same dead-input zeroing, damping and G = H @ B as prune_levels; the kept block is
    solved as W_kept = H_kk^{-1} G_k, i.e. local_prune_core's final reconstruction step
    applied to an externally chosen kept set. keep_sets: {key: kept column indices}.
    """
    W = module.weight.data.clone().float()          # (out, in)
    B = W.t().contiguous()                          # (in, out)
    H = H.clone()
    dead = torch.diag(H) == 0
    B[dead, :] = 0
    H += torch.eye(B.shape[0], device=H.device) * torch.mean(torch.diag(H)) * damp
    G = H @ B
    out = {}
    for key, kept in keep_sets.items():
        k = torch.as_tensor(np.asarray(kept), device=H.device, dtype=torch.long)
        Bk = torch.zeros_like(B)
        Bk[k, :] = torch.linalg.solve(H[k][:, k], G[k, :])
        out[key] = Bk.t().contiguous()
    return out


# ---------------------------------------------------------------------------
# RAC (chain-of-thought reconstruction) additions -- plans/2026-08-25_cot-reconstruction.md
#
# The Tyr path above accumulates ONE Hessian over the whole fused-prompt prefill
# ("hessian_tokens": "full fused prompt prefill, no labels"), i.e. exactly the
# input-only reconstruction arXiv:2509.12464 argues against. Because
# H = sum_t x_t x_t^T is additive over token subsets, keeping the streams apart
# lets any mixture be formed afterwards without re-running a forward.
# ---------------------------------------------------------------------------


class StreamHessianHook:
    """HessianHook split by token stream (and calibration fold).

    One buffer per key: H[k] = sum_{t in masks[k]} x_t x_t^T (fp32) at the module
    input. The caller fills `masks` with {key: (T,) bool} right before each
    forward; keys absent from `masks` are skipped for that forward.
    """

    def __init__(self, module, keys, device=None):
        d = module.in_features
        dev = device if device is not None else module.weight.device
        self.H = {k: torch.zeros((d, d), device=dev, dtype=torch.float32) for k in keys}
        self.n = {k: 0 for k in keys}
        self.masks = {}
        self.handle = module.register_forward_pre_hook(self._hook)

    @torch.no_grad()
    def _hook(self, _module, args):
        x = args[0].reshape(-1, args[0].shape[-1])       # (T, d)
        for k, m in self.masks.items():
            xs = x[m].float()                            # (n_k, d)
            if xs.shape[0] == 0:
                continue
            self.H[k] += xs.t() @ xs
            self.n[k] += xs.shape[0]

    def free(self):
        self.H = {}

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def mix_hessians(H, n, weights):
    """H(w) = sum_s w_s * H_s / N_s -- the per-token-mean mixture.

    Overall scale is irrelevant: the least-squares solve and the relative damping
    (`damp` x mean diag) are both invariant under H -> cH, so `weights` only has to
    be right up to a constant. Returns None if every requested stream is empty.
    """
    acc = None
    for s, w in weights.items():
        if w == 0.0 or n.get(s, 0) == 0:
            continue
        term = H[s].mul(w / n[s])
        acc = term if acc is None else acc.add_(term)
    return acc


@torch.no_grad()
def dense_energy(W, H_eval):
    """tr(W H W^T) -- the recon_error denominator, computed once per (module, H)."""
    return (W @ H_eval).mul_(W).sum().clamp_min(0.0)


@torch.no_grad()
def recon_error(W, W_hat, H_eval, denom=None):
    """rel_err = sqrt( tr(D H D^T) / tr(W H W^T) ), D = W - W_hat, W (out, in).

    Equals ||(W - W_hat) X||_F / ||W X||_F for any X with X X^T = H, so a held-out
    error needs only the eval-fold Hessian -- activations are never stored.
    """
    diff = W - W_hat
    num = (diff @ H_eval).mul_(diff).sum().clamp_min(0.0)
    if denom is None:
        denom = dense_energy(W, H_eval)
    if float(denom) <= 0.0:
        return float("nan")
    return float(torch.sqrt(num / denom))


@torch.no_grad()
def kept_groups(W_sol, num_groups):
    """Group indices surviving in a solution (removed groups are exactly zero).

    Same test run_tyr_supernet.py:180 asserts on, applied to an (out, in) tensor.
    """
    gs = W_sol.shape[1] // num_groups
    alive = W_sol.abs().sum(0).reshape(num_groups, gs).sum(1) != 0
    return torch.nonzero(alive).flatten()


@torch.no_grad()
def mask_only(W, kept_idx, group_size=1):
    """W with every non-kept input column zeroed -- pruning without reconstruction."""
    out = torch.zeros_like(W)
    if group_size > 1:
        cols = (kept_idx[:, None] * group_size
                + torch.arange(group_size, device=W.device)).reshape(-1)
    else:
        cols = kept_idx
    out[:, cols] = W[:, cols]
    return out


@torch.no_grad()
def energy_overlap(H_a, H_b, k=512, niter=4):
    """Fraction of H_b's energy captured by H_a's top-k eigenspace.

    tr(U^T H_b U) / tr(H_b) with U the leading k eigenvectors of H_a (PSD, so a
    randomized SVD gives them). 1.0 means the two streams live in the same subspace
    and mixing them cannot change the solve -- gate G1.
    """
    q = min(k + 32, H_a.shape[0] - 1)
    U, _, _ = torch.svd_lowrank(H_a, q=q, niter=niter)
    ut = U[:, :k].t()                                    # (k, d)
    num = (ut @ H_b).mul_(ut).sum()
    den = torch.diagonal(H_b).sum()
    return float(num / den) if float(den) > 0 else float("nan")


@torch.no_grad()
def cond_stats(H):
    """Cheap conditioning diagnostics: trace, squared Frobenius norm, effective rank.

    pr_rank = tr(H)^2 / ||H||_F^2 in [1, d] is the participation ratio, the effective
    number of directions carrying the energy. Chosen over eigvalsh because a 12288^2
    eigendecomposition per (layer, mixture) would dominate the run; the exact
    eigen-based version for four reference layers is in outputs/tyr_hdiag.json.
    """
    tr = float(torch.diagonal(H).sum())
    fro2 = float(H.pow(2).sum())
    return {"trace": tr, "fro2": fro2,
            "pr_rank": (tr * tr / fro2) if fro2 > 0 else float("nan")}
