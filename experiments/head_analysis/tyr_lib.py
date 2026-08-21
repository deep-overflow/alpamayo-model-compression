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
