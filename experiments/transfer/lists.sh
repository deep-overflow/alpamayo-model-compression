# Shared file-list generators for the transfer scripts. Sourced, not executed.
#
# Expects the caller to have set: HUB, PRE, OUT, PY, MODEL_REV, and to provide a
# `has <tier>` predicate. Every function prints paths RELATIVE to the root named in
# its comment, so the caller can hand them straight to `rsync --files-from`.

# Byte total of a --files-from list, so a preview reports volume and not just names.
total_of() {  # total_of <root> <listfile>
  awk -v root="$1" '{printf "%s/%s\n", root, $0}' "$2" \
    | tr '\n' '\0' | du -cLhs --files0-from=- 2>/dev/null | tail -1
}

# --- relative to $HUB: the pinned snapshot only, not the other five ---------------
# Listed file-by-file rather than as a directory so --files-from cannot recurse into
# a sibling snapshot's blobs; -L then dereferences each snapshot symlink into a real
# file on the target, which from_pretrained reads the same way.
#
# `.no_exist` travels too. Those empty markers record "this optional file is genuinely
# absent upstream", and offline mode needs them: without one, huggingface_hub cannot
# tell "not cached" from "does not exist" and raises LocalEntryNotFoundError instead of
# reporting the file absent.
#
# Three repos, not one. Loading Alpamayo's *config* builds a processor from
# `vlm_name_or_path` = nvidia/Cosmos-Reason2-8B, whose own processor chains to
# Qwen/Qwen3-VL-8B-Instruct. Neither chain step needs weights -- only configs,
# tokenizer and preprocessor files -- but every one of them must be cached, because
# HF_HUB_OFFLINE=1 turns a missing file into a hard error. Missing the Qwen step is
# what killed NEURON job 890907; missing `.no_exist` killed 890894/890895.
#
# `_repo_meta <dir> [pinned_rev]` emits refs, markers, and the snapshot's non-weight
# files. With no pinned rev it resolves the revision through refs/main rather than
# guessing at the snapshots directory, which would pick the wrong one for a repo that
# has more than one.
_repo_meta() {
  local repo=$1 rev=${2-}
  [ -d "$HUB/$repo" ] || { echo "hub_list: missing $HUB/$repo" >&2; return 1; }
  [ -n "$rev" ] || rev=$(cat "$HUB/$repo/refs/main")
  [ -d "$HUB/$repo/snapshots/$rev" ] || {
    echo "hub_list: $repo has no snapshot $rev" >&2; return 1; }
  ls "$HUB/$repo/refs" 2>/dev/null | sed "s|^|$repo/refs/|" || true
  ( cd "$HUB/$repo" && find .no_exist -type f -printf "$repo/%p\n" 2>/dev/null ) || true
  ls "$HUB/$repo/snapshots/$rev" | grep -vE '\.(safetensors|png)$' \
    | sed "s|^|$repo/snapshots/$rev/|"
}

hub_list() {
  # Alpamayo: pinned revision, weights included -- this is the model being trained.
  local repo=models--nvidia--Alpamayo-1.5-10B
  _repo_meta "$repo" "$MODEL_REV"
  ls "$HUB/$repo/snapshots/$MODEL_REV" | grep -E '\.safetensors$' \
    | sed "s|^|$repo/snapshots/$MODEL_REV/|"

  # The two processor-chain repos: metadata only. Their safetensors are never read on
  # the training path; `cosmos` sends them anyway if a run ever proves otherwise.
  local cos=models--nvidia--Cosmos-Reason2-8B
  local qwen=models--Qwen--Qwen3-VL-8B-Instruct
  _repo_meta "$cos"
  _repo_meta "$qwen"
  if has cosmos; then
    local r
    for r in "$cos" "$qwen"; do
      local rv
      rv=$(cat "$HUB/$r/refs/main")
      ls "$HUB/$r/snapshots/$rv" | grep -E '\.safetensors$' \
        | sed "s|^|$r/snapshots/$rv/|"
    done
  fi
}

# --- relative to $PRE: only the clips a manifest actually names -------------------
sample_list() {  # sample_list <namespace> <parquet>
  "$PY" - "$PRE" "$1" "$2" <<'PY'
import sys
from pathlib import Path

import pandas as pd

pre, ns, parquet = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
df = pd.read_parquet(parquet)
missing = []
for r in df.itertuples():
    t0 = int(getattr(r, "t0_us", 5_100_000))
    rel = f"{ns}/samples/{r.clip_id}__t0_{t0}.npz"
    if (pre / rel).exists():
        print(rel)
    else:
        missing.append(rel)
if missing:
    print(f"MISSING {len(missing)} of {len(df)}: {missing[:3]}", file=sys.stderr)
PY
}

# --- relative to $OUT: the pruning recipes, without the 14 GB state files ---------
recipe_list() {  # recipe_list [config ...]  -- no args means every slim_* on disk
  local pat
  if [ $# -eq 0 ]; then
    pat='slim_*'
  else
    pat=""
  fi
  (
    cd "$OUT" || return
    if [ -n "$pat" ]; then
      find . -maxdepth 2 \( -path "./$pat/slim_meta.json" -o -path "./$pat/config.json" \
          -o -path "./$pat/summary.txt" \) -printf '%P\n' | sort
    else
      for c in "$@"; do
        for f in slim_meta.json config.json summary.txt; do
          [ -f "$c/$f" ] && echo "$c/$f"
        done
      done
    fi
  )
}
