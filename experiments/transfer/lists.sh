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
# reporting the file absent. That is what killed NEURON jobs 890894/890895 -- the loader
# probes for adapter_config.json and model.safetensors, neither of which this repo has.
# 18 empty files, 28 KB.
hub_list() {
  local repo=models--nvidia--Alpamayo-1.5-10B
  ls "$HUB/$repo/refs" 2>/dev/null | sed "s|^|$repo/refs/|" || true
  ( cd "$HUB/$repo" && find .no_exist -type f -printf "$repo/%p\n" 2>/dev/null ) || true
  ls "$HUB/$repo/snapshots/$MODEL_REV" | sed "s|^|$repo/snapshots/$MODEL_REV/|"
  # Cosmos: config/tokenizer/processor only. The recovery path resolves this repo
  # through the local_files_only patch in expert_per_clip, but only reads these --
  # the four safetensors have not been touched since 2026-08-11. If a target run does
  # ask for them it fails at load, and the `cosmos` tier (or `hf download`) fixes it.
  local cos=models--nvidia--Cosmos-Reason2-8B
  local rev
  rev=$(ls "$HUB/$cos/snapshots" | head -1)
  ls "$HUB/$cos/refs" 2>/dev/null | sed "s|^|$cos/refs/|" || true
  ( cd "$HUB/$cos" && find .no_exist -type f -printf "$cos/%p\n" 2>/dev/null ) || true
  ls "$HUB/$cos/snapshots/$rev" | grep -vE '\.(safetensors|png)$' \
    | sed "s|^|$cos/snapshots/$rev/|"
  if has cosmos; then
    ls "$HUB/$cos/snapshots/$rev" | grep -E '\.safetensors$' \
      | sed "s|^|$cos/snapshots/$rev/|"
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
