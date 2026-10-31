#!/bin/bash
# Push the AlpaSim tree (code + small data) and one smoke scene to NEURON /scratch.
set -euo pipefail
A=/home/cvlab21/project/chan/alpasim
D=/scratch/e1997a06
SCENE=4f6bf6bb-9696-43b9-a1bf-04e12ccad5cc   # clipgt-01d503d4..., public_2601, first scene of s600 sh0
ssh -o BatchMode=yes neuron-dm "mkdir -p $D/project/alpasim $D/alpasim/nre-artifacts/all-usdzs $D/alpasim/sqsh $D/alpasim/runs"
rsync -a --info=stats1 \
  --exclude '.venv' --exclude 'outputs' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude 'data/nre-artifacts/all-usdzs' --exclude 'data/nre-artifacts/scenesets' \
  --exclude 'data/prometheus' --exclude 'data/drivers/*' --exclude 'tutorial*' \
  "$A/" "neuron-dm:$D/project/alpasim/" | tail -n 4
rsync -a --info=stats1 "/mnt/nvme1n1/ad_vla/data/nre-artifacts/all-usdzs/$SCENE.usdz" \
  "neuron-dm:$D/alpasim/nre-artifacts/all-usdzs/" | tail -n 2
ssh -o BatchMode=yes neuron-dm "du -s --apparent-size -BM $D/project/alpasim; find $D/project/alpasim | wc -l; \
  sha256sum $D/alpasim/nre-artifacts/all-usdzs/$SCENE.usdz" 2>&1 | grep -vE 'Deprecation|Redirecting'
sha256sum "/mnt/nvme1n1/ad_vla/data/nre-artifacts/all-usdzs/$SCENE.usdz"
