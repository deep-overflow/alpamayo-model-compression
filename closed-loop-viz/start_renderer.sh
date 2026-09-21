#!/bin/bash
# Bring up ONLY the NuRec renderer, on one GPU, as a standalone gRPC server.
#
# Copied from the renderer-0 service of a shipped run's docker-compose.yaml, with two
# changes: the card is ours to pick, and port 6007 is published to the host because there
# is no microservices_network to join -- nothing else is running.
#
# No driver, no physics, no trafficsim: a replay feeds this server the RGBRenderRequest
# messages already recorded in rollout.asl, so the trajectory and the actor poses are
# fixed and only the pixels are new.
#
# The container name is prefixed so a sweep can never touch another member's stack.
set -uo pipefail

GPU=${GPU:-1}
PORT=${PORT:-16007}
NAME=${NAME:-chan_nre_replay}
SCENESET=${SCENESET:-67e4e4dced92f27ad03a567be5b96c40}

docker rm -f "$NAME" >/dev/null 2>&1

docker run -d --name "$NAME" \
  --gpus "\"device=$GPU\"" \
  -p "$PORT:6007" \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e OMP_NUM_THREADS=1 \
  -v /mnt/nvme1n1/ad_vla/data/nre-artifacts:/mnt/nre-data \
  -v /home/cvlab21/project/chan/alpasim/data/nre-artifacts/ego-hoods:/mnt/ego-hoods \
  --entrypoint bash \
  nvcr.io/nvidia/nre/nre-ga:26.04 \
  -c "umask 0000
/app/internal/scripts/pycena/runtime/pycena_nrm_full serve-grpc \
  --port=6007 --host=0.0.0.0 \
  --artifact-glob='/mnt/nre-data/scenesets/$SCENESET/**/*.usdz' \
  --egocar-hood-dir=/mnt/ego-hoods \
  --no-enable-nrend \
  --download-cache-dir /tmp/nre-cache-dir --cache-size=3 --max-workers=4 \
  --enable-editing-actors"

echo "started $NAME on GPU $GPU, host port $PORT"
