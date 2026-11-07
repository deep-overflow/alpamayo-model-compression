"""Re-score hard100 rollouts that AlpaSim aborted mid-run (route fold-back check) from their ASL.

Scenes 3b934c9a / ceb3d0d5 abort when the route is extended near its end (alpasim_runtime
route_generator.py), so the ASL up to the abort survives. Every attempt of every arm on those scenes
(aborted + completed, retries included) is copied to a scratch job dir, given a `_complete` tracker,
run through alpasim `eval.main`, given the all-zero `img_is_black` rows the original runs carry
(images were skipped, so ImageScorer cannot run), and aggregated with `eval.aggregation.main`
(4 m truncation included). Writes <out>/reeval_aggregates/<scene>_<cfg>.json and each attempt's
per-timestep <out>/parquets/<scene>_<cfg>/<rollout_id>/metrics.parquet.

Usage (repo root):
    set -a; . /home/cvlab21/project/chan/alpasim-runs/.hf_env; set +a   # only if a usdz must be downloaded
    .venv/bin/python cvpr27/code/h100_reeval_aborted.py [--cfgs lp_r50 ...] [--scenes ceb3d0d5 ...]
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RUNS = '/mnt/nvme1n1/ad_vla/results/chan/alpasim-runs'
ALPASIM = '/home/cvlab21/project/chan/alpasim'
USDZ_DIR = '/mnt/nvme1n1/ad_vla/data/nre-artifacts/all-usdzs'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFGS = ['baseline', 'slim_dual_u40_v2', 'slim_dualexp_u40_em93p75', 'slim_coc_u40_v2', 'slim_traj_u40_v2',
        'slim_tyr_u40_r', 'slim_tyrK', 'lp_r50', 'slim_wanda_u40_v2']
SCENES = ['3b934c9a', 'ceb3d0d5']


def usdz_for(scene):
    for row in csv.DictReader(Path(f'{ALPASIM}/data/scenes/sim_scenes.csv').read_text().splitlines()):
        if row['scene_id'][7:15] == scene:
            local = f"{USDZ_DIR}/{row['uuid']}.usdz"
            if os.path.exists(local):
                return row['uuid'], local
            from huggingface_hub import hf_hub_download
            p = hf_hub_download('nvidia/PhysicalAI-Autonomous-Vehicles-NuRec', repo_type='dataset',
                                revision=row['hf_revision'], filename=row['path'], token=os.environ.get('HF_TOKEN'))
            return row['uuid'], p
    raise KeyError(scene)


def add_img_is_black(path):
    t = pq.read_table(path)
    if 'img_is_black' in set(t.column('name').to_pylist()):
        return
    o = t.filter(pc.equal(t['name'], 'offroad'))
    k = o.num_rows
    for col, val in (('name', 'img_is_black'), ('values', 0.0), ('time_aggregation', 'max')):
        o = o.set_column(o.schema.get_field_index(col), col, pa.array([val] * k, type=t.schema.field(col).type))
    pq.write_table(pa.concat_tables([t, o]), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfgs', nargs='+', default=CFGS)
    ap.add_argument('--scenes', nargs='+', default=SCENES)
    ap.add_argument('--out', default=f'{REPO}/outputs/cvpr27/closedloop_ci_2026-11-07/aborted_reeval')
    ap.add_argument('--procs', type=int, default=6)
    args = ap.parse_args()
    os.makedirs(f'{args.out}/reeval_aggregates', exist_ok=True)
    py = f'{ALPASIM}/.venv/bin/python'
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(f'{tmp}/usdz')
        for s in args.scenes:
            uid, p = usdz_for(s)
            os.symlink(p, f'{tmp}/usdz/{uid}.usdz')
        for s in args.scenes:
            for cfg in args.cfgs:
                d = f'{tmp}/{s}/{cfg}'
                os.makedirs(f'{d}/rollouts')
                shard = min(glob.glob(f'{RUNS}/h100_{cfg}_sh*/eval-config.yaml'))
                for f in ('eval-config.yaml', 'run_metadata.yaml', 'wizard-config.yaml'):
                    shutil.copy(os.path.join(os.path.dirname(shard), f), d)
                cfg_path = Path(f'{d}/eval-config.yaml')
                cfg_txt = [f'num_processes: {args.procs}' if x.startswith('num_processes:') else x
                           for x in cfg_path.read_text().splitlines()]
                cfg_path.write_text('\n'.join(cfg_txt) + '\n')
                attempts = glob.glob(f'{RUNS}/h100_merged_{cfg}/rollouts/clipgt-{s}*/*/rollout.asl')
                for a in attempts:
                    rid_dir = f"{d}/rollouts/{a.split('/')[-3]}/{a.split('/')[-2]}"
                    os.makedirs(rid_dir)
                    shutil.copy(a, f'{rid_dir}/rollout.asl')  # copy, never link: eval writes next to the ASL
                    Path(f'{rid_dir}/_complete').touch()
                subprocess.run([py, '-m', 'eval.main', '--asl_search_glob', f'{d}/rollouts/**/*.asl',
                                '--config_path', f'{d}/eval-config.yaml', '--usdz_glob', f'{tmp}/usdz/*.usdz'],
                               cwd=ALPASIM, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for p in glob.glob(f'{d}/rollouts/*/*/metrics.parquet'):
                    add_img_is_black(p)
                    keep = f"{args.out}/parquets/{s}_{cfg}/{p.split('/')[-2]}"  # per-timestep input of closedloop_gap_100.py
                    os.makedirs(keep, exist_ok=True)
                    shutil.copy(p, f'{keep}/metrics.parquet')
                subprocess.run([py, '-m', 'eval.aggregation.main', '--array_job_dir', d,
                                '--config_path', f'{d}/eval-config.yaml'],
                               cwd=ALPASIM, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                shutil.copy(f'{d}/aggregate/results-summary.json', f'{args.out}/reeval_aggregates/{s}_{cfg}.json')
                print(f'{s} {cfg}: {len(attempts)} attempts re-scored', flush=True)


if __name__ == '__main__':
    main()
