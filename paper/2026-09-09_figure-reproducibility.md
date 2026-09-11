# Criterion figures: 재현 방법 및 논문용 설명

현재 `figures/fig1_*`, `figures/fig2_*`의 실제 입력 파일·측정 코드·manifest를
2026-09-09에 확인했다. 이 문서는 기존 측정 결과를 설명하며 새 GPU 측정은 수행하지 않았다.

## 1. 데이터와 실행 조건

| 항목 | 현재 그림의 조건 |
|---|---|
| 모델 | dense, unpruned `nvidia/Alpamayo-1.5-10B` |
| 모델 revision | `7aba8293c09993f2e125c6819df05d7fa3e873ea` |
| 데이터 | `nvidia/PhysicalAI-Autonomous-Vehicles` |
| 데이터 revision | `b719eea7f0a63619ef51ec7f54178af0937ef050` (eval-set 문서에 기록된 추출 기준) |
| calibration | `outputs/eval_sets/calib_100.parquet`, official train 100 clips |
| 기준 시각 | 모든 clip에서 `t0_us=5_100_000` |
| seed | global seed 42; clip seed는 아래 식 |
| Fig. 1 점수 | `outputs/importance_v2/importance.npz` |
| Fig. 1 GPU | NVIDIA RTX PRO 5000 Blackwell |
| Fig. 2 점수 | `outputs/stepimp_fm_perstep_v2/step_importance.npz` |
| Fig. 2 GPU | NVIDIA RTX 5880 Ada Generation |
| 계산 설정 | model.eval(), 모델 가중치 고정, BF16 모델/autocast, FP32 unit gates, SDPA attention |

```python
seed_c = int.from_bytes(
    hashlib.sha256(f"42:{clip_id}".encode()).digest()[:4], "big"
)
```

두 run의 config에 저장된 clip ID 100개와 순서는 calibration manifest와 일치한다.
두 run 모두 `metrics.json`에 완료 clip 수를 기록한다. GPU가 다르므로 두 run의
CoC 생성 텍스트가 clip별로 동일하다고 가정해서는 안 된다. 정확한 기존 그림의
재현에는 저장된 NPZ를 사용하고, 재측정 시에는 그림별 원래 GPU 아키텍처를 맞춘다.

## 2. Calibration 구성

Official train 153,625 clips에서 `ood_reasoning.parquet`에 등장하는 1,450 clips를
제외한 152,175 clips를 후보로 삼았다. 전체 official train의 metadata 분포에
선택 집합의 분포가 가까워지도록 weighted L1을 최소화하는 greedy matching으로
100개를 선정했다. 동률은 seed 42의 NumPy RNG로 결정한다.

매칭 속성과 가중치는 country 4.0, platform_class 2.0, time_of_day 2.0,
season 1.5, month 1.0, radar_config 1.0이다. 개별 clip은 중요도 계산에서 동일한
가중치 1/100을 가진다. 분포 매칭 표본이며 단순 무작위 표본이나 통계적으로
불편성이 보장된 표본이라고 표현하지 않는다.

현재 manifest에서 직접 확인한 구성:

- 24개 국가, daytime 61 / nighttime 39.
- winter 34 / fall 27 / summer 22 / spring 17.
- hyperion_8.1 71 / hyperion_8 29.
- `val_500`, `test_500`, `ood`와 clip ID 교집합은 각각 0.

입력은 front-wide 120°, front-tele 30°, cross-left 120°, cross-right 120°의
4개 카메라에서 각각 4프레임이다. 프레임 요청 시각은 t0−0.3, t0−0.2, t0−0.1,
t0 초이며 총 16장이다. Ego history는 10 Hz의 16개 pose(t0−1.5초부터 t0),
GT future는 t0+0.1초부터 t0+6.4초까지의 64개 pose이다.

측정은 `pre_processed/calib`의 기존 cache를 읽는다. Cache summary에 JPEG quality
95, 100개 성공 / 실패 0으로 기록되어 있다. 실제 프레임은 JPEG 재인코딩을 거쳤으므로
원본 비디오 직접 로드와 픽셀 단위로 같지 않다. 동일 결과 재현에는 동일 cache가 필요하다.
`sample_cache.PRE`의 기본 경로는 `/mnt/nvme1n1/ad_vla/data/physicalai_av/pre_processed`이다.

근거: [선정 코드](../experiments/evaluation/make_eval_sets.py),
[manifest 설명](../outputs/eval_sets/EVAL_SETS.md),
[cache 생성](../experiments/evaluation/build_cache.py),
[cache 로딩 및 seed](../experiments/evaluation/sample_cache.py).

## 3. CoC 및 trajectory target

**CoC target:** 각 clip에서 같은 dense 모델이 생성한 CoC token sequence를 사용한다.
사람이 작성한 CoC annotation이나 OOD의 `gt_coc`를 사용하지 않는다. 생성 조건은
sampling=True, temperature=0.6, top_p=0.98, top_k=None, 1 sequence,
max_new_tokens=256이다. Prompt는 배포 모델의 chat template, 16개 이미지 및 fused
trajectory-history tokens로 만든다.

생성 후 `[prompt]+[generated tokens]`를 teacher forcing으로 다시 입력한다.
CoC loss는 prompt 이후부터 `<|traj_future_start|>`까지(해당 token 포함)의
next-token cross-entropy 평균이다. 종료 token이 생성되지 않은 경우 구현은
최종 생성 token까지 사용한다. 따라서 엄밀하게는 자연어 CoC만의 NLL이 아니라
생성 구간의 특수 token을 포함하는 self-generated continuation NLL이다.

\[
\mathcal L_{\mathrm{CoC},c}=-\frac1{T_c}\sum_{j=1}^{T_c}
\log p_\theta(y_{c,j}\mid x_c,y_{c,<j}).
\]

**Trajectory target:** 데이터의 `ego_future_xyz`, `ego_future_rot`와 history pose를
`model.action_space.traj_to_action`에 넣어 얻는 normalized action tensor
\(a_c\in\mathbb R^{64\times2}\)를 사용한다. 설치된 Alpamayo 1.5의 action 변환은
가속도와 곡률을 모델의 mean/std로 정규화한다. 정확한 변환에는 고정한 모델
revision의 action-space 설정과 구현을 사용한다.

Trajectory loss는 최종 예측 XY의 ADE/MSE가 아니라 아래의 action-space
flow-matching velocity MSE이다. Conditioning cache는 위에서 생성한 CoC를
teacher forcing한 VLM forward에서 얻는다.

근거: [process_clip](../experiments/head_analysis/run_importance.py),
[run_rollout / gt_actions](../experiments/head_analysis/analysis_lib.py),
[coc_nll / expert_fm_grads](../experiments/head_analysis/prune_lib.py).

## 4. 측정한 denoising time

양쪽 run 모두 S=10이며 다음 midpoint grid를 사용한다.

\[
t_s=(s+0.5)/10,\quad s=0,\ldots,9,
\qquad t_s\in\{0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95\}.
\]

그림의 step 0–9는 이 시간점의 인덱스다. 실제 Euler rollout의 연속 상태를
측정한 것이 아니라, 각 시간점에서 GT action에 기반한 FM 입력을 별도로 구성했다.

\[
\epsilon_{c,s}\sim\mathcal N(0,I),\qquad
x_{c,s}=(1-t_s)\epsilon_{c,s}+t_s a_c,\qquad
v^*_{c,s}=a_c-\epsilon_{c,s},
\]
\[
\mathcal L_{c,s}=\frac1{64\cdot2}\sum_{j,d}
\left[v_\theta(x_{c,s},t_s;\mathrm{KV}_c)_{j,d}-v^*_{c,s,j,d}\right]^2.
\]

CPU `torch.Generator`를 clip seed로 초기화하고, step 순서대로 `(1,64,2)` Gaussian
noise를 새로 추출한다. Fig. 2 실제 config는 **noise_mode=per_step**이다.
현재 측정 CLI 기본값 `shared`와 다르므로 반드시 명시해야 한다. Step 간 순위
변화에는 시간점과 noise draw 변화가 함께 반영된다. Noise를 고정한 순수한 시간
효과 실험이라고 기술하면 안 된다.

## 5. 구조 점수 → 레이어 점수

각 구조에 scalar gate \(g_{\ell,u}=1\)을 둔다. Q head는 `o_proj` 입력에서
해당 head의 모든 head-dimension 성분에 같은 gate를 곱한다. MLP channel은
`down_proj` 입력의 해당 intermediate channel에 gate를 곱한다. Gate는 token
위치 전체에 공유된다. 따라서 gate gradient에서 token/head-dimension 기여가
먼저 부호를 유지한 채 합쳐지고, 그 뒤 절댓값을 취한다.

Fig. 1의 저장된 구조 점수는 다음과 같다(N=100).

\[
I^{\mathrm{CoC}}_{\ell,u}=\frac1N\sum_c
\left|\frac{\partial\mathcal L_{\mathrm{CoC},c}}{\partial g_{\ell,u}}\right|,
\quad
I^{\mathrm{traj}}_{\ell,u}=\frac1N\sum_c
\left|\sum_{s=0}^{9}\frac{\partial\mathcal L_{c,s}}{\partial g_{\ell,u}}\right|.
\]

Trajectory backward는 step loss의 **합**이며 1/10으로 나누지 않는다. 로그에 나오는
FM loss는 step 평균이지만 gradient 점수는 합이라는 차이에 유의한다. VLM에는
step별 cache gradient를 누적한 뒤 한 번 backward한다. 서로 다른 step의 gradient는
절댓값 전에 상쇄될 수 있다. Clip 사이에서는 절댓값 후 평균이므로 상쇄되지 않는다.

레이어 점수는 구조별 중요도의 단순 합이다.

\[
D^{o,a}_{\ell}=\sum_{u=1}^{U_a}I^{o,a}_{\ell,u}.
\]

VLM은 36 layers, layer당 Q heads 32개 / MLP channels 12,288개이며 각 축을 별도로
그린다. Q와 MLP를 합치거나 parameter 수로 나누지 않는다. 여기서 Q-head score는
Q projection 가중치 norm이 아니라 attention head output의 gate Taylor score이다.

Fig. 2의 저장 점수는 expert의 **step별 절댓값의 clip 평균**이다.

\[
A_{s,\ell,u}=\frac1N\sum_c
\left|\frac{\partial\mathcal L_{c,s}}{\partial g^{\mathrm{expert}}_{\ell,u}}\right|.
\]

Expert는 36 layers, Q heads 16개 / MLP channels 8,256개다. 사용 배열은
`q_abs_step`=(10,36,16), `mlp_abs_step`=(10,36,8256)이다.
Fig. 2의 step 합 \(\sum_s A_{s,\ell,u}\)는 Fig. 1 방식의
\(E_c[|\sum_s\partial\mathcal L_{c,s}/\partial g|]\)와 다르다.

근거: [UnitGates / FM backward](../experiments/head_analysis/prune_lib.py),
[reduce_draws / save](../experiments/head_analysis/run_step_importance.py).

## 6. 각 그림의 집계 및 정규화

| 파일 | 계산 |
|---|---|
| `fig1_depth_q_head`, `fig1_depth_mlp` | 위의 D를 각 objective·구조 축마다 독립적으로 \(D_\ell/\max_k D_k\) 정규화. 합=1 정규화, z-score, rank 변환 아님. 서로 다른 loss의 절대 크기가 아닌 깊이에 따른 형태를 비교. |
| `fig1_rank_agreement` | 각 layer에서 두 objective의 구조별 I 벡터 사이 Spearman rho. 레이어 점수로 합친 뒤 계산하지 않음. |
| `fig1_kept_overlap` | 각 objective의 top-k 집합 교집합 크기/k. Q k=19/32, MLP k=7390/12288. 독립 무작위 선택 기준은 k/U. |
| `fig2_steps_q_head`, `fig2_steps_mlp` | 각 layer에서 A의 step 쌍별 구조 순위 Spearman rho를 구한 뒤 36 layers에 대해 산술 평균. 두 heatmap의 표시 범위는 동일하게 0–1. Clip별 rho의 평균이 아님. |
| `fig2_mass_curve` | B=step 합 A를 layer 내부에서 오름차순 정렬. 누적합을 해당 layer의 B 총합으로 나눈 뒤 layer 평균. x=k/U, y=평균 제거 중요도 비율. |
| `fig2_stability_vs_mass` | 한 점=한 expert layer. x는 서로 다른 step의 45쌍 rho 평균, y는 해당 layer에서 하위 floor(0.9U)개 B의 합/B 총합. y축은 log. |

Mass curve의 구체적인 식:

\[
B_{\ell,u}=\sum_s A_{s,\ell,u},\qquad
C(k)=\frac1{36}\sum_{\ell=0}^{35}
\frac{\sum_{i=1}^{k}B_{\ell,(i)}}{\max(\sum_{u=1}^{U}B_{\ell,u},10^{-12})},
\]

여기서 \((i)\)는 해당 layer 내 오름차순이다. 먼저 전체 layer를 pooling하거나
전체 중요도 총합으로 나누지 않는다. 원점 (0,0)을 명시적으로 추가한다.
Nominal 90%에서 Q는 14/16=87.5%, MLP는 7430/8256≈89.995%를 실제 제거한다.

현재 코드에서는 objective 점수가 상수인 layer의 rho와 top-k overlap을 NaN으로
제외한다. VLM 마지막 layer(35)의 trajectory Q/MLP 점수가 모두 0인 경우가 이에
해당한다. 임의로 동률을 풀어 retained-set overlap을 만들어내지 않는다.

Sanity checks: depth peak는 Q trajectory/CoC=19/24, MLP=18/27;
step off-diagonal mean rho는 Q≈0.668, MLP≈0.874;
nominal 90% 제거 중요도는 Q≈68.5%, MLP≈1.8%이다.
최종 scatter의 layer 간 Spearman rho는 Q≈−0.451, MLP≈−0.436이다.

근거: [전체 plotting 계산](../experiments/paper/fig_criterion.py).

## 7. 재현 명령 및 보관할 자료

저장된 점수에서 현재 8개 panel을 다시 그리는 명령(CPU):

```bash
MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python experiments/paper/fig_criterion.py \
  --importance outputs/importance_v2/importance.npz \
  --step-importance outputs/stepimp_fm_perstep_v2/step_importance.npz \
  --out-dir figures
```

중요도 재측정 명령 예시(원본 결과를 덮어쓰지 않는 새 experiment ID):

```bash
# Blackwell GPU를 선택; 이 환경의 기존 매핑은 GPU 0–3.
.venv/bin/python experiments/head_analysis/run_importance.py \
  --exp-id importance_repro --num-clips 100 --calib-manifest calib_100 \
  --cache calib --seed 42 --max-gen 256 --fm-steps 10 --traj-mode fm --gpu 0

# Ada GPU를 선택; 이 환경의 기존 매핑은 GPU 4–7.
.venv/bin/python experiments/head_analysis/run_step_importance.py \
  --exp-id stepimp_repro --num-clips 100 --calib-manifest calib_100 \
  --cache calib --seed 42 --max-gen 256 --fm-steps 10 \
  --mode fm --noise-mode per_step --gpu 4
```

이 명령은 이 문서 작성 중 실행하지 않았다. 실행 전 장비의 GPU 아키텍처와
Hugging Face model/cache 접근 환경을 맞춰야 한다. 관련 환경 설정은
[recipes README](../recipes/README.md)에 있다. 현재 측정 코드에는 원래 run 이후
추가된 옵션도 있으므로, 저장된 config/NPZ를 기존 결과의 기준으로 삼는다.

재현 패키지에는 두 run의 NPZ·config.json·metrics.json, calibration manifest,
동일 JPEG cache 또는 그 재생성 조건, 측정/plotting 코드의 commit과 소프트웨어
버전을 함께 보관한다. 기존 config에는 라이브러리 전체 버전과 측정 당시 코드
commit이 없으므로, 현재 환경의 버전을 과거 측정 버전으로 단정하지 않는다.
NPZ 없이 그래프를 재측정하려면 모델·데이터 접근 권한 및 원본 cache도 필요하다.

확인한 입력 파일의 SHA-256:

```text
35e062073005d69710423d963c5a682c61b4b8801e2057bb04c4f363a42123ec  outputs/eval_sets/calib_100.parquet
1f21effca2754aa0ace8e0876c234f65558e56a01fe0dd1ebabe641077502eda  outputs/importance_v2/importance.npz
aac400c309228b51f0e8660ebefedd6b78dbba377d2f0e781fa3e3ea5ddc4f75  outputs/stepimp_fm_perstep_v2/step_importance.npz
```
