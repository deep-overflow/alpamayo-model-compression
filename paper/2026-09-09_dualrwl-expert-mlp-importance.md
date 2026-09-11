# dualrwl + expert MLP pruning: 최종 중요도 계산

## 결론

Expert MLP 채널은 **step-normalized flow-matching Taylor importance**로 선택했다.
계산 순서는 `step별 gate gradient → 절댓값 → calibration clip 평균 →
각 step·layer 내 채널 간 z-score → step 평균 → layer별 하위 채널 제거`이다.
코드상의 집계명은 `znorm`이며, expert 기준에 CoC 점수나 dual max-rank 연산을 섞지 않는다.

## 측정 조건

- 원본 모델: dense `nvidia/Alpamayo-1.5-10B`, revision
  `7aba8293c09993f2e125c6819df05d7fa3e873ea`.
- Calibration: `calib_100`, official train의 동일한 100 clips, seed 42.
- 측정 GPU: NVIDIA RTX 5880 Ada Generation.
- Expert: 36 layers, layer당 intermediate channels 8,256개.
- 각 `mlp.down_proj` 입력 channel에 scalar gate g=1을 적용한다. Gate는 token 위치에 공유된다.
- CoC는 dense 모델이 생성한 sequence이며, teacher forcing한 VLM cache에 expert를 조건화한다.
- Target action은 실제 미래 pose를 `traj_to_action`으로 변환한 64×2 normalized
  acceleration/curvature이다.
- S=10, t_s=(s+0.5)/10: 0.05, 0.15, …, 0.95.
- Noise는 `per_step`: clip seed로 초기화한 CPU torch.Generator에서 각 step마다
  새로운 Gaussian noise를 추출한다.
- 중요도는 **dualrwl의 VLM pruning/refit 이전 dense 모델에서 측정**했다.
  Pruned/refitted VLM의 cache로 재보정한 점수가 아니다.

Calibration, target 및 seed의 상세는
[공통 재현 설명](2026-09-09_figure-reproducibility.md)을 따른다.

## 정확한 수식

Clip c, step s, layer l, channel u에 대해 FM loss를 다음과 같이 둔다.

\[
x_{c,s}=(1-t_s)\epsilon_{c,s}+t_s a_c,\qquad
v^*_{c,s}=a_c-\epsilon_{c,s},
\]
\[
L_{c,s}=\operatorname{MSE}\bigl(v_\theta(x_{c,s},t_s;\mathrm{KV}_c),v^*_{c,s}\bigr).
\]

MSE는 64×2 원소에 대한 평균이다. Gate 값이 1이므로 first-order gate Taylor
점수 |g dL/dg|는 |dL/dg|와 같다.

먼저 각 step·layer·channel의 절대 gradient를 N=100 clips에 대해 평균한다.

\[
A_{s,l,u}=\frac1N\sum_{c=1}^N
\left|\left.\frac{\partial L_{c,s}}{\partial g_{l,u}}\right|_{g=1}\right|.
\]

그 다음 각 step과 layer에서 U=8256 channels에 대한 평균과 표준편차를 계산한다.

\[
\mu_{s,l}=\frac1U\sum_u A_{s,l,u},\qquad
\sigma_{s,l}=\sqrt{\frac1U\sum_u(A_{s,l,u}-\mu_{s,l})^2}.
\]

표준편차는 NumPy 기본값인 population standard deviation (`ddof=0`)이다.

\[
Z_{s,l,u}=\frac{A_{s,l,u}-\mu_{s,l}}{d_{s,l}},\qquad
 d_{s,l}=\begin{cases}\sigma_{s,l},&\sigma_{s,l}>0\\1,&\sigma_{s,l}=0.\end{cases}
\]

마지막으로 step 축을 동일 가중치로 평균한다.

\[
\boxed{I^{\mathrm{expert\text{-}MLP}}_{l,u}=\frac1{10}\sum_{s=0}^{9}Z_{s,l,u}.}
\]

`znorm`은 **clip별 z-score 후 평균**이 아니다. Clip 평균을 먼저 만든 뒤 z-score한다.
원본 step별 loss scale의 영향을 줄여 step별 상대적 채널 중요도를 같은 표준화
척도에서 결합한다. Final score는 음수가 될 수 있으며, 이후 절댓값이나 clipping을
적용하지 않는다. 음수는 해당 표준화 기준에서 상대적으로 낮다는 뜻이지 채널을
제거하면 loss가 감소한다는 의미가 아니다.

## 채널 선택 및 pruning 범위

각 layer에서 독립적으로 최종 점수 I를 오름차순 정렬하고
`k = int(round(8256 * pruning_ratio))`개의 최저 점수 채널을 제거한다.
실제 구현은 `np.argsort(scores[layer])[:k]`이며 별도의 동률 처리 규칙은 없다.
전체 layer를 합친 global ranking이나 layer 중요도 기반 예산 배분은 사용하지 않는다.

| Expert MLP 제거율 | 제거 channels/layer | 유지 channels/layer |
|---:|---:|---:|
| 50% | 4,128 | 4,128 |
| 75% | 6,192 | 2,064 |
| 87.5% | 7,224 | 1,032 |
| 93.75% | 7,740 | 516 |
| 96.875% | 7,998 | 258 |
| 98.4375% | 8,127 | 129 |
| 100% | 8,256 | 0 |

모든 layer에 같은 비율을 적용한다. Expert Q heads(16/layer), KV groups 및
head dimension은 유지한다. VLM의 dualrwl 선택·refit 결과와 expert의 채널 선택은
별도로 적용한다. Expert MLP의 선택을 위해 VLM의 CoC/Taylor 점수를 결합하거나
OSSCAR refit 점수를 사용하는 과정은 없다.

## 파일 및 구현 근거

1. `outputs/stepimp_fm_perstep_v2/step_importance.npz`의
   `mlp_abs_step`, shape=(10,36,8256): A 배열.
2. [make_stepexp_importance.py](../experiments/head_analysis/make_stepexp_importance.py)의
   `aggregate('znorm', ...)`: step별 z-score 후 평균.
3. [run_expert_agg.py](../experiments/head_analysis/run_expert_agg.py)의
   `zscore_layers`: layer 내부 mean/std 및 zero-std 처리.
4. `outputs/importance_stepexp_znorm/importance.npz`의
   `traj_exp_mlp`, shape=(36,8256): 최종 I 배열.
5. [make_slim.py](../experiments/head_analysis/make_slim.py)의
   `dualrc_u40_s<N>_em<M>` 분기: `--expert-importance importance_stepexp_znorm`에서
   `traj_exp_mlp`를 읽어 expert MLP mask 생성.
6. [mask_lib.py](../experiments/head_analysis/mask_lib.py)의 `select_mask`:
   layer별 최저 점수 round(Ur)개 제거.

주의: checkpoint의 `config.json`에 있는 `importance_from: importance_v2`는
expert 점수의 충분한 출처 기록이 아니다. 이 분기는 VLM용 `importance_v2`와
expert용 `importance_stepexp_znorm` 두 파일을 읽는다. 후자의 config에서
`step_run=stepimp_fm_perstep_v2`, `aggregation=znorm`이 확인된다.
`reference=importance_v2_ada`는 나머지 배열을 복사하는 출처이며 최종 expert MLP
배열에 두 점수를 혼합한다는 뜻이 아니다.

현재 figure의 raw importance mass curve는 step별 A를 단순 합한 B=sum_s A로
그린 것이다. 실제 expert pruning 점수 I=mean_s z(A_s)와 다르므로 raw mass
곡선을 실제 최종 pruning score의 누적 질량이라고 설명하면 안 된다.

## CPU 재현 코드

```python
import numpy as np

A = np.load(
    'outputs/stepimp_fm_perstep_v2/step_importance.npz'
)['mlp_abs_step'].astype(np.float64)

standardized = []
for a in A:  # a: (36, 8256)
    mean = a.mean(axis=1, keepdims=True)
    std = a.std(axis=1, keepdims=True)  # ddof=0
    standardized.append((a - mean) / np.where(std > 0, std, 1.0))
score = np.mean(standardized, axis=0)  # (36, 8256)

saved = np.load(
    'outputs/importance_stepexp_znorm/importance.npz'
)['traj_exp_mlp']
assert np.array_equal(score, saved)

ratio = 0.9375
keep = np.ones_like(score, dtype=bool)
k = int(round(score.shape[1] * ratio))
for layer in range(score.shape[0]):
    keep[layer, np.argsort(score[layer])[:k]] = False
assert np.all(keep.sum(axis=1) == 516)
```

2026-09-09 검증: 재계산 score와 저장 배열이 정확히 일치(max absolute difference=0).
50/75/87.5/93.75/96.875/98.4375/100%의 7개 checkpoint 모두 36 layers의 expert
MLP kept indices가 재계산과 일치하고 Q heads 16개가 전부 유지됨을 확인했다.
새 GPU calibration이나 모델 평가는 수행하지 않았다.

## 논문용 영문 문단

We prune expert MLP channels using step-normalized first-order Taylor scores
measured on the dense model over 100 calibration clips. A scalar gate initialized
to one is placed on each intermediate channel immediately before the MLP down
projection. At each of ten flow-matching times, t_s=(s+0.5)/10, we compute the
absolute gradient of the ground-truth-action flow-matching loss with respect to
each gate and average it over calibration clips. We then standardize these scores
across channels separately for each layer and time step using the population
standard deviation, and average the standardized scores across the ten steps.
The lowest-scoring channels are removed independently within each layer at a
uniform pruning ratio. Expert attention heads and KV groups are retained. These
scores are measured with dense-model conditioning and reused when composing
expert MLP pruning with the pruned and refitted dualrwl VLM, without conditional
recalibration. Independent Gaussian noise is sampled at each measurement time.
