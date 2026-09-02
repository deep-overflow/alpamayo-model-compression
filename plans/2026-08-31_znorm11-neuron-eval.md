# znorm11 · dualfix — NEURON에서 4 arm 자체 완결 비교

날짜: 2026-08-31. 브랜치: `znorm11-criterion`. 상태: **준비 완료, 전송·제출 대기**(OTP 필요).

## 왜 NEURON인가, 그리고 그 대가

이 비교의 질문은 "`znorm11`(CoC + 10 FM 스텝, 각 레이어 내 z-정규화 후 1/11 평균)과 `dualfix`(층 35
퇴화 가드)가 dual과 12%/0.6% 다른 선택을 하는데, 그것이 성능 차이인가"이다. 답하려면 **dual과
클립·시드 페어드**여야 한다.

페어링은 **한 아키텍처 안에서만** 유효하다(같은 클립·시드가 Ada 0.286 / Blackwell 0.291, 클립의
3–4%는 CoC 텍스트 자체가 다르다). 출시된 모든 arm은 Ada에서 측정됐으므로 A100 결과와 섞을 수 없다.
따라서 NEURON에서 돌리려면 **baseline과 dual도 A100에서 다시 재야 한다** — 그것이 이 선택의 비용이고,
사용자가 그 비용을 받아들여 4 arm 전부를 NEURON에서 돌리기로 했다(2026-08-31).

결과 표는 **Ada 표와 합치지 않고 별도 블록**으로 보고한다. 세 번째 아키텍처가 생기므로 각 표에
측정 카드를 명시한다(기존 각주: expert-axis의 q50/m50은 Blackwell).

## arm과 예산

| arm | 정의 | 빌드 |
|---|---|---|
| `baseline_neuron` | 무압축 | 없음 |
| `dual_neuron` | `dual_u40_v2` (출시 기준 재현) | `--config dual_u40_v2` |
| `znorm11_neuron` | 11-손실 z-평균 | `--config znorm11_u40_v2 --stepvlm importance_stepvlm_v1` |
| `dualfix_neuron` | dual + 상수-레이어 가드 | `--config dualfix_u40_v2` |

전부 u40 균일(레이어당 Q 19/32, MLP 7390/12288), VLM 전용, expert·KV 무접촉, −2,657,452,032.
셋 다 **선택-only**라 `--no-state`로 빌드한다(레시피 ~3 MB; `load_slim`이 평가 시 base 가중치에서
복원). 이 플래그를 tyr/dualr/dualrc 계열에 쓰면 안 된다 — make_slim이 거부한다.

**중요**: `--importance importance_v2_ada`. per-step 파일(`importance_stepvlm_v1`)이 Ada에서
측정됐으므로 CoC 반도 Ada 본이어야 기준이 one-factor다. *평가* 카드가 A100인 것과는 다른 문제다.

## 전송 (증분 ~4.1 GB)

기존 `push_neuron.sh`의 code/data/weights/recipes 티어는 2026-08-23에 이미 보냈다(37 GB: 모델
가중치 rev 7aba8293, train/ood 캐시, eval 238클립, 매니페스트). 이번에 추가로 필요한 것만 새
티어 `evalsets`로 묶었다:

- `indist_500` 500클립(`eval` 네임스페이스) + `test_500` 500클립(**`test` 네임스페이스** — 두 세트가
  서로 다른 캐시 트리에 있다) = **4.0 GB**. OOD-val 262클립은 `data` 티어의 ood 네임스페이스에
  이미 들어가 있다.
- `importance_v2_ada`(12 MB) + `importance_stepvlm_v1`(41 MB) + `importance_v2` + `slim_integrated_mag`
  (u40 예산을 유도하는 참조 메타).

```bash
ssh -fNM -S ~/.ssh/cm-neuron -o ControlPersist=8h e1997a06@neuron-dm.ksc.re.kr   # OTP 1회
LIST_ONLY=1 bash experiments/transfer/push_neuron.sh e1997a06 evalsets           # 미리보기
bash experiments/transfer/push_neuron.sh e1997a06 code evalsets                  # 코드 갱신 + 데이터
```

## 실행

1. **빌드 — 디버깅 노드**(`experiments/transfer/build_criteria.sbatch`): 3개 config, 각각 모델 로드
   + 수술 ≈ 8–12분. 기본은 `amd_a100nv_8`(전 파티션 상한 2-00:00:00)이지만, 디버그 큐의 `MaxTime`이
   30분이면 config당 한 job으로 쪼개 제출한다:
   ```bash
   sinfo -o "%P %l %D %G"                 # 파티션·시간 상한·노드·GPU
   scontrol show partition <debug>        # MaxTime / MaxNodes 확인
   sbatch --partition=<debug> --time=00:30:00 experiments/transfer/build_criteria.sbatch dual_u40_v2
   ```
   **아직 확인 못 한 것**: NEURON의 디버그 파티션 이름과 시간 상한. Datamover에서는 `sinfo`가 없으므로
   `ssh glogin01`에서 위 두 명령으로 확인한 뒤 `--partition`을 채운다.
2. **평가**(`eval_arms.sbatch`): 4 arm × 3 세트 × 2 shard = 24 job을 4카드에 4개씩 겹쳐 돌린다.
   arm당 1,262클립 ≈ 3.2 GPU-h → 총 ≈ 13 GPU-h → **4카드에서 ≈ 3.5 h**. 계정 상한이 A100 4장이라
   한 job이 자원을 다 쓴다.
3. **회수·분석**: `outputs/*_neuron_*` 행을 이 박스로 rsync 후
   `analyze_cacheproxy.py --val-arms ... --sets indist test oodval`(baseline은 `baseline_neuron`)로
   페어드 비교. Ada 기존 표와 섞지 않는다.

## 판정 (사전 등록)

- **G0 재현**: `dual_neuron`의 kept set이 `slim_dual_u40_v2/slim_meta.json`과 bit-identical
  (빌드는 결정론적이므로 아키텍처와 무관해야 한다). 어긋나면 전송/버전 문제이므로 중단.
- **G1 주 판정**: `znorm11 − dual` paired ΔminADE@6(중앙값 [95% CI], val500). CI가 0을 포함하면
  "선택 12% 차이는 성능에서 구별되지 않음" — 이 경우 스텝 축 집계·CoC 가중은 u40 예산에서 자유도가
  아니라는 뜻이다.
- **G2 부수**: `dualfix − dual`. 0.6%만 다르므로 차이가 없을 것으로 예상하고, 있다면 층 35 하나가
  그만큼 민감하다는 뜻이라 그 자체로 보고 대상이다.
- **G3 일관성**: test500·OOD-val에서 부호 유지, CoC 붕괴율이 dual 수준.

## 코드 (이번 커밋)

| 파일 | 역할 |
|---|---|
| `experiments/transfer/push_neuron.sh` | 새 티어 `evalsets`(val_500 `eval` + test_500 `test` 네임스페이스, importance 파일). 기존 티어 불변 |
| `experiments/transfer/build_criteria.sbatch` | 디버그/기본 파티션에서 3개 recipe 빌드, `--no-state` |
| `experiments/transfer/eval_arms.sbatch` | 4 arm × 3 세트, 카드당 1 job, 아키텍처 혼합 금지 주석 |

## 남은 미지수

디버그 파티션의 실제 이름과 시간 상한(위 1번). OTP 세션이 열리면 `sinfo` 한 번으로 확정하고 이
계획의 해당 줄을 갱신한다.
