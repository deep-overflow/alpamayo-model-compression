# LLM-Pruner 기준선: 원본 빌드를 따르되 MoOP와 제거량을 맞춰 A100에서 평가

날짜: 2026-10-30. 상태: **승인됨(2026-10-30): (A) Q head 단위, 정확 일치 — 4–33번 층 Q head 15개, MLP는 4–27번 층 5,929·28–33번 층 5,928(앞쪽 층이 +1, l0keep 관례), 총 2,657,452,032 = MoOP. calib_100 점수 재사용, `lp_r50`도 A100에서 측정(권장안 적용).**

## 1. 목적

- **기존 기준선의 문제:** 논문과 기존 표의 LLM-Pruner 기준선 `lp_r50`은 모델의 **24.99%**를 지운다.
  MoOP(`dual_u40_v2`)는 23.99%다. 1.1억 개(1%p)를 더 지운 상태로 비교해 왔다.
- **이번 목표:** 원본 LLM-Pruner의 빌드 절차는 최대한 그대로 따르고, **제거 파라미터 수만 MoOP와 같게** 맞춘 arm을 만든다.
  그다음 A100 트랙(test_2000, OOD-Val262, LingoQA)에서 MoOP·MoOP l0keep·dense와 짝지어 평가한다.

## 2. 원본 LLM-Pruner의 빌드 절차 (공식 저장소에서 확인)

- **출처:** 공식 저장소 `/home/cvlab21/project/chan/LLM-Pruner`(commit `128a07d`).
  `hf_prune.py`, `llama3.py`, `LLMPruner/torch_pruning/` 전체가 soowon 트리의 사본과 바이트 단위로 같다(2026-10-30 확인).
- **논문 설정:** `scripts/llama_prune.sh`, README:121-136.

| 항목 | 원본 | 확인 위치 |
|---|---|---|
| 논문 설정(LLaMA-7B, 파라미터 약 20%) | `--pruning_ratio 0.25 --block_wise`, 층 4–30, `--pruner_type taylor --taylor param_first`, 이후 LoRA 복원(Alpaca, r=8, 2 epoch) | `scripts/llama_prune.sh` |
| 절단 단위(MHA, `hf_prune.py`) | attention은 head 단위(루트 `q_proj`, `consecutive_groups = head_dim`). MHA라 같은 번호의 k/v/o도 함께 잘림. MLP는 channel 단위(루트 `gate_proj`) | `hf_prune.py:95-112, 156-157` |
| **절단 단위(GQA, `llama3.py`)** | 루트를 **`k_proj`**에 두고 head_dim 단위로 묶는다. KV head 하나를 지우면 그 KV를 공유하는 Q head들·`v_proj`·`o_proj` 열이 함께 잘린다. 비율은 **KV head 수**에 적용된다 | `llama3.py:97-116, 163-164` |
| 중요도 | Taylor `param_first`(원소별 \|W·g\|), group 합 `sum` | `hf_prune.py:297-299` |
| 선택 | `--global_pruning` 꺼짐(층 안에서만 비교), `--iterative_steps 1` | `hf_prune.py:296-298` |
| calibration | bookcorpus `--num_examples`(기본 10)개 × 64토큰을 한 배치로 넣고 언어 모델 손실을 한 번 역전파 | `hf_prune.py:128, 146-148` |
| 층 범위 | 앞 4개·뒤 2개 층 제외(32층 예시 `4–30`; CLI 기본값은 3–31) | `scripts/llama_prune.sh` |
| 비율의 뜻 | 선택한 층의 group 중 지우는 비율(파라미터 비율이 아님). 채널 수는 `n − int(n·(1−r))`, head처럼 묶인 단위는 그 수를 head_dim으로 내림 | README:136, `metapruner.py:249-276` |
| 3단계(복원) | LoRA 학습(위 스크립트에서는 기본으로 수행) | `scripts/llama_prune.sh` |

## 3. 원본과 이번 arm의 대응

| 항목 | 원본 | 기존 `lp_r50`(soowon 포트) | 이번 arm | 다른 이유 |
|---|---|---|---|---|
| 대상 | LLaMA 전체 디코더 | VLM 텍스트 타워만(ViT·expert·embedding 제외) | 같음 | expert가 VLM의 KV 캐시를 읽음. MoOP도 같은 범위 |
| attention 그룹 기준 | GQA 모델은 `llama3.py`: KV 그룹 단위(k/v/Q 4개/o), 비율은 KV head 8개에 적용 | MHA용 `hf_prune.py` 방식: Q head 하나 단위(q 행 + o 열), `k/v_proj` 고정, 비율은 Q head 32개에 적용 | **Q0에 따라 결정** | Alpamayo는 GQA(Q 32 / KV 8)인데 KV head는 expert가 읽는 캐시라 지울 수 없음(강제 이탈) |
| 중요도 | `param_first`, 합 | 같음(`coc_param_first`) | 같음 | — |
| 손실 | 언어 모델 교차 엔트로피(한 배치) | 모델이 생성한 CoC 구간의 teacher-forcing 교차 엔트로피(클립별 기울기 합) | 같음 | 원본 손실의 직접 대응. `param_first` 순위는 기울기 배율과 무관해 배치/합 방식이 동등 |
| calibration 데이터 | bookcorpus 10문장 | `calib_100`(100클립) | 같음 | MoOP와 같은 데이터로 공정 비교 |
| 선택 | 층 안 비교, 1회 | 같음 | 같음 | — |
| 층 범위 | 앞 4·뒤 2 제외 | 4–33번(30개) | 같음 | 36층으로 옮긴 같은 규칙 |
| 비율 | 사용자 지정 | r = 0.50 → 24.99% | **r = 0.4825 → 23.988%** | MoOP와 제거량을 맞춤(이번 변경) |
| 복원 | 선택 | 하지 않음 | 하지 않음 | 모든 arm이 학습 없이 비교됨 |

**바뀌는 것은 비율 하나뿐**이다. 그래서 중요도 점수는 새로 계산하지 않고 기존 것을 그대로 쓴다.
soowon `outputs/lp_importance_v1/lp_importance.npz`의 `coc_param_first`이고, `.pre_dual.npz`와 값이 완전히 같음을 확인했다.

## 4. 제거량 맞추기 (원본 반올림 규칙으로 계산, 4–33번 층 30개)

| `--pruning_ratio` | 층당 head 제거 | 층당 MLP 제거 | 총 제거 | 전체 대비 | MoOP 대비 |
|---|---|---|---|---|---|
| 0.48 | 15/32 | 5,899/12,288 | 2,646,466,560 | 23.888% | −10,985,472 |
| 0.482 | 15/32 | 5,923 | 2,655,313,920 | 23.968% | −2,138,112 |
| **0.4825** | **15/32** | **5,929** | **2,657,525,760** | **23.988%** | **+73,728 (+0.003%)** |
| 0.483 | 15/32 | 5,936 | 2,660,106,240 | 24.011% | +2,654,208 |
| 0.50 (`lp_r50`) | 16/32 | 6,144 | 2,768,240,640 | 24.987% | +110,788,608 |

- **기준값:** MoOP(`slim_dual_u40_v2`)는 2,657,452,032개, MoOP l0keep은 2,657,447,936개를 지운다.
- **권장: r = 0.4825.** 원본 규칙(모든 층 같은 비율) 안에서 가장 가깝다. r이 0.482422–0.482503이면 같은 결과가 나온다.
- **완전 일치:** 30개 층 중 24개 층만 MLP를 하나 더 지우면 정확히 같아진다. 하지만 원본의 균일 규칙을 깨므로 권하지 않는다.
- **(B) KV 그룹 단위(`llama3.py` 방식, k/v는 남김)일 때:** head를 4개씩만 지울 수 있어 비율 하나로는 MoOP에 이만큼까지만 다가간다.
  - r = 0.499: 층당 Q head 12개(3그룹) + MLP 6,132개 → 2,637,987,840(23.81%), MoOP 대비 −19,464,192(−0.18%p)
  - r ≥ 0.5: 16개(4그룹) + 6,144개 → 24.99%(`lp_r50`과 같은 개수)
  - 그룹 단위 점수는 원본처럼 그룹의 4개 head 점수를 합한다. 원본은 k/v 행의 점수도 합하지만, 여기서는 지우지 않는 k/v를 점수에 넣을지가 또 하나의 선택이다(기본: 넣지 않음).
- **구성은 원본 방식 그대로 MoOP와 다르다.** LLM-Pruner는 30개 층에서 head 47%·MLP 48%를 지운다. MoOP는 36개 층에서 head 41%·MLP 40%를 지운다.
  0번 층을 자르지 않으므로 l0keep 문제는 해당되지 않는다.

## 5. 절차와 게이트

1. **중요도 복사:** soowon의 `lp_importance_v1/`(`lp_importance.npz` 32 MB + `config.json`)를 `outputs/cvpr27/lp_importance_v1/`로 복사한다.
   `lp_prune.py --importance cvpr27/lp_importance_v1`로 읽는다.
2. **R0 재현 게이트:** 이 저장소의 `lp_prune.py`로 r = 0.50을 다시 만든다.
   soowon `outputs/lp_r50/slim_meta.json`과 유닛 단위로 같아야 한다. 어긋나면 중단한다.
   (README가 계획했지만 한 번도 안 한 단계다.)
3. **빌드:** r = 0.4825로 만든다(`outputs/cvpr27/slim_lp_r4825`). 검증 항목은 다음과 같다.
   - 층당 head 15 / MLP 5,929, 총 제거 2,657,525,760(`lp_prune`의 assert)
   - 0–3·34–35번 층과 expert는 손대지 않음
   - 4–33번 층에서 남은 유닛이 `lp_r50`이 남긴 유닛의 상위집합(같은 점수에서 덜 지움)
   - 내장 smoke(롤아웃 1회 + NLL)
4. **A100 평가:** 기존 큐·홀더 잡으로 test_2000(4 shard), OOD-Val262(2 shard), LingoQA를 돌린다. 약 4.3 A100-시간이다.
5. **보고:**
   - 평균(1.6 / 3.2 / 6.4 s), CoC 퇴화율, LingoQA
   - MoOP·MoOP l0keep·dense 대비 짝지은 Δ와 95% CI
   - 기존 `lp_r50`과의 관계(Ada 기존 수치는 참고만)

## 6. 코드 변경 (최소)

| 파일 | 변경 |
|---|---|
| `experiments/llm_pruner/lp_prune.py` | `--no-state` 추가: `slim_meta.json`만 저장(`save_slim(write_state=False)`, `PORT:` 표시). 지금은 16.6 GB 가중치를 항상 쓴다 |
| `cvpr27/code/eval_a100.sbatch` | arm `lp_r4825`(필요하면 `lp_r50`) 등록 |
| `cvpr27/code/analyze_a100.py` | 위 arm 등록. MoOP와 MoOP l0keep 둘 다와 짝 비교하도록 `REFS`를 목록으로 확장 |

## 7. 결정이 필요한 것

- **Q0. attention 절단 단위 (공식 저장소 확인 후 추가):**
  - **(A) Q head 하나 단위(포트, `lp_r50`과 같음):**
    - 장점: MoOP와 예산을 0.003%까지 맞출 수 있고, 기존 `lp_r50`과 이어진다.
    - 단점: 공식의 GQA 경로(`llama3.py`)가 아니라 MHA 경로(`hf_prune.py`) 방식이다.
  - **(B) KV 그룹 단위(`llama3.py`에 더 가까움, k/v는 남김):**
    - 장점: Alpamayo 같은 GQA 모델에 대한 공식 처리를 최대한 따른다.
    - 단점: 비율 하나로는 MoOP보다 0.18%p 적게(23.81%) 지우는 것이 최선이다. 중요도를 다시 모아야 한다. 저장된 head별 점수를 그룹별로 더하면 되므로 GPU는 필요 없다.
  - **권장: (A)를 주 arm으로 한다.** 예산을 맞추는 것이 이번 목적이고, k/v를 지울 수 없어 (B)도 어차피 `llama3.py` 그대로는 아니기 때문이다. (B)는 원하면 보조 arm으로 둔다.

- **Q1. 비율:**
  - 권장: r = 0.4825. 원본 규칙을 그대로 지키고, MoOP와 +73,728개(0.003%) 차이다.
  - 대안: 24개 층만 하나 더 지워 정확히 일치시킨다.
- **Q2. 기존 `lp_r50`(24.99%)도 A100에서 함께 잴지:** 같은 카드에서 논문 수치의 계보와 예산 1%p 차이의 효과를 볼 수 있다.
  추가 비용은 약 4.3 A100-시간이다. 권장은 함께 재는 것이다.
- **Q3. calibration:**
  - 권장: MoOP와 같은 `calib_100`(기존 점수 재사용, GPU 불필요)
  - 대안: 원본처럼 10개만 쓰려면 중요도를 새로 계산해야 한다(GPU 1장, 약 1–2시간)
- **Q4. 빌드 GPU:** 빌드에는 비어 있는 Ada 카드가 필요하다(약 1분). 지금 4–7번은 다른 작업이 쓰고 있다.
  비는 대로 진행해도 될지 정해야 한다.

## 8. 하지 않는 것

- 2차 Taylor(`param_mix`), LoRA 복원, 폐루프
