# 최신 프레임 블랙아웃 — 폐루프 (2026-10-19, 확정)

## 목적
개루프(2026-10-18)에서 최신(t0) 프레임 1장을 검게 바꾸면 minADE@6이 2–4% 나빠졌고, 카메라별로는 전방 망원만
일관된 비용을 냈다. 폐루프에서는 같은 개입이 **매 정책 스텝(0.1초)마다** 반복된다. 질문은
**매 스텝 무작위로 프레임 1장이 검게 들어올 때 폐루프 주행 성능이 얼마나 떨어지는가**이다.

가설
- H1: 블랙아웃 조건의 씬 점수가 baseline보다 낮다.
- 반대 가능성: 매 스텝 다른 카메라가 가려지고 다음 스텝에는 같은 프레임이 정상으로 다시 보이므로 섭동이 평균되어
  사라진다. 개루프 효과(폐루프 실효 지평 ~1.9초 근처에서 +2–4%)가 작아서, 점수로는 해상도 아래일 가능성이 크다.

## 설계 (2026-10-19 사용자 결정: Q1 최신 4장 중 1장, Q2 남지 않음, Q3 Origin150만, Q4 기존 baseline 재사용)
**개입 (alpasim 드라이버, `src/driver/src/alpasim_driver/main.py::_prepare_camera_images`)**
```python
# 버퍼에서 꺼낸 카메라별 4장(oldest -> newest) 중, 무작위 카메라 1대의 최신 프레임만 검은 사본으로 교체
rng = random.Random(f"blackout:{session.seed}:{session.inference_count}")
cam = rng.choice(self._model.camera_ids)
frames[cam][-1] = (ts, np.zeros_like(img))      # 버퍼(FrameCache)의 원본은 그대로
session.last_blackout_cam = cam                 # drive() 응답의 debug_info에 기록
```
- 설정: `driver.inference.blackout_t0_random_camera: bool = False` (`schema.py`). 기본값이 꺼져 있으므로 다른 런에는
  영향이 없다. `src/`가 컨테이너에 바인드 마운트되어 있어 이미지 재빌드가 필요 없다.
- 카메라 선택은 ASL에 기록되는 세션 시드(`driver_session_request.random_seed`)와 추론 횟수에서 정해지므로
  사후에 재계산할 수 있고, 매 스텝 `debug_info`에도 `blackout_cam`을 기록한다.
- 버퍼 원본을 건드리지 않으므로 다음 스텝에서는 같은 프레임이 t−0.1 자리에 **정상 이미지로** 들어간다
  (개루프와 같은 개입, Q2).

**런처** — `launch_alpasim_shards.sh`에 `EXTRA_OVERRIDES` 환경변수를 추가해
`driver.inference.blackout_t0_random_camera=true`를 넘긴다. config는 `baseline`(무압축), 로그는 `PREFIX`로 구분:
- Origin150: `PREFIX=m2601_bkrand_`, 150씬 × 2 rollout
- 공통: Ada 4–7, 카드당 shard 1개(드라이버·렌더러·물리 같은 카드), `DRIVER_OMP_THREADS=8`.
- 병합: `merge_alpasim_shards.py` → `alpasim-runs/m2601_merged_bkrand_baseline`.

**기준** — 기존 `m2601_merged_baseline`(2026-08-10, 150×2) 재사용.
확인한 것: 시뮬레이터 코드는 그 뒤 바뀐 파일이 없고(최신 수정 2026-07-25), 같은 shard 경로·Ada 4장·8스레드였다
(rollout 완료 간격으로 씬당 12.6분). alpasim에는 시드 제어가 없으므로 비교는 씬 단위 페어드다.

## 스모크 (본 실행 전)
1씬 × 1 rollout으로:
- ASL의 매 `driver_return` debug_info에 `blackout_cam`이 있고, 네 카메라가 고르게 나온다.
- 세션 시드와 추론 횟수로 재계산한 카메라가 기록과 일치한다.
- 빈 궤적 0개(카메라당 프레임 4장 유지), 블랙아웃이 꺼진 런과 설정 외 차이가 없다.

## 지표
- 주: 씬 점수(rollout 평균 → 씬 평균), baseline 대비 씬 페어드 Δ, 부트스트랩 평균 CI(주) + Wilcoxon(부).
- 게이트: offroad, at-fault collision 비율.
- 연속 대리지표(점수보다 민감): `analyze_lateral.py`(차선 이탈 시간, 최장 이탈), `analyze_longitudinal.py`
  (근접 시 제동, 차간 시간), 둘 다 `--reference baseline`.
- CoC 퇴화율(`analyze_alpasim.py`).
- 블랙아웃 카메라별 분해(탐색): 전방 망원이 가려진 스텝 직후의 계획 변화.

## 사전 등록 게이트
- **G0 (스모크)**: 위 스모크 항목 전부 통과. 실패하면 본 실행 없음.
- **G1 (효과 존재)**: 씬 점수 Δ의 부트스트랩 CI가 0을 배제하는지 보고.
  검출 한계: 씬 페어드 Δ의 σ≈0.32라 Origin150은 ~0.080이다. 0을 포함하면
  "효과 없음"이 아니라 "해상도 아래"로 기록하고 연속 대리지표로 판단한다.

## 해석상 주의
- **인계 제동 인공물**(2026-10-08 메모): 강제 GT가 끝날 때 차가 −8 m/s²로 넘어오고, 무압축 VLM은 그 감속을 이어
  간다. 메모의 규칙은 "인공물 수정 전 새 폐루프 arm 금지"인데, 이번 비교는 **같은 모델**의 블랙아웃 유무이므로
  인공물은 두 조건에 똑같이 들어간다. 다만 블랙아웃이 그 감속 반응을 바꿀 수 있어 상호작용은 배제하지 못한다.
- 4 m 절단: 점수는 GT 경로에서 4 m 벗어난 뒤를 채점하지 않는다(무압축 rollout 57.7%가 해당).
- 시드 제어 없음: 같은 arm을 두 번 돌린 잡음 바닥이 없다(필요하면 baseline 재실행, Q4).

## 비용
Origin150 ~8시간, Ada 4장. 스모크 ~30분.

## 출력
- `alpasim-runs/m2601_bkrand_baseline_sh{0..3}`, 병합 `m2601_merged_bkrand_baseline`
- 분석: `outputs/blackout_closedloop/` (`config.json`, `metrics.json`, `summary.txt`, `plots/`)
- 리포트: `reports/evaluation/2026-10-19_camera-blackout-closedloop.html`

## 결정 (2026-10-19)
- Q1 최신 프레임 4장 중 1장 · Q2 검은 프레임은 다음 스텝에 남지 않음 · Q3 Origin150만 · Q4 기존 baseline 재사용

## 결과 (2026-10-19 14:05–23:00 KST 실행)
- **G0 PASS**: 300 rollout, 드라이버 응답 54,260개 전부 `blackout_cam` 기록·재계산 일치, 빈 궤적 0, 카메라 균등(13.4–13.7k).
  렌더러 버전·장면 파일(cmp)·채점 기준 baseline과 동일.
- **G1 FAIL**: 씬 점수 Δ +0.0238 [−0.0067, +0.0546], Wilcoxon p=0.14 (블랙아웃이 오히려 높은 방향). 평균 손해 > 0.007은 95%로 배제.
- 게이트·차선·제동·차간·CoC 모두 유의한 악화 없음. 과실 충돌 12→6건(p=0.034)은 다중 비교·잡음 바닥 미측정이라 효과로 읽지 않음.
- 연속 계획 간 변화도 baseline과 같고(1초 앞 −0.003 [−0.008, +0.001]) 가려진 카메라와 무관(망원−나머지 +0.001, p=0.65).
- 리포트: `reports/evaluation/2026-10-19_camera-blackout-closedloop.html`. docker 컨테이너 30개·네트워크 4개 정리 완료.
