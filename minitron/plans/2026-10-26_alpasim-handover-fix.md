# AlpaSim 인계 시점 제동 버그 수정 → 폐루프 재평가 계획 (2026-10-26)

**사용자 결정(2026-10-26).**
1. A안(업스트림 수정만 이식)으로 한다.
2. 본 런은 **Origin150만** 돌린다.
3. **원본 Alpamayo 1.5(dense)부터** 평가한다.

이에 따라 아래 §5·§6을 다음과 같이 바꾼다.
- 검증 게이트는 Origin150 앞 20개 장면에서 dense로 본다.
- 본 런 순서는 dense → MoOP → MT-W + KD → MT-W zs다.
- Origin150 sceneset `6f937b0c…`은 남아 있다. 이전 dense 런 장면과 150/150 일치하므로 복원할 필요가 없다.

## 0. 목적

- **가설.** 인계 시점 제동 버그를 없애면 폐루프 arm 서열이 하네스가 아니라 모델의 주행을 반영한다.
- **왜 지금인가.**
  - MT-W + KD는 개루프에서 MoOP 무학습과 구별되지 않는다(test_2000 +0.019, OOD-Val +0.005; 둘 다 CI가 0을 포함).
  - 이 버그는 유지한 Q head 수와 상관된다(ρ −0.76). MT-W는 head를 하나도 자르지 않아서, MT-W vs MoOP 비교를 바로
    오염시킬 수 있다.
- **기존 규칙.** 2026-10-11 메모리: 버그를 없애고 dense·MoOP를 다시 돌리기 전에는 새 폐루프 arm을 금지한다.

## 1. 버그 (현재 코드로 확인, 2026-10-26)

1. **기준 궤적이 인계 시점에서 끝난다.**
   - 강제 GT 구간의 기준 궤적은 `event_loop.py`의 `_build_force_gt_physics_blend_trajectory`가 만든다.
   - 이 궤적은 인계 시점(`blend_end_us`)에서 끝나고, `force_gt_utils.controller_reference_trajectory`가 그 끝까지만
     자른다.
2. **MPC가 마지막 자세에서 멈추려고 감속한다.**
   - MPC는 기준 범위를 넘는 시점을 마지막 자세로 고정한다(`mpc_impl/linear_mpc.py:187`).
   - 그래서 그 자리에 멈추려 하며, 가속도가 상태 변수라 하한 −8 m/s²까지 내려간다. 입력은 가속도 변화율이고 −9~6이다.
3. **강제 단계는 속도만 덮어쓴다.**
   - 위치: `alpasim_controller/system.py:155-157`의 `set_velocity`.
   - 가속도 상태 `state[7]`은 MPC가 쌓은 감속을 그대로 들고 인계된다.
4. **결과(10월 분석).**
   - 인계 후 0.1초에 속도가 중앙값 1.06 m/s 줄고, 움직이며 시작한 경우의 77%가 −8에서 포화돼 있다.
   - 이 감속이 ego-history에 남고, VLM을 덜 자른 arm일수록 감속을 이어 간다.

## 2. 업스트림 상태 (NVlabs/alpasim main, 2026-08-11 동기화본, 원문 확인)

- 로컬 체크아웃은 2026-07-12 동기화본이다.
- 업스트림은 이 버그를 **이미 고쳤다**. CHANGELOG에는 항목이 없다.
  - **런타임 `events/controller.py`.** 강제 GT 단계와 **인계 시점**에 `coerce_dynamic_state`를 켠다. 그리고
    `state.force_gt_dynamics`에서 보간한 속도, 각속도, **선형 가속도**를 넘긴다(`rig_linear_acceleration_in_rig`).
    자유 주행 단계의 가속도는 0으로 넘긴다.
  - **`events/force_gt_utils.py`.** 새 함수 `force_gt_dynamic_trajectory`가 GT 자세 궤적에서 cubic 평활
    속도·가속도·yaw rate를 추정한다.
  - **제어기 `system.py`.** `self._vehicle_model.coerce_dynamic_state(v_cg_x, v_cg_y, yaw_rate,
    longitudinal_acceleration=request.state.state.linear_acceleration.x)`.
  - **`vehicle_model.py`.** `coerce_dynamic_state`가 `state[3]`(vx), `[4]`(vy), `[5]`(yaw rate), `[6]`(조향각, 운동학으로
    유도), **`[7]`(종방향 가속도)**를 덮어쓴다. `set_velocity`는 없어졌다.
  - **바뀌지 않은 것.** `controller_reference_trajectory`는 여전히 인계 시점에서 자른다. 하지만 인계 시점에 가속도를
    GT로 맞추므로, 남는 오차는 한 제어 스텝의 MPC 변화량(가속도 변화율 −9 이하 × 0.1 s ≈ 0.9 m/s² 이하)뿐이다.

## 3. 선택지와 추천

| 선택지 | 내용 | 장점 | 위험 |
|---|---|---|---|
| **A. 업스트림 수정만 이식 (추천)** | 위 4개 파일 + `controller_service.py`(요청에 가속도·coerce 플래그)를 7월 체크아웃에 옮김 | 이전 런과의 차이가 **이 수정 하나**뿐이라 버그 효과를 깨끗이 잴 수 있다(dense·MoOP의 수정 전후 = MoOP 논문 폐루프 행이 버그에 얼마나 영향받았는지) | 7월 → 8월 사이 연결된 변경을 놓칠 수 있음 → 파일별 diff로 확인 |
| B. 8월 동기화본으로 전체 업그레이드 | 업스트림 통째로 + 우리 slim-eval 패치 재적용 | 업스트림과 같은 상태 | 채점·장면·스케줄링이 함께 바뀌어 이전 런과 비교 불가. entry point가 바뀌면 이미지 재빌드 필요 |
| C. 직접 수정(기준 궤적을 인계 뒤까지 연장) | GT를 인계 뒤 5초까지 이어 붙임 | 근본 원인 제거 | 업스트림과 다른 해법이 됨. A가 이미 충분 |

## 4. 구현 (A안)

1. **백업과 기록.**
   - alpasim 체크아웃에 **`.git`이 없다**(메모리의 `slim-eval` 브랜치는 현재 버전 관리 밖이다).
   - 바꿀 파일의 원본을 `outputs/minitron/alpasim_patch/orig/`에 복사한다.
   - 변경은 우리 레포 `minitron/alpasim/handover_fix.patch`로 남긴다.
2. **우리 패치와의 충돌 확인.**
   - 업스트림 7월 동기화본과 로컬을 파일별로 비교해 우리 slim-eval 수정(ASL 이미지 필터 등)을 가려낸다.
   - 이식할 파일에 우리 수정이 있으면 함께 보존한다.
3. **proto 확인.** 7월 proto의 `DynamicState`에 `linear_acceleration`이 있는지 확인한다(`system.py:222`가 이미 가속도를 채우므로
   있을 것으로 보인다). 없으면 proto 재생성이 필요하므로 중단하고 보고한다.
4. **업스트림 테스트도 이식해 실행한다.** controller/runtime 테스트, alpasim venv.
5. **MT-W 로더.**
   - driver의 vendored `alpamayo1_5_slim.py`에는 hidden 절단(`vlm_hidden`, `slice_hidden`)이 없다. 그래서 MT-W
     체크포인트가 strict 로드에서 실패한다.
   - 우리 `slim_lib.py`의 해당 부분(`_keep_rows`, `_keep_norm`, `slice_hidden`, `apply_surgery(vlm_hidden=)`)을 옮긴다.
   - CPU에서 `load_slim` 결과가 우리 로더와 텐서 단위로 같은지 확인한다.
6. **편집 시점.**
   - `src/`는 컨테이너에 bind-mount돼 수정이 즉시 반영된다. 그래서 우리 alpasim 컨테이너가 돌고 있지 않을 때 편집한다.
   - 현재 `docker ps`에는 `hk`(7일째, 우리 것 아님)만 있다.

## 5. 검증 게이트 (사전 등록)

같은 Hard100 앞 20개 장면 × 1 rollout, dense, 수정 전 런과 비교한다. 측정은 `analyze_handover.py`로 한다.

| 게이트 | 판정 |
|---|---|
| G1 | 인계 후 0.1초 속도 변화 − 같은 구간 GT 속도 변화의 중앙값 절댓값 **< 0.1 m/s** (수정 전 약 1.06) |
| G2 | 움직이며 시작한 경우 인계 시점 가속도가 −8에서 포화된 비율 **< 5%** (수정 전 77%) |
| G3 | 강제 GT 구간 ego 자세가 수정 전과 같음(최대 차이 < 1 cm). 이 구간은 GT로 덮으므로 바뀌면 안 된다 |
| G4 | rollout 오류 0 (smoke) |

하나라도 실패하면 본 런을 하지 않고 원인을 보고한다.

## 6. 본 런 (게이트 통과 후)

- **장면.** Hard100(`outputs/scene_difficulty/hard100_suite.csv`), 장면당 rollout 2개. 이전 런과 같은 설정이다.
- **arm.** 같은 하네스로 4개를 돌린다.
  1. dense
  2. MoOP(`slim_dual_u40_v2`)
  3. MT-W + KD(`ckpt_epoch3`)
  4. MT-W zs
- **시간.** 이전 Hard100에서 dense가 Ada 4–7 기준 310분 걸렸다. slim arm은 driver 스텝이 약 절반이라 3–4시간으로
  보인다. 합계 약 15시간이다.
- **Origin150(선택).** sceneset 파일이 없어져 기존 런 결과에서 장면 ID를 복원해야 한다.
- **분석.**
  - 장면별 페어드로 비교한다. 지표는 score와 progress를 함께 보고, at-fault 충돌, 정지 준수율, 경로 끝을 고려한
    절단 버전을 본다(메모리 alpasim-metric-selection, 4m 절단).
  - dense와 MoOP는 수정 전 런과도 비교해, 버그가 서열에 준 영향을 잰다.
- **정리.** 런이 끝나면 우리 docker 컨테이너와 네트워크를 정리한다(사용자 규칙).

## 6b. 진행 기록 (2026-10-26)

- **업스트림 수정 이식 완료.**
  - 패치는 `minitron/alpasim/handover_fix.patch`이고, 원본은 `outputs/minitron/alpasim_patch/orig/`에 백업했다.
  - 업스트림은 가속도 덮어쓰기에 더해 **강제 GT 기준 궤적을 인계 뒤 5초까지 연장**했다(`event_loop.py`의 `reference_end_us`).
    둘 다 이식했다.
  - alpasim 테스트는 controller 52개, runtime 464개가 모두 통과했다.
- **driver 로더.**
  - `slice_hidden`을 추가했다. `mtw24_kd/ckpt_epoch3`가 `slim_lib`과 텐서 1,159개 모두 같다.
  - MT-W zs는 `slim_state.pt`가 없었다. 그래서 고정 revision으로 `outputs/minitron/slim_mtw24_state/`를 만들었다.
  - driver 폴더에 `slim_mtw24`(zs)와 `slim_mtw24_kd`를 하드링크했다.
- **게이트 판정 스크립트** `minitron/code/hofix_gates.py`.
  - 수정 전 런으로 시험한 결과, 게이트 20개 장면의 움직이며 시작한 36개 rollout에서 인계 직후 차 −1.058 m/s,
    사람 +0.005 m/s였다. 인계 시점 가속도는 중앙값 −8.00 m/s², 포화 72%로 10월 분석이 재현됐다.
- **게이트 실행.** `STAGE=gate minitron/code/closedloop_hofix.sh`로 01:42–02:16 UTC에 돌렸다. 대상은 Origin150 앞 20개
  장면 × rollout 1개, dense다. 결과는 `outputs/minitron/hofix_gates/`.

| 게이트 | 수정 전 (`m2601_merged_baseline`, 움직이며 시작한 36개) | 수정 후 (`hofix_gate_merged_baseline`, 18개) | 판정 |
|---|---|---|---|
| 인계 직후 첫 스텝: 차 / 사람 | −1.058 / +0.005 m/s | +0.003 / +0.005 m/s | |
| G1 \|차 − 사람\| 중앙값 < 0.1 m/s | 1.055 | **0.015** | PASS |
| G2 인계 시점 가속도 포화(≤ −7.9) < 5% | 72% (중앙값 −8.00) | **0%** (중앙값 +0.05) | PASS |
| G3 강제 GT 구간 자세 차이 < 1 cm | | **0.0000 m** (20개 장면) | PASS |
| G4 rollout 완료 | | **20/20** | PASS |

- **본 런.** `STAGE=main`을 02:20 UTC에 시작했다. 150개 장면 × 2, dense → MoOP → MT-W + KD → MT-W zs 순서다.
- **2026-10-26 사용자 결정: "나머지 arm은 안 해도 될 것 같아."** dense만 돌린다.
  - 04:20 UTC에 상위 스크립트를 멈췄고, dense 런처는 계속 돈다.
  - 병합과 컨테이너 정리는 `finish_dense.sh`가 맡는다. 결과는 `alpasim-runs/hofix_merged_baseline`에 들어가고, 완료 표시는
    `outputs/minitron/closedloop_hofix_main.DONE`이다.

## 7. 결정 필요

1. A안(업스트림 수정만 이식)으로 진행할지.
2. 본 런 arm 4개와 Hard100만으로 할지, 아니면 Origin150 복원까지 할지.
3. 수정 후 수치는 MoOP 논문 Table 2의 폐루프 행과 직접 비교할 수 없다. 새 연구에서는 모든 arm을 새 하네스로 다시
   돌린다.
