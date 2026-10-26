# AlpaSim 인계 시점 제동 버그 수정 (2026-10-26)

계획서: [`../plans/2026-10-26_alpasim-handover-fix.md`](../plans/2026-10-26_alpasim-handover-fix.md)

`/home/cvlab21/project/chan/alpasim`(2026-07-12 동기화본)에는 `.git`이 없다. 그래서 변경 내용을
`handover_fix.patch`로 이 레포에 남긴다. 바꾸기 전 원본은
`/mnt/nvme1n1/ad_vla/outputs/chan/minitron/alpasim_patch/orig/`에 있다.

## 무엇을 바꿨나

**NVlabs/alpasim main(2026-08-11 동기화본)의 수정만 이식했다.** 해당 파일은 업스트림과 같아졌다.
다른 점은 이식 주석과, 남겨 둔 `VehicleModel.set_velocity`뿐이다.

| 파일 | 변경 |
|---|---|
| `controller/.../vehicle_model.py` | `coerce_dynamic_state`: vx, vy, yaw rate, 조향각, **종방향 가속도 `state[7]`**을 GT로 덮어씀 |
| `controller/.../system.py` | 강제 GT 단계에서 `set_velocity` 대신 `coerce_dynamic_state`(요청의 `linear_acceleration.x` 사용) |
| `runtime/.../events/force_gt_utils.py` | `force_gt_dynamic_trajectory`: GT 자세 궤적에서 cubic 평활 속도·가속도·yaw rate 추정 |
| `runtime/.../events/state.py` | `RolloutState.force_gt_dynamics` |
| `runtime/.../events/controller.py` | 강제 GT 단계와 **인계 시점(`closed_loop_start_us`)**에 GT 동역학을 덮어쓰고 가속도를 전달 |
| `runtime/.../services/controller_service.py` | 요청에 `linear_acceleration`, `coerce_dynamic_state` 플래그 |
| `runtime/.../event_loop.py` | 강제 GT 기준 궤적을 인계 뒤 5초까지 연장(MPC가 마지막 자세에 멈추려고 −8 m/s²로 감속하던 원인). `force_gt_dynamics` 생성 |
| 테스트 4개 | 업스트림판(`test_event_parity`, `test_controller_api`, `test_vehicle_model`)과 `conftest`의 `force_gt_dynamics` |

**뺀 업스트림 변경**(이 버그와 무관):
- 강제 GT 프레임 캐시
- driver `random_seed`
- `policy.py` 경로 좌표계

우리 수정인 `ALPASIM_ASL_SKIP_IMAGES` 로그 필터는 그대로 두었다.

**driver slim 로더.** `driver/.../alpamayo1_5_slim.py`에 hidden 절단(`slice_hidden`, `vlm_hidden`)을 추가했다. 우리
`slim_lib.py`와 같은 코드다. Minitron-width 체크포인트가 strict 로드되게 하려는 것이다.

## 검증 (2026-10-26)

- **alpasim 테스트**(alpasim venv, 각 패키지 디렉터리에서 실행): controller 52개, runtime 464개 모두 통과.
  업스트림의 `test_controller_event_recoerces_gt_speed_at_handoff`도 포함한다.
- **slim 로더.** `mtw24_kd/ckpt_epoch3`를 vendored 로더와 `slim_lib.load_slim`으로 각각 CPU에 로드해 비교했다.
  텐서 1,159개가 전부 같고, hidden은 3328, 파라미터는 8,418,678,770개다.
- 시뮬레이터 게이트(G1–G4, 계획서 §5)는 GPU가 빈 뒤 Origin150 앞 20개 장면에서 dense로 돌린다.

## 적용과 되돌리기

```bash
# 적용 (이미 적용된 상태)
patch -d /home/cvlab21/project/chan/alpasim -p1 < minitron/alpasim/handover_fix.patch
# 되돌리기
patch -d /home/cvlab21/project/chan/alpasim -p1 -R < minitron/alpasim/handover_fix.patch
```
