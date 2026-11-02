# tyr_vt: tyr_off에서 헤시안의 비전·텍스트 비중만 1:1로 (A100 빌드·평가)

날짜: 2026-11-01. 상태: **사용자 지시로 진행** ("(a)만 진행해줘. A100에서 빌드랑 평가 둘다").

## 목적
- `tyr_off`의 OSSCAR 재구성 헤시안은 calib_100 prefill 토큰을 한 번씩 더해 비전 쪽(비전 + 과거 궤적 + sink)이 94.9%,
  프롬프트 텍스트가 5.1%다. 결과는 주행 = dense, LingoQA 38.0%(붕괴).
- 두 쪽을 토큰 수로 정규화한 뒤 1:1로 섞으면(H = ½·H_V/N_V + ½·H_T/N_T) 재구성이 텍스트 위치 출력을 더 지키는지,
  그 대가로 주행이 얼마나 변하는지 잰다. 저장소에서 V:T 동일 비중은 처음이다.
- 참고(dual 선택 + 재구성, 2026-08-30): 텍스트만의 H는 조건수로 탈락, 자기 CoC 16% 혼합은 LingoQA −18.2pp(암기),
  LingoQA train을 H에 넣은 dualr_wl만 72.6%로 회복. 따라서 V:T 1:1의 LingoQA 회복은 제한적일 수 있다(예상).

## 설정
- `tyr_off`와 같은 것: 출발점 s = 0.39856(제거 2,657,452,032 = MoOP), supernet·탐색 4회, 수준 격자, 식 5 오차 누적,
  탐색(50 × 128, 1/5/42 클립, 생존 16/4/2), top-8,192 텍스트 위치 KL 적합도, teacher(`cvpr27/tyr_off/teacher` 재사용).
- 다른 것 하나: `supernet.py --hessian-mix vt_equal`. 클립마다 `compute_spans`의 text 마스크로 V/T를 나눠 `StreamHessianHook`으로
  쌓고 `mix_hessians(H, n, {V: ½, T: ½})`. CPU 단위 검사로 식 일치 확인(차 2.4e-7).
- 빌드·평가 모두 KISTI A100(`build_tyr_off.sbatch`에 `TYR_OUT=cvpr27/tyr_vt` 등 환경 변수, 4장 잡).
  계정 한도(A100 4장) 때문에 l0keep 비율 평가가 끝난 뒤 시작한다.
- 평가: A100 트랙 arm `tyr_vt`(test_2000, OOD-Val262, LingoQA), REFS = MoOP, MoOP l0keep, tyr_off.

## 게이트
- 매 supernet 실행의 0번 블록에서 조기 종료 헤시안 = 전체 prefill 헤시안(혼합 후).
- 라운드마다 유지 총량 불변(684 head, 266,040 채널), 최종 구성에 head·채널 0개인 층 없음.

## 비용
- 빌드 약 7–8 A100-시간 × 4장 ≈ 30 A100-시간(벽시계 7–8시간), 평가 약 6 A100-시간.
