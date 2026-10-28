# MoOP(dual) 폐루프: s600 기준선과 짝지은 비교 (2026-10-28)

## 0. 목적

- **무엇을 재나.** 출하본 MoOP(`slim_dual_u40_v2`, VLM −24.0%)의 폐루프 성능을 dense 기준선(`s600_merged_baseline`, score 0.729)과
  **같은 600씬에서 씬별로 짝지어** 비교한다. 하네스는 인계 버그를 고친 버전이다.
- **왜 다시 재나.** 기존 MoOP 폐루프 결과(Origin150 +0.079, Hard100 +0.085)는 모두 버그 수정 전 하네스에서 나왔다. 그 버그는 VLM을
  안 자른 모델만 인계 감속을 이어 가게 만들어서 dense에 불리했다. 그래서 "MoOP > dense"가 하네스 탓인지 모델 탓인지 아직 모른다.
- **가설 검정은 아니다.** 방향을 미리 정하지 않고, 차이와 CI를 보고한다.
  - 짝지은 차이의 검출 한계는 약 0.035다(600씬 × 1 rollout, 씬 안 분산 0.026 기준).

## 1. 사용자 결정 (2026-10-28)

- 체크포인트: 출하본 MoOP `slim_dual_u40_v2`. 0번 층 Q head 13개를 자른 논문 arm 그대로다. driver 폴더에 하드링크가 이미 있다.
- 씬과 rollout: 기준선과 같은 `outputs/alpasim_dense_s600/scenes_s600.csv` 600씬 × 1.

## 2. 설계

- **설정은 기준선과 같다.** 패치된 alpasim, `pair_a15`, 카드당 shard 1개, `DRIVER_OMP_THREADS=8`, `ALPASIM_ASL_SKIP_IMAGES=1`, GPU 8장.
  바뀌는 것은 driver 체크포인트 하나다(`driver.model.checkpoint_path=/mnt/drivers/slim_dual_u40_v2`).
- **씬마다 기준선과 같은 카드에서 돌린다.** 아키텍처 효과가 미확정이라(같은 씬 안 −0.027 [−0.105, +0.045]), 짝지은 비교에서 그 효과가
  상쇄되게 한다.
  - 런처는 지금 scene_id 순 round-robin으로만 나눈다. 그래서 기준선의 배정(`scene_gpu.csv`, smoke + 본 런)이 재현되지 않는다.
  - **런처 수정**: `SCENES_CSV`에 `shard` 열이 있으면 round-robin 대신 그 열대로 나눈다. 없으면 기존 동작 그대로라서 이전 런과
    호환된다.
  - 카드당 75씬으로 고정되므로 Blackwell이 일찍 끝나는 불균형은 감수한다.
- **smoke (게이트, 본 추정에 포함).** 카드마다 그 카드에 배정된 씬 1개씩, 총 8씬을 돌린다. 중단 7씬은 고르지 않는다.
  - G1: 8/8 채점. slim 체크포인트가 Blackwell에서도 로드되고 돈다(slim 경로는 Blackwell 폐루프에서 처음이다).
  - G2: 인계 직후 \|차 − 사람\| 속도 변화 중앙값 < 0.1 m/s, 가속도 포화 0.
  - driver 로그에서 slim 로드(`slim_meta.json` 경로)와 파라미터 수를 확인한다.
  - 통과하면 본 런은 이 8씬을 건너뛰고, 끝난 뒤 합친다(기준선과 같은 방식).
- **시간.** slim driver는 dense보다 스텝이 짧다. 8장 기준 약 4–5시간으로 예상한다.
- **이름.** 로그 접두사 `d600smoke_` / `d600_`, 병합 결과 `alpasim-runs/d600_merged_slim_dual_u40_v2`.
- **정리.** 끝나면 우리 컨테이너와 네트워크를 지운다.

## 3. 분석

- **짝지은 차이.** dual − dense, 씬 단위. 층 가중 평균과 층 안 부트스트랩 20,000회, Wilcoxon을 함께 낸다.
  - 지표: score, progress, passed, 충돌(전체, 과실), 도로 이탈, 주행 거리, plan_deviation.
  - 경로 끝을 인지하는 4 m 절단 버전도 낸다.
- **중단 씬.** 두 arm 각각의 중단 씬과 합집합을 뺀 값도 낸다. 기준선 7씬이 dual에서도 중단되는지 확인한다.
- **부분 집합.** Origin150, Hard100 ∩ 표본, 카드 아키텍처별(짝지은 차이가 아키텍처마다 같은지).
- **CoC 퇴화율.** ASL의 `reasoning_text`로 낸다(`analyze_alpasim.py`의 규칙).
- **참고.** 수정 전 하네스의 dual − dense(Origin150 +0.079)를 같은 겹침 씬에서 나란히 놓는다. 하네스 수정이 격차를 얼마나 바꿨는지
  보는 용도다.
- **산출물.**
  - `outputs/alpasim_dual_s600/`: `config.json`, `metrics.json`, `summary.txt`, 그림
  - 리포트: `reports/evaluation/<날짜>_dual-closedloop-s600.md`

## 4. git

- 브랜치 `dual-closedloop-s600`
- 커밋 대상: 런처 `shard` 열 지원, 실행 래퍼, 분석 스크립트, 리포트
- 끝나면 PR을 연다.

## 5. 진행 기록 (2026-10-28 ~ 29)

- **smoke (15:28–15:39 UTC).** 8/8 채점, 8개 카드 모두 slim 체크포인트 인식, G2 0.019 m/s, 포화 0/8. 통과했다.
- **본 런 (15:39–22:47 UTC).** 592씬, 600/600 병합. 600씬 모두 기준선과 같은 카드에서 돌았다. 컨테이너와 네트워크는 모두 정리했다.
  - 처리량은 dense와 거의 같았다(100회 추론당 168–196초). 카드 하나에 모든 서비스를 올린 배치에서는 slim driver의 이점이
    처리량으로 드러나지 않았다. 그래서 4–5시간 예상이 약 7시간으로 늘었다.
  - 중단은 공통 7씬과 MoOP에서만 중단된 `ceb3d0d5`(경로 후반 Waypoint 237)다.
- **결과.** `reports/evaluation/2026-10-29_dual-closedloop-s600.md`. MoOP − dense = +0.037 [+0.012, +0.061]이고, progress는 +0.043이다.
  같은 씬에서 수정 전 격차와 비교하면 Origin150은 +0.074 → −0.006, Hard100은 +0.099 → +0.073이다.
