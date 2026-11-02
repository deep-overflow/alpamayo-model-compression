# long-tail-detection

Claude(`claude-opus-5-5`)가 개루프 평가 클립을 long-tail / NOMINAL로 라벨링하고, 저장된 A100 개루프 결과를
그 라벨로 나눠 압축 비용이 long-tail에 몰리는지 본다. 계획과 결정 사항은 [plan.md](plan.md).

## 실행

API 키는 이 폴더의 `.env`(`ANTHROPIC_API_KEY=...`, git 무시)에 둔다. 라벨러는 torch 없는 별도 venv에서 돈다.

```bash
# 라벨링 (Claude API, 클립당 약 $0.024; labels.jsonl에 이어 쓰므로 재실행하면 빠진 것만 보냄)
/mnt/nvme1n1/ad_vla/venvs/longtail/bin/python cvpr27/long-tail-detection/label_longtail.py \
    --exp-id label_v1 --set ood_val:ood --set test_2000:test

# 분석 (CPU, 저장된 A100 클립별 결과만 읽음)
.venv/bin/python cvpr27/long-tail-detection/analyze_longtail_openloop.py --labels label_v1 --out openloop_v1
```

라벨러 venv를 다시 만들 때:

```bash
uv venv /mnt/nvme1n1/ad_vla/venvs/longtail --python 3.12
VIRTUAL_ENV=/mnt/nvme1n1/ad_vla/venvs/longtail uv pip install anthropic pillow numpy pandas pyarrow
```

결과는 `outputs/cvpr27/long_tail_detection/<exp_id>/`.

## 주의

- Claude에 보내는 것은 카메라 4대의 t0 프레임과 ego 과거 1.5 s 요약뿐이다. GT 미래와 OOD npz의 `gt_coc`/`cluster`는 보내지 않는다.
- NVIDIA AV Dataset License §4.5에 따라 프롬프트는 사람의 개인 속성 기술과 번호판 언급을 금지한다.
  데이터셋 프레임을 Anthropic API로 보내는 것은 사용자가 2026-11-02에 허용했다.
- Claude 출력에는 시드가 없다. 같은 클립을 두 번 라벨링하면 파일럿 기준 96–100% 일치한다.
