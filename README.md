# Roadwatch

**What If? #01 — 처음 보는 도로 장애물도 발견할 수 있다면?**

스카이오토넷 재직 당시 진행한 지능형 도로 관제 시스템 프로젝트에서, 학습 대상으로 정의하지 않은 장애물을 놓치는 문제가 있었다.
Roadwatch는 VFM과 VLM 모델을 활용해 기존 검출 모델의 학습 범위 밖에 있는 도로 장애물의 인식 가능성을 실험하는 프로젝트다.

## 사용법

Ubuntu 22.04, Python 3.10, RTX 3070 Ti 8GB에서 실행을 확인했다. 아래 명령은 저장소 디렉터리에서 실행한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/prepare_road_anomaly.py
```

필요하면 다음 명령으로 현재 PyTorch 환경의 CUDA 사용 가능 여부를 확인할 수 있다.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

`outputs/road-anomaly-audit.json`의 `records[].image`에서 입력 이미지 경로를 선택한다.

```bash
export HF_HOME="$PWD/.cache/huggingface"
python scripts/probe_detector.py <이미지_경로>
python scripts/probe_vlm.py <이미지_경로>
```

최초 실행 시 모델 가중치를 다운로드하므로 수 GB 이상의 저장 공간이 필요하다.
두 추론 스크립트는 순서대로 실행하며, 결과는 `outputs/`에 JSON으로 저장된다. 같은 출력 경로로 재실행하면 이전 결과를 덮어쓴다.

데이터는 [Road Anomaly](https://www.epfl.ch/labs/cvlab/data/road-anomaly/)를 사용한다.
연구용으로 제공되며 이미지별 출처와 이용 조건은 다운로드한 데이터의 `credits.txt`에서 확인할 수 있다.
