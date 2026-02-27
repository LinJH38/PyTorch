# Deep Learning for Automatic Modulation Classification (AMC)

**RadioML 2016.10a** 데이터셋을 활용하여 무선 통신 신호의 변조 방식(Modulation Schemes)을 자동으로 분류하는 딥러닝 모델 구현 과정

## Project Overview
통신 환경에서는 다중 경로 페이딩이나 AWGN(가산 백색 가우시안 잡음)으로 인해 수신된 I/Q 신호가 크게 훼손될 수 있음. 이 프로젝트는 노이즈가 섞인 원시 시계열 I/Q 데이터를 입력받아 1D CNN과 LSTM을 거쳐 어떤 변조 방식(예: QAM16, GFSK 등 11개 클래스)이 사용되었는지 예측하는 것.

## Dataset
* **RadioML 2016.10a** (`RML2016.10a_dict.pkl`)
* **Features:** I/Q 채널의 시계열 데이터 (Shape: `2 x 128`)
* **Labels:** 11개 변조 방식 (8PSK, AM-DSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM 등)
* **SNR Range:** -20dB ~ +18dB

> **Note:** 데이터셋은 용량 문제로 포함되어 있지 않음. 코드를 실행하려면 데이터를 다운로드하여 `data/raw/` 폴더에 위치.

## Model Architectures
1. **Basic 1D CNN:** * `Conv1d` 커널을 주파수 판별기(Frequency Detector)처럼 활용하여 I/Q 채널의 위상 변화(Phase Shift) 패턴 추출.
   * `MaxPool1d`를 통해 노이즈(AWGN)를 무시하고 가장 특징적인 파동(Peak)만 남겨 모델의 Robustness를 높입니다.
2. **CNN + LSTM:**
   * 공간적/주파수적 특징은 CNN으로 추출하고, 그 특징들의 시계열적 흐름을 LSTM으로 기억하여 최종 분류를 수행합니다.

## Getting Started
```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 주피터 노트북 실행
jupyter notebook notebooks/01_RadioML2016_10A_tutorial.ipynb
