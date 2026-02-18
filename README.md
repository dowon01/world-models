### World Model for CarRacing-v3
PyTorch를 활용하여 OpenAI Gym의 CarRacing 환경에서 에이전트의 미래 상태를 예측하는 World Model 구현 프로젝트입니다.

### Model Architecture

이 프로젝트는 크게 두 단계로 구성됩니다.

1. Vision State Abstraction: EMA(Exponential Moving Average) 기반의 VQ-VAE를 사용하여 고차원 이미지 데이터를 이산적(Discrete) 잠재 토큰으로 압축합니다.

2. Sequence Modeling: World Transformer를 이용해 과거 프레임과 액션 데이터를 바탕으로 미래의 잠재 토큰을 예측합니다.

---------------------------------------------------------------

- Data Collection (data_play.py)

Environment: OpenAI Gym CarRacing-v3

Method: 수동 플레이를 통해 수집된 고퀄리티 데이터 사용

Dataset: 이미지-액션 쌍(Action-Image Pair) 데이터 10,000장 구축

- Vision Model: EMA VQ-VAE (vq_vae_train.py)

이미지의 특징을 효과적으로 추출하기 위해 단순 MSE Loss 외에 인지적 유사도를 반영했습니다.

Technique: EMA(Exponential Moving Average) 업데이트 방식을 적용하여 코드북 학습의 안정성 확보

Loss Function: Reconstruction Loss + VGG LPIPS (Weight: 0.1) 적용 (시각적 디테일 보존)

Testing: vq_vae_test.py를 통해 복원 성능 검증 후 잠재 토큰(Latent Tokens) 생성

- World Model: Sequence Prediction (world_transformer.py)

Task: 연속된 5개 프레임이 주어졌을 때, 마지막 프레임의 정보를 바탕으로 다음 시점의 프레임을 예측

Training: 9(train):1(valid)로 나누어서 진행

Epoch 1/1 [train]: 100%|████████| 563/563 [5:34:47<00:00, 35.68s/batch, loss=1.8029] 
Epoch [1] 완료 - Train Loss: 3.0982, Valid Loss: 2.1327


### Result
![결과 이미지](./result.png)

​
