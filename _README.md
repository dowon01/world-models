# data
openai gym -> CarRacing-v3
수동으로 플레이한 데이터 1만장을 action과 image를 세트로 하여 저장하도록 함 - data_play.py

# VAE
- EMA VQ-VAE 적용
- VGG LPIPS 적용 (가중치 0.1)
- 1만장으로 학습 진행 (valid x) - vq_vae_train.py
- 학습된 VAE 성능 확인 - vq_vqe.test.py
- 학습된 VAE를 이용하여 Latent Token 생성


# World Model
- 학습
9(train):1(valid)로 나누어서 진행
- 결과
Epoch 1/1 [train]: 100%|████████| 563/563 [5:34:47<00:00, 35.68s/batch, loss=1.8029] 
Epoch [1] 완료 - Train Loss: 3.0982, Valid Loss: 2.1327

- 연속된 5장의 프레임이 주어지면, 마지막 프레임 결과를 바탕으로 다음 프레임을 예측
- result.png


​
