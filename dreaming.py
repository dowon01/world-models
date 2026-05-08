import torch
import numpy as np
import matplotlib.pyplot as plt
from world_transformer import WorldTransformer
from vq_vae import VQVAE
import random

def visualize_imagination():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 모델 초기화
    vae = VQVAE(num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, num_embeddings=512, embedding_dim=64, commitment_cost = 0.5).to(device)
    vae.load_state_dict(torch.load("model/vq_vae_racing.pth", map_location=device))

    world_model = WorldTransformer(num_tokens=512, embedding_dim=64, n_heads=8, n_layers=4, seq_len=5).to(device)
    world_model.load_state_dict(torch.load("model/world_transformer.pth", map_location=device))

    vae.eval()
    world_model.eval()

    # 2. 데이터 준비 (랜덤 시퀀스)
    tokens = np.load("racing_data/latent_tokens.npy")  # (N, 16, 16)
    actions = np.load("racing_data/play_actions.npy")  # (N, 3)
    original_frames = np.load("racing_data/play_action_frames.npy")  # (N, 64, 64, 3)

    idx = random.randint(0, len(tokens)-5-1)  # 시퀀스 길이 고려
    input_tokens = torch.LongTensor(tokens[idx:idx+5]).view(1, 5, -1).to(device)  # (1, 5, 256)
    input_actions = torch.FloatTensor(actions[idx:idx+5]).view(1, 5, -1).to(device)  # (1, 5, 3)
    target_token = tokens[idx+5]  # 다음 프레임의 정답 토큰

    # 3. 상상하기
    with torch.no_grad():
        logits = world_model(input_tokens, input_actions)  # (1, 5, 256, 512)
        # 마지막 프레임의 결과를 바탕으로 그 다음을 예측함
        pred_token_indices = torch.argmax(logits[:, -1, :, :], dim=-1)

        # 4. VQ-VAE로 디코딩
        # 예측된 토큰을 16x16 형태로 변환
        z = vae._vq_vae.get_codebook_entry(pred_token_indices, shape=(1, 16, 16, 64)) 
        imagined_frame = vae._decoder(z).cpu().squeeze(0).permute(1,2,0).numpy()

        # 실제 정답 이미지 복원
        target_indices = torch.LongTensor(target_token).view(1, 256).to(device)
        target_z = vae._vq_vae.get_codebook_entry(target_indices, shape=(1, 16, 16, 64))
        target_frame = vae._decoder(target_z).cpu().squeeze(0).permute(1,2,0).numpy()

        # 원본 프레임 (복원 전)
        raw_frame = original_frames[idx+5] # 디코더를 거치지 않음 (64, 64, 3)

    # 5. 시각화
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.title("Actual Next Frame")
    plt.imshow(target_frame)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Imagined Next Frame")
    plt.imshow(imagined_frame)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Original Raw Frame")
    plt.imshow(raw_frame)
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    visualize_imagination()