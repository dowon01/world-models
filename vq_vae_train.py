from dataset import RacingDataset
from vq_vae import VQVAE
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import lpips

def train():
    # 장치 설정 (MAC -> mps)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 장치: {device}")

    # lpips 모델 초기화
    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)

    # 하이퍼파라미터
    batch_size = 64
    num_training_steps = 5000
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 64
    num_embeddings = 512    # 단어장 크기
    commitment_cost = 0.5
    learning_rate = 2e-4

    # 모델 및 옵티마이저 초기화
    model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    # 데이터셋 로드 (RacingDataset 사용)
    dataset = RacingDataset('racing_data/play_action_frames.npy')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 학습 시작
    print(f"학습 시작 - Device : {device}")
    model.train()
    
    for i in range(num_training_steps):
        try:
            data = next(iter_dataloader)
        except:
            iter_dataloader = iter(dataloader)
            data = next(iter_dataloader)

        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data)

        # LPIPS loss -> LPIPS는 입력 범위를 [-1, 1]로 기대하므로 정규화가 필요
        data_input_norm = (data * 2) - 1
        data_recon_norm = (data_recon * 2) - 1
        lpips_loss = loss_fn_vgg(data_input_norm, data_recon_norm).mean()

        # LPIPS 가중치 조정
        loss = recon_error + vq_loss + (0.1 * lpips_loss)
        
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            # unique_indices = torch.unique(encoding_indices)
            # print(f"Unique encodings used: {len(unique_indices)}/{num_embeddings}")

            print(f"Step {i+1}/{num_training_steps}, Loss: {loss.item():.4f}, Recon Error: {recon_error.item():.4f}, VQ Loss: {vq_loss.item():.4f}, LPIPS: {lpips_loss.item():.4f}, Perplexity: {perplexity.item():.4f}")

    # 모델 저장
    torch.save(model.state_dict(), "model/vq_vae_racing.pth")
    print("모델 저장 완료: model/vq_vae_racing.pth")

if __name__ == "__main__":
    train()