import torch
import numpy as np
from torch.utils.data import DataLoader
from vq_vae import VQVAE
from dataset import RacingDataset

def generate_latent_dataset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. VQ-VAE 모델 로드
    num_hiddens = 128
    num_residual_layers = 2
    num_residual_hiddens = 32
    num_embeddings = 512
    embedding_dim = 64
    commitment_cost = 0.5

    model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost).to(device)
    model.load_state_dict(torch.load("model/vq_vae_racing.pth", map_location=device))

    # 2. 데이터셋 로드
    dataset = RacingDataset("racing_data/play_action_frames.npy")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    latent_tokens = []

    print("image -> latent token 변환 시작...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)

            # encoder -> pre-vq -> vector quantizer -> index 추출
            z = model._encoder(batch)
            z = model._pre_vq_conv(z)
            _, _, _, encoding_indices = model._vq_vae(z)

            # encoding_indices : (batch * 16 * 16, 1) -> (batch, 16, 16)로 변환
            # 모델의 출력 해상도 (16x16)에 맞게 reshape
            tokens = encoding_indices.view(-1, 16, 16).cpu().numpy()
            latent_tokens.append(tokens)

            if (i+1) % 50 == 0:
                print(f"진행 상황: {(i+1)} / 150 frames")

    # 3. 결과 저장
    latent_data = np.concatenate(latent_tokens, axis=0)
    np.save("racing_data/latent_tokens.npy", latent_data)
    print(f"Latent token 데이터 저장 완료: racing_data/latent_tokens.npy")


if __name__ == "__main__":
    generate_latent_dataset()