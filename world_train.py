from world_transformer import WorldTransformer, WorldSequenceDataset
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def train_world_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"학습 시작 : {device} 사용")

    # 설정
    seq_len = 5
    batch_size = 16
    num_epochs = 1
    learning_rate = 2e-4
    embedding_dim = 64
    num_tokens = 512
    n_heads = 8
    n_layers = 4

    # 데이터셋 및 데이터로더
    dataset = WorldSequenceDataset(
        token_path='racing_data/latent_tokens.npy', 
        action_path='racing_data/play_actions.npy',
        seq_len=seq_len
    )

    # 90%는 학습용, 10%는 검증용으로 분할
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 모델, 손실 함수, 옵티마이저
    model = WorldTransformer(num_tokens=num_tokens, embedding_dim=embedding_dim, n_heads=n_heads, n_layers=n_layers, seq_len=seq_len).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_history = []
    valid_history = []

    for epoch in range(num_epochs):
        # 학습 단계
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]", unit="batch")

        for batch in train_pbar:
            states = batch["states"].to(device) # (B, seq_len, 256)
            actions = batch["actions"].to(device) # (B, seq_len, 3)
            targets = batch["targets"].to(device) # (B, seq_len, 256)

            optimizer.zero_grad()

            # 예측 (logits 모양 : B, T, 256, 512)
            logits = model(states, actions)
            
            # loss 계산을 위해 차원 정렬 (CrossEntropy는 클래스 차원이 두 번째여야 함)
            loss = criterion(logits.reshape(-1, 512), targets.reshape(-1))
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        train_history.append(avg_train_loss)

        # 검증 단계
        model.eval()
        total_valid_loss = 0
        with torch.no_grad(): # 검증 시에는 gradient 계산 안 함 (메모리 절약)
            for batch in valid_loader:
                states = batch["states"].to(device)
                actions = batch["actions"].to(device)
                targets = batch["targets"].to(device)
                
                logits = model(states, actions)
                loss = criterion(logits.reshape(-1, 512), targets.reshape(-1))
                total_valid_loss += loss.item()
        
        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_history.append(avg_valid_loss)

        print(f"Epoch [{epoch+1}] 완료 - Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

        # 모델 저장
        torch.save(model.state_dict(), "model/world_transformer.pth")

    # 손실 그래프 저장
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label="Train Loss")
    plt.plot(valid_history, label="Valid Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("model/world_model_loss.png")
    plt.show()

    print("학습 완료")

if __name__ == "__main__":
    train_world_model()