import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# 1. 시퀀스 데이터셋 정의
class WorldSequenceDataset(Dataset):
    def __init__(self, token_path, action_path, seq_len=5):
        # 토큰과 행동 데이터를 로드
        self.tokens = np.load(token_path) # (N, 16, 16)
        self.actions = np.load(action_path) # (N, 3)
        self.seq_len = seq_len

        # 16x16 토큰을 256개의 시퀀스로 변환
        self.tokens = self.tokens.reshape(len(self.tokens), -1) # (N, 256)

    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx):
        # 입력: 현재부터 seq_len만큼의 상태와 액션
        states = self.tokens[idx : idx + self.seq_len] # (seq_len, 256)
        actions = self.actions[idx : idx + self.seq_len] # (seq_len, 3)

        # 타겟: 다음 시점의 상태 (토큰)
        targets = self.tokens[idx + 1 : idx + self.seq_len + 1] # (seq_len, 256)
        
        return {
            "states": torch.LongTensor(states),
            "actions": torch.FloatTensor(actions),
            "targets": torch.LongTensor(targets)
        }
    
# 2. 트랜스포머 모델 정의 (next-token prediction)
class WorldTransformer(nn.Module):
    num_tokens = 512
    embedding_dim = 256
    n_heads = 8
    n_layers = 4
    seq_len = 5

    def __init__(self, num_tokens=num_tokens, embedding_dim=embedding_dim, n_heads=n_heads, n_layers=n_layers, seq_len=seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, embedding_dim)
        self.action_embedding = nn.Linear(3, embedding_dim) # 행동도 임베딩

        # 256(한 프레임 토큰 수) * seq_len 만큼의 위치 정보
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len * 256, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(embedding_dim, num_tokens) # 다음 토큰 예측

    def forward(self, states, actions):
        b, t, s = states.size() # (Batch, seq_len, 256)

        # token 임베딩 (B, T, 256, D)
        token_embeddings = self.token_embedding(states)
        # action 임베딩을 시퀀스 차원에 맞게 확장 (B, T, 1, D)
        action_embeddings = self.action_embedding(actions).unsqueeze(2)

        # action 정보를 각 token에 더함 (conditioning)
        x = (token_embeddings + action_embeddings).view(b, t * s, -1)

        # 위치 정보 추가 및 트랜스포머 연산
        x = x + self.positional_embedding
        x = self.transformer(x)

        # 결과값 출력 (B, T, 256, 512)
        logits = self.output_layer(x)
        return logits.view(b, t, s,-1)
    
    