import json
import numpy as np
from itertools import combinations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Library.TF_IDF.TF_IDF import TFIDF
from helper import  load_documents_from_json, ranknet_loss, relevance_score_tfidf
from Library.rank_net import RankNet



# Dataset cho RankNet
class RankNetDataset(Dataset):
    def __init__(self, pairs, tfidf_matrix):
        self.pairs = pairs
        self.tfidf_matrix = tfidf_matrix

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (di_vec, dj_vec), pij = self.pairs[idx]
        return (torch.tensor(di_vec, dtype=torch.float32),
                torch.tensor(dj_vec, dtype=torch.float32),
                torch.tensor(pij, dtype=torch.float32))

if __name__ == "__main__":
    # Bước 1: Tải tài liệu và biểu diễn TF-IDF
    documents_path = "FetchData/output/cadao_tucngu_mini.json"
    documents = load_documents_from_json(documents_path)
    print("Số tài liệu:", len(documents))

    stopword_path = "../../vietnamese-stopwords.txt"
    tfidf_model = TFIDF(documents, stopword_path=stopword_path)
    matrix = tfidf_model.fit_transform()
    vocab = tfidf_model.get_feature_names()
    print("Số từ trong từ điển:", len(vocab))
    print("Kích thước ma trận TF-IDF:", len(matrix))

    # Bước 2: Tạo cặp dữ liệu huấn luyện
    query = "ăn"
    scores = [relevance_score_tfidf(matrix, vocab, query, i) for i in range(len(documents))]
    print(f"Điểm liên quan của từ '{query}': {scores}")

    pairs = []
    for i, j in combinations(range(len(documents)), 2):
        if scores[i] == scores[j]:
            continue
        dij = (matrix[i], matrix[j])  # Cặp vector TF-IDF
        pij = 1 if scores[i] > scores[j] else 0
        pairs.append((dij, pij))
    print(f"Số cặp tài liệu: {len(pairs)}")

    # Xuất cặp dữ liệu (tùy chọn)
    # export_training_pairs(pairs, query, "training_pairs.json")

    # Bước 3: Huấn luyện RankNet
    model = RankNet(input_size=len(vocab))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = RankNetDataset(pairs, matrix)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    epochs = 100
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for di_vec, dj_vec, pij in dataloader:
            optimizer.zero_grad()
            si = model(di_vec)  # Điểm số cho dᵢ
            sj = model(dj_vec)  # Điểm số cho dⱼ
            loss = ranknet_loss(si, sj, pij)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")

    # Bước 4: Dự đoán và xếp hạng
    model.eval()
    scores_rank = []
    with torch.no_grad():
        for i in range(len(documents)):
            score = model(torch.tensor(matrix[i], dtype=torch.float32)).item()
            scores_rank.append((score, documents[i]))

    ranked = sorted(scores_rank, key=lambda x: x[0], reverse=True)
    print("Kết quả xếp hạng:")
    for score, doc in ranked:
        print(f"{score:.4f} - {doc}")