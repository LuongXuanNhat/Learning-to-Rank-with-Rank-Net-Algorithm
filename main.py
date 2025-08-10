import json
import numpy as np
from itertools import combinations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Library.TF_IDF import TFIDF
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
    # documents_path = "FetchData/output/cadao_tucngu_complete.json"
    documents_path = "FetchData/output/cadao_tucngu_medium.json"
    documents = load_documents_from_json(documents_path)
    print("Số tài liệu:", len(documents))

    stopword_path = "../../vietnamese-stopwords.txt"
    tfidf_model = TFIDF(documents, stopword_path=stopword_path)
    matrix = tfidf_model.fit_transform()
    vocab = tfidf_model.get_feature_names()
    print("Số từ trong từ điển:", len(vocab))
    print("Kích thước ma trận TF-IDF:", len(matrix))

    # Bước 2: Tạo cặp dữ liệu huấn luyện
    query = "ăn quả"
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

    # Split training/validation
    split_idx = int(0.8 * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    print(f"Training pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}")

    # Xuất cặp dữ liệu (tùy chọn)
    # export_training_pairs(pairs, query, "training_pairs.json")

    # Bước 3: Huấn luyện RankNet
    model = RankNet(input_size=len(vocab))
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Giảm learning rate
    
    train_dataset = RankNetDataset(train_pairs, matrix)
    val_dataset = RankNetDataset(val_pairs, matrix)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Giảm batch size
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    epochs = 200
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    model.train()
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for di_vec, dj_vec, pij in train_dataloader:
            optimizer.zero_grad()
            si = model(di_vec)  # Điểm số cho dᵢ
            sj = model(dj_vec)  # Điểm số cho dⱼ
            loss = ranknet_loss(si, sj, pij)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for di_vec, dj_vec, pij in val_dataloader:
                si = model(di_vec)
                sj = model(dj_vec)
                loss = ranknet_loss(si, sj, pij)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

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