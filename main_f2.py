import json
import math
from typing import List, Dict
import re
from collections import Counter, defaultdict
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import combinations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer
from tqdm import tqdm


class RankNet(nn.Module):
    def __init__(self, input_size, hidden_size1=256, hidden_size2=128, dropout=0.2):
        """
        RankNet với 2 tầng ẩn
        
        Args:
            input_size (int): Số features đầu vào
            hidden_size1 (int): Số neurons tầng ẩn thứ nhất
            hidden_size2 (int): Số neurons tầng ẩn thứ hai  
            dropout (float): Tỷ lệ dropout để tránh overfitting
        """
        super(RankNet, self).__init__()
        
        # Định nghĩa các tầng
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)  # Output layer cho score
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer normalization thay vì batch normalization để tránh lỗi khi batch_size=1
        self.ln1 = nn.LayerNorm(hidden_size1)
        self.ln2 = nn.LayerNorm(hidden_size2)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor có shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output scores có shape (batch_size, 1)
        """
        # Đảm bảo input là tensor và có đúng kiểu dữ liệu
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Tầng ẩn thứ nhất
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Tầng ẩn thứ hai
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x
    
    def predict_rank(self, x1, x2):
        """
        So sánh ranking giữa hai samples
        
        Args:
            x1, x2 (torch.Tensor): Hai samples cần so sánh
            
        Returns:
            torch.Tensor: Xác suất x1 được rank cao hơn x2
        """
        score1 = self.forward(x1)
        score2 = self.forward(x2)
        
        # Sử dụng sigmoid để chuyển về xác suất
        prob = torch.sigmoid(score1 - score2)
        return prob

def ranknet_loss(s_i, s_j, P_ij):
    diff = s_i - s_j
    P_hat = torch.sigmoid(diff)  # Xác suất dự đoán P̂ᵢⱼ
    # Thêm epsilon để tránh log(0)
    epsilon = 1e-10
    loss = -P_ij * torch.log(P_hat + epsilon) - (1 - P_ij) * torch.log(1 - P_hat + epsilon)
    return loss.mean()

class RankNetDataset(Dataset):
    """Dataset cho RankNet training"""
    def __init__(self, pairwise_data, sentence_model):
        self.data = pairwise_data
        self.sentence_model = sentence_model
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Encode query và documents
        query_emb = self.sentence_model.encode([sample['query']])[0]
        doc1_emb = self.sentence_model.encode([sample['doc1_text']])[0]
        doc2_emb = self.sentence_model.encode([sample['doc2_text']])[0]
        
        # Tạo features bằng cách concat query+doc
        feature1 = np.concatenate([query_emb, doc1_emb])
        feature2 = np.concatenate([query_emb, doc2_emb])
        
        return {
            'feature1': torch.tensor(feature1, dtype=torch.float32),
            'feature2': torch.tensor(feature2, dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.float32)
        }

class DocumentRetriever:
    def __init__(self, documents, sentence_model, stopwords_path=None):
        self.documents = documents
        self.model = sentence_model
        self.stopwords = set()
        
        # Load stopwords nếu có
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f if line.strip())
    
    def preprocess(self, text):
        """
        [Đã nâng cấp] Sử dụng PyVi
        Tách từ đơn giản (theo khoảng trắng) và loại bỏ stopword.
        Args:
            text (str): Câu đầu vào.
        Returns:
            List[str]: Danh sách từ đã lọc stopword.
        """
        # 1. Chuẩn hóa: bỏ dấu câu, ký tự đặc biệt, số (nếu cần)
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # loại bỏ dấu câu
        text = re.sub(r'\d+', ' ', text)      # loại bỏ số
        text = re.sub(r'\s+', ' ', text).strip()  # bỏ khoảng trắng thừa

        # 2. Tokenize với PyVi
        tokenized = ViTokenizer.tokenize(text) # từ đơn, từ ghép, cụm từ cố định | Ông Nguyễn_Tấn_Dũng là cựu_thủ_tướng Việt_Nam .

        # 3. Tách thành danh sách từ
        tokens = tokenized.split()

        # 4. Loại bỏ stopwords nếu có
        if self.stopwords:
            tokens = [w for w in tokens if w not in self.stopwords]

        return tokens

    def generate_pairwise_training_data(self, queries: List[str], documents: List[Dict]) -> List[Dict]:
        """
        Tạo dữ liệu pairwise training cho RankNet
        Mỗi sample là một cặp (query, doc1, doc2, label)
        label = 1 nếu doc1 relevance > doc2, ngược lại = 0
        """
        pairwise_data = []
        
        for query in tqdm(queries, desc="Tạo dữ liệu pairwise"):
            # Preprocess query
            processed_query_tokens = self.preprocess(query)
            processed_query = " ".join(processed_query_tokens)
            
            # Tạo embeddings
            query_embedding = self.model.encode([processed_query])
            doc_texts = []
            for doc in documents:
                processed_doc_tokens = self.preprocess(doc['value'])
                processed_doc = " ".join(processed_doc_tokens)
                doc_texts.append(processed_doc)
            
            doc_embeddings = self.model.encode(doc_texts)
            
            # Tính cosine similarity
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Gán điểm relevance dựa trên similarity
            doc_relevances = []
            doc_similarities = [(i, sim) for i, sim in enumerate(similarities)]
            doc_similarities.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (doc_idx, similarity) in enumerate(doc_similarities):
                if rank < 3:  # Top 1-3
                    relevance = 3
                elif rank < 7:  # Top 4-7
                    relevance = 2
                elif rank < 12:  # Top 8-12
                    relevance = 1
                else:  # Còn lại
                    relevance = 0
                
                doc_relevances.append({
                    'doc_idx': doc_idx,
                    'relevance': relevance,
                    'similarity': similarity,
                    'document': documents[doc_idx]
                })
            
            # Tạo tất cả các cặp document có relevance khác nhau
            for i in range(len(doc_relevances)):
                for j in range(i + 1, len(doc_relevances)):
                    doc1 = doc_relevances[i]
                    doc2 = doc_relevances[j]
                    
                    # Gán nhãn theo thuật toán RankNet
                    if doc1['relevance'] > doc2['relevance']:
                        label = 1.0  # Doc1 tốt hơn Doc2
                    elif doc1['relevance'] < doc2['relevance']:
                        label = 0.0  # Doc1 kém hơn Doc2
                    else:
                        label = 0.5  # Doc1 và Doc2 tương đương (cùng relevance)
                    
                    pairwise_sample = {
                        "query": processed_query,
                        "doc1_id": doc1['document']['id'],
                        "doc1_text": doc_texts[doc1['doc_idx']],
                        "doc1_relevance": doc1['relevance'],
                        "doc1_similarity": float(doc1['similarity']),
                        "doc2_id": doc2['document']['id'],
                        "doc2_text": doc_texts[doc2['doc_idx']],
                        "doc2_relevance": doc2['relevance'],
                        "doc2_similarity": float(doc2['similarity']),
                        "label": label  # 1.0, 0.5, hoặc 0.0
                    }
                    pairwise_data.append(pairwise_sample)
        return pairwise_data

def train_ranknet(model, dataloader, optimizer, device, num_epochs=10):
    """Huấn luyện mô hình RankNet"""
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            feature1 = batch['feature1'].to(device)
            feature2 = batch['feature2'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            score1 = model(feature1)
            score2 = model(feature2)
            
            # Tính loss
            loss = ranknet_loss(score1.squeeze(), score2.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return model

def evaluate_ranking(model, documents, queries, sentence_model, device, top_k=10):
    """
    Đánh giá mô hình bằng cách rank lại documents cho mỗi query
    và tính pairwise accuracy
    """
    model.eval()
    model.to(device)
    
    retriever = DocumentRetriever(documents, sentence_model, "vietnamese-stopwords.txt")
    results = {}
    
    with torch.no_grad():
        for query in queries:
            # Preprocess query
            processed_query_tokens = retriever.preprocess(query)
            processed_query = " ".join(processed_query_tokens)
            
            # Encode query
            query_emb = sentence_model.encode([processed_query])[0]
            
            # Score tất cả documents
            doc_scores = []
            for i, doc in enumerate(documents):
                # Preprocess document
                processed_doc_tokens = retriever.preprocess(doc['value'])
                processed_doc = " ".join(processed_doc_tokens)
                
                # Encode document
                doc_emb = sentence_model.encode([processed_doc])[0]
                
                # Tạo feature vector
                feature = torch.tensor(np.concatenate([query_emb, doc_emb]), 
                                     dtype=torch.float32).unsqueeze(0).to(device)
                
                # Tính score
                score = model(feature).item()
                doc_scores.append((i, score, doc))
            
            # Sắp xếp theo score giảm dần
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Lấy top k
            top_docs = doc_scores[:top_k]
            
            results[query] = {
                'top_documents': [(doc['id'], doc['value'], score) for _, score, doc in top_docs],
                'all_scores': doc_scores
            }
    
    return results

def calculate_pairwise_accuracy(model, test_data, sentence_model, device):
    """Tính Pairwise Accuracy"""
    model.eval()
    model.to(device)
    
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for sample in tqdm(test_data, desc="Tính Pairwise Accuracy"):
            # Encode features
            query_emb = sentence_model.encode([sample['query']])[0]
            doc1_emb = sentence_model.encode([sample['doc1_text']])[0]
            doc2_emb = sentence_model.encode([sample['doc2_text']])[0]
            
            feature1 = torch.tensor(np.concatenate([query_emb, doc1_emb]), 
                                  dtype=torch.float32).unsqueeze(0).to(device)
            feature2 = torch.tensor(np.concatenate([query_emb, doc2_emb]), 
                                  dtype=torch.float32).unsqueeze(0).to(device)
            
            # Tính scores
            score1 = model(feature1).item()
            score2 = model(feature2).item()
            
            # Dự đoán
            if sample['label'] == 1.0:  # doc1 should be ranked higher
                if score1 > score2:
                    correct_predictions += 1
            elif sample['label'] == 0.0:  # doc2 should be ranked higher
                if score2 > score1:
                    correct_predictions += 1
            else:  # Equal relevance (label = 0.5)
                # Coi như đúng nếu difference nhỏ
                if abs(score1 - score2) < 0.1:
                    correct_predictions += 1
            
            total_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

if __name__ == "__main__":
    try:
        # Thiết lập device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Sử dụng device: {device}")
        
        # Bước 1: Tải tài liệu và khởi tạo sentence-transformer
        documents_path = "FetchData/output/cadao_tucngu_medium.json"
        # documents_path = "FetchData/train/anh_em_mot_nha/data.json"
        
        print("Đang tải dữ liệu...")
        with open(documents_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        print(f"Số tài liệu: {len(documents)}")

        # Khởi tạo sentence transformer
        print("Đang khởi tạo sentence-transformer...")
        sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2') # 384 chiều
        
        # Khởi tạo DocumentRetriever với stopwords
        stopword_path = "vietnamese-stopwords.txt"
        retriever = DocumentRetriever(documents, sentence_model, stopword_path)

        # Tạo cặp dữ liệu huấn luyện - Cải thiện với multiple queries
        queries = [
            "Nhớ ơn công lao cha mẹ",
            "tấm gương hiếu thảo",
            "bài học cuộc sống",
            "câu nói hay về tình bạn",
            "lòng biết ơn",
            "tình cảm gia đình",
            "đạo lý làm người",
            "giá trị truyền thống"
        ]

        print("Đang tạo dữ liệu pairwise training...")
        pairwise_data = retriever.generate_pairwise_training_data(queries, documents)
        print(f"Số cặp dữ liệu training: {len(pairwise_data)}")
        
        # Lưu dữ liệu pairwise vào file để tái sử dụng
        with open("input.json", 'w', encoding='utf-8') as f:
            json.dump(pairwise_data, f, ensure_ascii=False, indent=2)
        print("Đã lưu dữ liệu pairwise vào input.json")

        # Tạo dataset và dataloader
        print("Đang chuẩn bị dataset...")
        dataset = RankNetDataset(pairwise_data, sentence_model)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        
        # Bước 2: Khởi tạo và huấn luyện mô hình RankNet
        print("Đang khởi tạo mô hình RankNet...")
        
        # Tính kích thước input (query embedding + document embedding)
        sample_query_emb = sentence_model.encode(["test"])[0]
        input_size = len(sample_query_emb) * 2  # query + document embeddings
        print(f"Input size: {input_size}")
        
        model = RankNet(input_size=input_size, hidden_size1=256, hidden_size2=128, dropout=0.2)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        print("Bắt đầu huấn luyện mô hình...")
        model = train_ranknet(model, dataloader, optimizer, device, num_epochs=40)
        
        # Lưu mô hình
        torch.save(model.state_dict(), 'ranknet_model.pth')
        print("Đã lưu mô hình vào ranknet_model.pth")

        # Bước 3: Đánh giá độ chính xác của mô hình
        print("\n" + "="*50)
        print("ĐÁNH GIÁ MÔ HÌNH")
        print("="*50)
        
        # Test queries
        test_queries = [
            "gia đình hạnh phúc",
            "tình yêu quê hương",
            "phẩm chất tốt đẹp",
            "câu nói ý nghĩa"
        ]
        
        print("Đang đánh giá ranking...")
        ranking_results = evaluate_ranking(model, documents, test_queries, sentence_model, device, top_k=10)
        
        # Hiển thị kết quả top 10 cho mỗi query
        for query, result in ranking_results.items():
            print(f"\nQuery: '{query}'")
            print("Top 10 documents:")
            for i, (doc_id, doc_text, score) in enumerate(result['top_documents'], 1):
                print(f"{i:2d}. [ID: {doc_id}] Score: {score:.4f}")
                print(f"    Text: {doc_text[:100]}...")
                print()
        
        # Tính Pairwise Accuracy trên tập test
        print("Đang tính Pairwise Accuracy...")
        
        # Tạo test data từ test queries
        test_pairwise_data = retriever.generate_pairwise_training_data(test_queries, documents[:50])  # Giới hạn để tính nhanh
        accuracy = calculate_pairwise_accuracy(model, test_pairwise_data, sentence_model, device)
        
        print(f"\nPairwise Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\n" + "="*50)
        print("HOÀN THÀNH!")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()