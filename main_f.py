import json
import math
import re
from collections import Counter, defaultdict
import os
import numpy as np
from itertools import combinations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pyvi import ViTokenizer

class TFIDF:
    def __init__(self, documents, stopword_path=None):
        """
        Khởi tạo đối tượng TFIDF với tập văn bản dạng chuỗi, tự động lọc stopword và tách từ.
        Args:
            documents (List[str]): Danh sách các câu (chuỗi).
            stopword_path (str): Đường dẫn file stopword (mỗi dòng 1 từ/cụm từ).
        """
        self.stopwords = set()
        if stopword_path and os.path.exists(stopword_path):
            try:
                with open(stopword_path, encoding="utf-8") as f:
                    self.stopwords = set(line.strip() for line in f if line.strip())
            except Exception as e:
                print(f"Warning: Could not load stopwords from {stopword_path}: {e}")
        
        # Tiền xử lý: tách từ và lọc stopword
        self.documents = [self._preprocess(doc) for doc in documents if doc.strip()]
        self.N = len(self.documents)
        if self.N == 0:
            raise ValueError("No valid documents found after preprocessing")
        self.idf = self._compute_idf()

    def _preprocess(self, text):
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

    def _compute_idf(self):
        """
        Tính toán giá trị IDF (Inverse Document Frequency) cho từng từ trong tập văn bản.
        Returns:
            dict: Từ điển chứa idf của từng từ.
        """
        idf = defaultdict(lambda: 0)
        for doc in self.documents:
            unique_terms = set(doc)
            for term in unique_terms:
                idf[term] += 1
        for term in idf:
            idf[term] = math.log((self.N + 1) / (idf[term] + 1)) + 1
        return dict(idf)

    def compute_tf(self, doc):
        """
        Tính toán giá trị TF (Term Frequency) cho một văn bản.
        Args:
            doc (List[str]): Văn bản cần tính TF, là list các từ đã tách.
        Returns:
            dict: Từ điển chứa tf của từng từ trong văn bản.
        """
        if not doc:
            return {}
        tf = Counter(doc)
        total_terms = len(doc)
        return {term: count / total_terms for term, count in tf.items()}

    def compute_tfidf(self, doc):
        """
        Tính toán giá trị TF-IDF cho một văn bản.
        Args:
            doc (List[str]): Văn bản cần tính TF-IDF, là list các từ đã tách.
        Returns:
            dict: Từ điển chứa tf-idf của từng từ trong văn bản.
        """
        tf = self.compute_tf(doc)
        tfidf = {}
        for term, tf_val in tf.items():
            idf_val = self.idf.get(term, math.log((self.N + 1) / 1) + 1)
            tfidf[term] = tf_val * idf_val
        return tfidf

    def get_feature_names(self):
        """
        Lấy danh sách các từ (feature) đã xuất hiện trong tập văn bản huấn luyện.
        Returns:
            List[str]: Danh sách các từ (feature names).
        """
        return list(self.idf.keys())

    def transform(self, doc):
        """
        Chuyển một văn bản thành vector TF-IDF theo thứ tự các từ trong get_feature_names().
        Args:
            doc (str): Văn bản cần chuyển đổi, là chuỗi.
        Returns:
            List[float]: Vector TF-IDF tương ứng với văn bản.
        """
        tokens = self._preprocess(doc)
        tfidf = self.compute_tfidf(tokens)
        features = self.get_feature_names()
        return [tfidf.get(term, 0.0) for term in features]

    def fit_transform(self):
        """
        Trả về ma trận TF-IDF cho toàn bộ tập văn bản ban đầu (mỗi dòng là vector của 1 văn bản).
        Returns:
            List[List[float]]: Ma trận TF-IDF (số văn bản x số feature).
        """
        features = self.get_feature_names()
        matrix = []
        for doc in self.documents:
            tfidf = self.compute_tfidf(doc)
            row = [tfidf.get(term, 0.0) for term in features]
            matrix.append(row)
        return matrix
    
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

def load_documents_from_json(documents_path):
    try:
        with open(documents_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        documents = [item['value'] for item in data if 'value' in item and item['value'].strip()]
        if not documents:
            raise ValueError("No valid documents found in JSON file")
        return documents
    except Exception as e:
        print(f"Error loading documents from {documents_path}: {e}")
        raise

def relevance_score_tfidf(tfidf_matrix, vocab, query, doc_idx, tfidf_model=None):
    """
    Tính điểm liên quan của tài liệu dựa trên query.
    Cải thiện: sử dụng cosine similarity giữa query vector và document vector.
    
    Args:
        tfidf_matrix: Ma trận TF-IDF
        vocab: Từ điển các từ
        query: Chuỗi truy vấn (có thể có nhiều từ)
        doc_idx: Index của tài liệu
        tfidf_model: Đối tượng TFIDF để transform query
    
    Returns:
        float: Điểm cosine similarity giữa query và document
    """
    if tfidf_model:
        # Sử dụng TF-IDF model để transform query
        query_vector = tfidf_model.transform(query)
    else:
        # Fallback về phương pháp cũ
        query = query.lower().strip()
        query_words = query.split()
        
        # Tạo query vector
        query_vector = [0.0] * len(vocab)
        for word in query_words:
            word = word.strip()
            if word and word in vocab:
                word_idx = vocab.index(word)
                query_vector[word_idx] = 1.0  # Binary weight
    
    # Lấy document vector
    doc_vector = tfidf_matrix[doc_idx]
    
    # Tính cosine similarity
    query_norm = np.linalg.norm(query_vector)
    doc_norm = np.linalg.norm(doc_vector)
    
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    
    cosine_sim = np.dot(query_vector, doc_vector) / (query_norm * doc_norm)
    return cosine_sim

def calculate_precision_at_k(true_relevance_scores, predicted_ranking, k=10, threshold=0.1):
    """
    Tính Precision@K cho kết quả ranking.
    
    Args:
        true_relevance_scores (list): Điểm liên quan thực tế của các documents (theo thứ tự gốc)
        predicted_ranking (list): Danh sách (score, doc_index) được sắp xếp theo score giảm dần
        k (int): Số lượng documents top-k cần đánh giá
        threshold (float): Ngưỡng để xác định document có relevant hay không
    
    Returns:
        float: Precision@K (từ 0 đến 1)
    """
    if k <= 0 or len(predicted_ranking) == 0:
        return 0.0
    
    # Lấy top-k documents từ kết quả ranking
    top_k_docs = predicted_ranking[:k]
    
    # Đếm số documents relevant trong top-k
    relevant_in_topk = 0
    for score, doc_idx in top_k_docs:
        if doc_idx < len(true_relevance_scores):
            if true_relevance_scores[doc_idx] >= threshold:
                relevant_in_topk += 1
    
    precision_at_k = relevant_in_topk / k
    return precision_at_k

def calculate_recall_at_k(true_relevance_scores, predicted_ranking, k=10, threshold=0.1):
    """
    Tính Recall@K cho kết quả ranking.
    
    Args:
        true_relevance_scores (list): Điểm liên quan thực tế của các documents (theo thứ tự gốc)
        predicted_ranking (list): Danh sách (score, doc_index) được sắp xếp theo score giảm dần
        k (int): Số lượng documents top-k cần đánh giá
        threshold (float): Ngưỡng để xác định document có relevant hay không
    
    Returns:
        float: Recall@K (từ 0 đến 1)
    """
    if k <= 0 or len(predicted_ranking) == 0:
        return 0.0
    
    # Đếm tổng số documents relevant trong toàn bộ tập dữ liệu
    total_relevant = sum(1 for score in true_relevance_scores if score >= threshold)
    
    if total_relevant == 0:
        return 0.0  # Không có document nào relevant
    
    # Lấy top-k documents từ kết quả ranking
    top_k_docs = predicted_ranking[:k]
    
    # Đếm số documents relevant trong top-k
    relevant_in_topk = 0
    for score, doc_idx in top_k_docs:
        if doc_idx < len(true_relevance_scores):
            if true_relevance_scores[doc_idx] >= threshold:
                relevant_in_topk += 1
    
    recall_at_k = relevant_in_topk / total_relevant
    return recall_at_k

def calculate_f1_score_at_k(true_relevance_scores, predicted_ranking, k=10, threshold=0.1):
    """
    Tính F1-Score@K cho kết quả ranking.
    F1-Score là trung bình điều hòa của Precision và Recall.
    
    Args:
        true_relevance_scores (list): Điểm liên quan thực tế của các documents (theo thứ tự gốc)
        predicted_ranking (list): Danh sách (score, doc_index) được sắp xếp theo score giảm dần
        k (int): Số lượng documents top-k cần đánh giá
        threshold (float): Ngưỡng để xác định document có relevant hay không
    
    Returns:
        float: F1-Score@K (từ 0 đến 1)
    """
    precision_k = calculate_precision_at_k(true_relevance_scores, predicted_ranking, k, threshold)
    recall_k = calculate_recall_at_k(true_relevance_scores, predicted_ranking, k, threshold)
    
    # Tính F1-Score = 2 * (precision * recall) / (precision + recall)
    if (precision_k + recall_k) == 0:
        return 0.0
    
    f1_score = 2 * (precision_k * recall_k) / (precision_k + recall_k)
    return f1_score

def calculate_average_precision(true_relevance_scores, predicted_ranking, threshold=0.1):
    """
    Tính Average Precision (AP) cho kết quả ranking.
    AP tính trung bình của Precision@k cho tất cả các vị trí k mà có document relevant.
    
    Args:
        true_relevance_scores (list): Điểm liên quan thực tế của các documents (theo thứ tự gốc)
        predicted_ranking (list): Danh sách (score, doc_index) được sắp xếp theo score giảm dần
        threshold (float): Ngưỡng để xác định document có relevant hay không
    
    Returns:
        float: Average Precision (từ 0 đến 1)
    """
    if len(predicted_ranking) == 0:
        return 0.0
    
    # Đếm tổng số documents relevant
    total_relevant = sum(1 for score in true_relevance_scores if score >= threshold)
    if total_relevant == 0:
        return 0.0
    
    precision_sum = 0.0
    relevant_found = 0
    
    for i, (score, doc_idx) in enumerate(predicted_ranking):
        if doc_idx < len(true_relevance_scores):
            if true_relevance_scores[doc_idx] >= threshold:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i
    
    average_precision = precision_sum / total_relevant if total_relevant > 0 else 0.0
    return average_precision

if __name__ == "__main__":
    try:
        # Bước 1: Tải tài liệu và biểu diễn TF-IDF
        documents_path = "FetchData/output/cadao_tucngu_medium.json"
        # documents_path = "FetchData/train/anh_em_mot_nha/data.json"
        documents = load_documents_from_json(documents_path)
        print("Số tài liệu:", len(documents))

        stopword_path = "../../vietnamese-stopwords.txt"
        tfidf_model = TFIDF(documents, stopword_path=stopword_path)
        matrix = tfidf_model.fit_transform()
        vocab = tfidf_model.get_feature_names()
        print("Số từ trong từ điển:", len(vocab))
        print("Kích thước ma trận TF-IDF:", len(matrix))

        # Bước 2: Tạo cặp dữ liệu huấn luyện - Cải thiện với multiple queries
        queries = ["nhớ về quê", "quê hương", "về quê", "nhớ nhà", "quê ngoại", "quê nội"]
        
        # Tính điểm liên quan tổng hợp từ nhiều query
        all_scores = []
        for query in queries:
            query_scores = [relevance_score_tfidf(matrix, vocab, query, i, tfidf_model) for i in range(len(documents))]
            all_scores.append(query_scores)
            print(f"Điểm liên quan của từ '{query}': min={min(query_scores):.4f}, max={max(query_scores):.4f}")
        
        # Tính điểm trung bình từ các query
        scores = [sum(score_list[i] for score_list in all_scores) / len(queries) 
                 for i in range(len(documents))]
        print(f"Điểm liên quan tổng hợp: min={min(scores):.4f}, max={max(scores):.4f}")

        pairs = []
        # Tăng threshold để tạo ra các cặp có sự khác biệt rõ ràng hơn
        threshold = 0.001
        for i, j in combinations(range(len(documents)), 2):
            if abs(scores[i] - scores[j]) < threshold:  # Chỉ lấy cặp có sự khác biệt rõ ràng
                continue
            dij = (matrix[i], matrix[j])  # Cặp vector TF-IDF
            pij = 1.0 if scores[i] > scores[j] else 0.0
            pairs.append((dij, pij))
        print(f"Số cặp tài liệu: {len(pairs)}")

        if len(pairs) == 0:
            print("Warning: No training pairs generated. Check your data and query.")
            exit(1)

        # Split training/validation
        split_idx = int(0.8 * len(pairs))
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]
        
        print(f"Training pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}")

        # Bước 3: Huấn luyện RankNet - Cải thiện hyperparameters
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Cải thiện model architecture
        model = RankNet(input_size=len(vocab), hidden_size1=256, hidden_size2=128, dropout=0.3).to(device)
        
        # Sử dụng learning rate scheduling
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.001)
        
        train_dataset = RankNetDataset(train_pairs, matrix)
        val_dataset = RankNetDataset(val_pairs, matrix)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        epochs = 500  # Tăng số epoch
        best_val_loss = float('inf')
        patience = 50  # Tăng patience
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            total_loss = 0
            for di_vec, dj_vec, pij in train_dataloader:
                di_vec, dj_vec, pij = di_vec.to(device), dj_vec.to(device), pij.to(device)
                
                optimizer.zero_grad()
                si = model(di_vec)  # Điểm số cho dᵢ
                sj = model(dj_vec)  # Điểm số cho dⱼ
                loss = ranknet_loss(si, sj, pij)
                loss.backward()
                
                # Gradient clipping để tránh exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_dataloader)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for di_vec, dj_vec, pij in val_dataloader:
                    di_vec, dj_vec, pij = di_vec.to(device), dj_vec.to(device), pij.to(device)
                    si = model(di_vec)
                    sj = model(dj_vec)
                    loss = ranknet_loss(si, sj, pij)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else float('inf')
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:  # In thông tin mỗi 20 epoch
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
            
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
                input_tensor = torch.tensor(matrix[i], dtype=torch.float32).to(device)
                score = model(input_tensor.unsqueeze(0)).item()  # Thêm batch dimension
                scores_rank.append((score, i))  # Lưu index để tính evaluation metrics

        ranked = sorted(scores_rank, key=lambda x: x[0], reverse=True)
        
        # Tính toán evaluation metrics
        print(f"\n📊 ĐÁNH GIÁ HIỆU SUẤT RANKING:")
        print("=" * 80)
        
        # Tính Precision@K và Recall@K cho các giá trị K khác nhau
        k_values = [5, 10, 15, 20]
        threshold = 0.05  # Ngưỡng để xác định document relevant
        
        print(f"Threshold để xác định relevant document: {threshold}")
        print(f"Số documents có điểm ≥ {threshold}: {sum(1 for s in scores if s >= threshold)}")
        print()
        
        for k in k_values:
            precision_k = calculate_precision_at_k(scores, ranked, k, threshold)
            recall_k = calculate_recall_at_k(scores, ranked, k, threshold)
            f1_k = calculate_f1_score_at_k(scores, ranked, k, threshold)
            
            print(f"📈 Top-{k:2d} Results:")
            print(f"   Precision@{k}: {precision_k:.4f}")
            print(f"   Recall@{k}:    {recall_k:.4f}")
            print(f"   F1-Score@{k}:  {f1_k:.4f}")
            print()
        
        # Tính Average Precision (AP) - một metric quan trọng khác
        avg_precision = calculate_average_precision(scores, ranked, threshold)
        print(f"📊 ADDITIONAL METRICS:")
        print(f"   Average Precision (AP): {avg_precision:.4f}")
        print()
        
        # Hiển thị kết quả ranking
        print(f"🏆 KẾT QUẢ XẾP HẠNG cho queries: {', '.join(queries)}")
        print("=" * 80)
        for i, (score, doc_idx) in enumerate(ranked[:15]):  # Hiển thị top 15
            relevance_mark = "⭐" if scores[doc_idx] >= threshold else "  "
            print(f"{i+1:2d}. {score:.4f} {relevance_mark} - {documents[doc_idx][:120]}...")
            
        # Hiển thị thống kê scores
        scores_only = [score for score, _ in scores_rank]
        print(f"\n📊 THỐNG KÊ SCORES:")
        print(f"Min: {min(scores_only):.4f}, Max: {max(scores_only):.4f}")
        print(f"Mean: {np.mean(scores_only):.4f}, Std: {np.std(scores_only):.4f}")
        print(f"Relevant docs (score ≥ {threshold}): {sum(1 for s in scores if s >= threshold)}/{len(scores)}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()