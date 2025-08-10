from flask import Flask, render_template, request, jsonify
import json
import time
import os
import torch
import numpy as np
from Library.TF_IDF import TFIDF  # Trở lại import gốc
from helper import load_documents_from_json, relevance_score_tfidf
from Library.rank_net import RankNet

app = Flask(__name__)

# Global variables để lưu model và dữ liệu
model = None
documents = []
tfidf_model = None
matrix = []
vocab = []

def initialize_system():
    """Khởi tạo hệ thống tìm kiếm"""
    global model, documents, tfidf_model, matrix, vocab
    
    print("Đang khởi tạo hệ thống...")
    
    try:
        # Tải tài liệu - dùng medium để có đủ vocab
        documents_path = "FetchData/output/cadao_tucngu_medium.json"  # Đổi sang medium
        if not os.path.exists(documents_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {documents_path}")
        
        documents = load_documents_from_json(documents_path)
        print(f"Đã tải {len(documents)} tài liệu")
        
        # Khởi tạo TF-IDF - TẮT stopwords để test
        stopword_path = None  # TẮT stopwords tạm thời
        print(f"⚠️  TẮT stopwords để test (như test_multi_word_query.py có thể đang làm)")
        
        tfidf_model = TFIDF(documents, stopword_path=stopword_path)
        matrix = tfidf_model.fit_transform()
        vocab = tfidf_model.get_feature_names()
        print(f"Đã tạo ma trận TF-IDF với {len(vocab)} từ")
        
        # Debug: kiểm tra vocab có từ quan trọng không
        important_words = ["anh", "em", "quả", "cây", "người"]
        print("🔍 Kiểm tra từ quan trọng trong vocab:")
        for word in important_words:
            if word in vocab:
                print(f"   ✅ '{word}' có trong vocab")
            else:
                print(f"   ❌ '{word}' KHÔNG có trong vocab")
        
        print(f"🔍 Vocab sample: {vocab[:20]}")  # In 20 từ đầu
        
        # Khởi tạo model RankNet
        model = RankNet(input_size=len(vocab))
        model.eval()  # Chế độ đánh giá
        print("Hệ thống đã sẵn sàng!")
        return True
        
    except Exception as e:
        print(f"Lỗi khởi tạo hệ thống: {str(e)}")
        return False

def train_model_for_query(query):
    """Huấn luyện model cho một query cụ thể"""
    global model, documents, matrix, vocab
    
    from itertools import combinations
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from main import RankNetDataset
    from helper import ranknet_loss
    
    print(f"Bắt đầu huấn luyện model cho query: '{query}'")
    
    # Tính điểm liên quan
    scores = [relevance_score_tfidf(matrix, vocab, query, i) for i in range(len(documents))]
    print(f"Điểm TF-IDF cho từ '{query}': {scores}")
    
    # Tạo cặp dữ liệu huấn luyện
    pairs = []
    for i, j in combinations(range(len(documents)), 2):
        if scores[i] == scores[j]:
            continue
        dij = (matrix[i], matrix[j])
        pij = 1 if scores[i] > scores[j] else 0
        pairs.append((dij, pij))
    
    print(f"Số cặp tài liệu được tạo: {len(pairs)}")
    
    if len(pairs) == 0:
        print("Không có cặp dữ liệu để huấn luyện!")
        return False
    
    # Huấn luyện nhanh
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = RankNetDataset(pairs, matrix)
    dataloader = DataLoader(dataset, batch_size=min(16, len(pairs)), shuffle=True)
    
    print(f"Bắt đầu huấn luyện với {len(pairs)} cặp dữ liệu...")
    
    model.train()
    for epoch in range(200):  # Sử dụng 100 epochs như main.py
        total_loss = 0
        for di_vec, dj_vec, pij in dataloader:
            optimizer.zero_grad()
            si = model(di_vec)
            sj = model(dj_vec)
            loss = ranknet_loss(si, sj, pij)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0:  # In log mỗi 20 epochs
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")
    
    model.eval()
    print("Huấn luyện hoàn tất!")
    return True

def search_documents(query):
    """Tìm kiếm và xếp hạng tài liệu - Logic chính xác như main.py"""
    global model, documents, matrix, vocab
    
    start_time = time.time()
    print(f"Tìm kiếm cho từ khóa: '{query}'")
    
    # Tính điểm liên quan TF-IDF trước khi huấn luyện (để debug)
    scores = [relevance_score_tfidf(matrix, vocab, query.lower(), i) for i in range(len(documents))]
    print(f"Điểm liên quan TF-IDF: {scores}")  # In tất cả điểm để debug
    
    # Huấn luyện model cho query này
    training_success = train_model_for_query(query.lower())
    
    if not training_success:
        print("Fallback về TF-IDF vì không có dữ liệu huấn luyện")
        # Fallback về TF-IDF nếu không có dữ liệu huấn luyện
        results = [(scores[i], documents[i]) for i in range(len(documents)) if scores[i] > 0]
    else:
        print("Sử dụng RankNet để tính điểm")
        # Bước 4: Dự đoán và xếp hạng - CHÍNH XÁC như main.py
        model.eval()
        scores_rank = []
        with torch.no_grad():
            for i in range(len(documents)):
                score = model(torch.tensor(matrix[i], dtype=torch.float32)).item()
                scores_rank.append((score, documents[i]))
        
        # Trả về TẤT CẢ kết quả như main.py (không lọc)
        results = scores_rank
    
    # Sắp xếp theo điểm số giảm dần - như main.py
    results.sort(key=lambda x: x[0], reverse=True)
    
    # Giới hạn kết quả để hiển thị web (top 10 thay vì 20)
    results = results[:10]
    
    end_time = time.time()
    search_time = round(end_time - start_time, 3)
    
    print(f"Tìm thấy {len(results)} kết quả trong {search_time}s")
    print("Kết quả xếp hạng:")
    for score, doc in results[:5]:  # In 5 kết quả đầu
        print(f"  {score:.4f} - {doc}")
    
    return results, search_time

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """API tìm kiếm"""
    try:
        print("Nhận yêu cầu tìm kiếm...")
        print("Dữ liệu yêu cầu:", request.get_data(as_text=True))
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query không được để trống'}), 400
        
        results, search_time = search_documents(query)
        
        # Format kết quả
        formatted_results = [
            {
                'score': score,
                'content': content
            }
            for score, content in results
        ]
        
        return jsonify({
            'results': formatted_results,
            'total_results': len(formatted_results),
            'total_time': search_time,
            'query': query
        })
        
    except Exception as e:
        print(f"Lỗi tìm kiếm: {str(e)}")
        return jsonify({'error': 'Có lỗi xảy ra trong quá trình tìm kiếm'}), 500

@app.route('/health')
def health():
    """Kiểm tra trạng thái hệ thống"""
    return jsonify({
        'status': 'healthy',
        'documents_loaded': len(documents),
        'vocab_size': len(vocab) if vocab else 0
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Khởi động server Flask...")
    print("📍 Truy cập: http://localhost:5000")
    print("="*50 + "\n")
    
    # Khởi tạo hệ thống
    if initialize_system():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Không thể khởi tạo hệ thống!")
        print("💡 Thử chạy app_simple.py để test giao diện:")
        print("   python app_simple.py")
        input("Nhấn Enter để thoát...")
