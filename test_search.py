"""
Script test để kiểm tra logic tìm kiếm
Chạy để debug và so sánh với main.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Library.TF_IDF.TF_IDF import TFIDF
from helper import load_documents_from_json, relevance_score_tfidf
from Library.rank_net import RankNet
import torch

def test_search_logic():
    print("=== KIỂM TRA LOGIC TÌM KIẾM ===")
    
    # Test với file mini giống main.py
    documents_path = "FetchData/output/cadao_tucngu_mini.json"
    
    if not os.path.exists(documents_path):
        print(f"❌ Không tìm thấy file: {documents_path}")
        return
    
    print(f"✅ Tìm thấy file: {documents_path}")
    
    # Tải dữ liệu
    documents = load_documents_from_json(documents_path)
    print(f"📊 Số tài liệu: {len(documents)}")
    
    # In ra một vài tài liệu mẫu
    print("\n📝 Một vài tài liệu mẫu:")
    for i, doc in enumerate(documents[:5]):
        print(f"  {i+1}. {doc}")
    
    # Khởi tạo TF-IDF
    stopword_path = "vietnamese-stopwords.txt"
    tfidf_model = TFIDF(documents, stopword_path=stopword_path)
    matrix = tfidf_model.fit_transform()
    vocab = tfidf_model.get_feature_names()
    
    print(f"📖 Số từ trong từ điển: {len(vocab)}")
    print(f"📏 Kích thước ma trận TF-IDF: {len(matrix)}")
    
    # Test với từ khóa "học" thay vì "ăn"
    query = "học"
    print(f"\n🔍 Test với từ khóa: '{query}'")
    
    # Kiểm tra từ có trong vocab không
    if query.lower() in vocab:
        query_idx = vocab.index(query.lower())
        print(f"✅ Từ '{query}' có trong từ điển tại vị trí: {query_idx}")
    else:
        print(f"❌ Từ '{query}' KHÔNG có trong từ điển!")
        
        # Thử từ khác
        test_words = ["người", "nhà", "học", "làm", "quả", "cây"]
        print(f"🔤 Test các từ khác:")
        for word in test_words:
            if word in vocab:
                print(f"  ✅ '{word}' có trong vocab")
                query = word  # Sử dụng từ này để test
                break
            else:
                print(f"  ❌ '{word}' không có trong vocab")
        
        if query not in vocab:
            print(f"🔤 Một vài từ trong vocab: {vocab[:20]}")
            return
    
    # Tính điểm TF-IDF
    scores = [relevance_score_tfidf(matrix, vocab, query, i) for i in range(len(documents))]
    print(f"📊 Điểm TF-IDF cho từ '{query}': {scores}")
    
    # Tìm tài liệu có điểm cao nhất
    max_score = max(scores)
    max_idx = scores.index(max_score)
    
    print(f"🏆 Điểm cao nhất: {max_score} tại tài liệu {max_idx}")
    print(f"📄 Tài liệu có điểm cao nhất: {documents[max_idx]}")
    
    # Kiểm tra những tài liệu có chứa từ "ăn"
    print(f"\n📋 Tài liệu có chứa từ '{query}':")
    for i, doc in enumerate(documents):
        if query.lower() in doc.lower():
            print(f"  {i+1}. (Điểm: {scores[i]:.4f}) {doc}")
    
    print("\n=== HOÀN THÀNH KIỂM TRA ===")

if __name__ == "__main__":
    test_search_logic()
