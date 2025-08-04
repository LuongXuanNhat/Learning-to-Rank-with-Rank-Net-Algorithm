"""
Test script để kiểm tra app.py sau khi sửa
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Library.TF_IDF import TFIDF  # Trở lại import gốc
from helper import load_documents_from_json, relevance_score_tfidf
from Library.rank_net import RankNet
import torch

def test_fixed_app_logic():
    print("=== TEST LOGIC APP.PY ĐÃ SỬA ===")
    
    # Tải dữ liệu giống app.py
    documents_path = "FetchData/output/cadao_tucngu_medium.json"  # Đổi sang medium
    documents = load_documents_from_json(documents_path)
    print(f"📊 Số tài liệu: {len(documents)}")
    
    # TF-IDF giống app.py
    stopword_path = "vietnamese-stopwords.txt"
    print(f"🔍 Stopword path: {stopword_path}")
    print(f"🔍 File tồn tại: {os.path.exists(stopword_path)}")
    
    # Thử cả hai path
    if not os.path.exists(stopword_path):
        stopword_path = "../../vietnamese-stopwords.txt"
        print(f"🔍 Thử path khác: {stopword_path}")
        print(f"🔍 File tồn tại: {os.path.exists(stopword_path)}")
    
    tfidf_model = TFIDF(documents, stopword_path=stopword_path)
    matrix = tfidf_model.fit_transform()
    vocab = tfidf_model.get_feature_names()
    print(f"📖 Từ điển: {len(vocab)} từ")
    print(f"📝 Một vài từ đầu tiên: {vocab[:10]}")
    
    # Test với các query
    test_queries = ["anh", "anh em", "quả cây"]
    
    for query in test_queries:
        print(f"\n🔍 Test query: '{query}'")
        
        # Test hàm relevance_score_tfidf đã sửa
        scores = [relevance_score_tfidf(matrix, vocab, query, i) for i in range(len(documents))]
        max_score = max(scores) if scores else 0
        non_zero = sum(1 for s in scores if s > 0)
        
        print(f"   📈 Điểm cao nhất: {max_score:.4f}")
        print(f"   📊 Số tài liệu có điểm > 0: {non_zero}")
        
        if max_score > 0:
            best_idx = scores.index(max_score)
            print(f"   🏆 Tài liệu tốt nhất: {documents[best_idx]}")
            
            # Hiển thị top 3 tài liệu
            doc_scores = [(scores[i], documents[i]) for i in range(len(documents))]
            doc_scores.sort(key=lambda x: x[0], reverse=True)
            
            print(f"   📋 Top 3 kết quả:")
            for i, (score, doc) in enumerate(doc_scores[:3]):
                if score > 0:
                    print(f"      {i+1}. ({score:.4f}) {doc}")

    print("\n=== HOÀN THÀNH TEST ===")

if __name__ == "__main__":
    test_fixed_app_logic()
