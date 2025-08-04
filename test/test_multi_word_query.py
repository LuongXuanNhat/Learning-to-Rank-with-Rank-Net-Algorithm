# Tạo file test để kiểm tra xử lý query nhiều từ
# filepath: test_multi_word_query.py
import json
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Library.TF_IDF import TFIDF
from helper import load_documents_from_json, relevance_score_tfidf


def test_query_processing():
    """Test xử lý query đơn từ vs nhiều từ"""
    
    # Load dữ liệu
    documents_path = "../FetchData/output/cadao_tucngu_medium.json"
    documents = load_documents_from_json(documents_path)
    print(f"📊 Số tài liệu: {len(documents)}")
    
    # Tạo TF-IDF
    stopword_path = "../../vietnamese-stopwords.txt"
    tfidf_model = TFIDF(documents, stopword_path=stopword_path)
    matrix = tfidf_model.fit_transform()
    vocab = tfidf_model.get_feature_names()
    print(f"📖 Số từ trong từ điển: {len(vocab)}")
    
    # Test các query khác nhau
    test_queries = [
        "anh",           # 1 từ
        "anh em",        # 2 từ
        "quả cây",    # 3 từ
        "người",         # 1 từ khác
        "người xa"       # 2 từ khác
    ]
    
    for query in test_queries:
        print(f"\n🔍 Test query: '{query}'")
        
        # Kiểm tra từng từ có trong vocab không
        query_words = query.split()
        print(f"   Các từ trong query: {query_words}")
        
        for word in query_words:
            if word in vocab:
                print(f"   ✅ Từ '{word}' có trong vocab")
            else:
                print(f"   ❌ Từ '{word}' KHÔNG có trong vocab")
        
        # Tính điểm liên quan
        try:
            scores = [relevance_score_tfidf(matrix, vocab, query, i) for i in range(len(documents))]
            max_score = max(scores)
            non_zero_count = sum(1 for s in scores if s > 0)
            print(f"   📈 Điểm cao nhất: {max_score:.4f}")
            print(f"   📊 Số tài liệu có điểm > 0: {non_zero_count}")
            
            if max_score > 0:
                # Tìm tài liệu có điểm cao nhất
                best_idx = scores.index(max_score)
                print(f"   🏆 Tài liệu tốt nhất: {documents[best_idx]}")
            else:
                print(f"   ⚠️  Không có tài liệu nào phù hợp")
                
        except Exception as e:
            print(f"   💥 Lỗi: {str(e)}")

if __name__ == "__main__":
    test_query_processing()