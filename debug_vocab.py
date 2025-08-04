"""
Debug vocab để tìm từ khóa phù hợp
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Library.TF_IDF import TFIDF
from helper import load_documents_from_json, relevance_score_tfidf

def debug_vocab():
    print("=== DEBUG VOCAB ===")
    
    # Tải dữ liệu
    documents_path = "FetchData/output/cadao_tucngu_mini.json"
    documents = load_documents_from_json(documents_path)
    print(f"📊 Số tài liệu: {len(documents)}")
    
    print("\n📄 Danh sách tài liệu:")
    for i, doc in enumerate(documents):
        print(f"  {i+1}. {doc}")
    
    # TF-IDF
    stopword_path = "vietnamese-stopwords.txt"
    tfidf_model = TFIDF(documents, stopword_path=stopword_path)
    matrix = tfidf_model.fit_transform()
    vocab = tfidf_model.get_feature_names()
    
    print(f"\n📖 Từ điển ({len(vocab)} từ):")
    print(f"   {vocab}")
    
    # Kiểm tra từ "anh" có trong stopwords không
    print(f"\n🔍 Kiểm tra các từ cụ thể:")
    test_words = ["anh", "em", "quả", "cây", "nhà", "người", "làm", "học"]
    
    for word in test_words:
        if word in vocab:
            print(f"   ✅ '{word}' có trong vocab")
        else:
            print(f"   ❌ '{word}' KHÔNG có trong vocab")
    
    # Tìm từ nào có trong vocab để test
    print(f"\n🔎 Test với từ có trong vocab:")
    for word in vocab[:5]:  # Test 5 từ đầu tiên
        print(f"\n   Test từ: '{word}'")
        scores = [relevance_score_tfidf(matrix, vocab, word, i) for i in range(len(documents))]
        max_score = max(scores) if scores else 0
        non_zero = sum(1 for s in scores if s > 0)
        
        print(f"      📈 Điểm cao nhất: {max_score:.4f}")
        print(f"      📊 Số tài liệu có điểm > 0: {non_zero}")
        
        if max_score > 0:
            best_idx = scores.index(max_score)
            print(f"      🏆 Tài liệu tốt nhất: {documents[best_idx]}")

if __name__ == "__main__":
    debug_vocab()
