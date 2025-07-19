import json
import numpy as np
from itertools import combinations
from Library.TF_IDF.TF_IDF import TFIDF
from helper import load_documents_from_json, relevance_score_tfidf


if __name__ == "__main__":
    
    # 1: Text representation to vector format
    # Load documents from JSON file
    documents_path = "FetchData/output/cadao_tucngu_mini.json"
    documents = load_documents_from_json(documents_path)

    # Convert document to vector format by TF-IDF (has stopword)
    stopword_path = "../../vietnamese-stopwords.txt"
    tfidf_model = TFIDF(documents, stopword_path=stopword_path)
    matrix = tfidf_model.fit_transform()


    # 2: Create training data pair
    query = "ăn"
    vocab = tfidf_model.get_feature_names()

    # Tính điểm liên quan cho mỗi tài liệu
    scores = [relevance_score_tfidf(matrix, vocab, query, i) for i in range(len(documents))]

    # Tạo các cặp (dᵢ, dⱼ, Pᵢⱼ)
    pairs = []
    for i, j in combinations(range(len(documents)), 2):
        if scores[i] == scores[j]:
            continue  # Bỏ qua nếu điểm liên quan bằng nhau
        dij = (documents[i], documents[j])  # Cặp tài liệu (dᵢ, dⱼ)
        pij = 1 if scores[i] > scores[j] else 0  # Gán nhãn Pᵢⱼ
        pairs.append((dij, pij))

    # In ra một số cặp để kiểm tra
    for (di, dj), pij in pairs[:5]:  # In 5 cặp đầu tiên
        print(f"Cặp: ({di[:50]}..., {dj[:50]}...), Pᵢⱼ = {pij}")
    
    