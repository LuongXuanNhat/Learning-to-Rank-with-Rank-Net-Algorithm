import json

def load_documents_from_json(documents_path):
    """
    Đọc file json và trả về danh sách chuỗi từ trường 'value'.
    Args:
        documents_path (str): Đường dẫn file json.
    Returns:
        List[str]: Danh sách chuỗi văn bản.
    """
    with open(documents_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return [item['value'] for item in data if 'value' in item]

# Hàm tính điểm liên quan dựa trên TF-IDF của từ được truy vấn
def relevance_score_tfidf(tfidf_matrix, vocab, query, doc_idx):
    query = query.lower()
    if query in vocab:
        query_idx = vocab.index(query)  # Lấy chỉ số của từ được truy vấn trong từ điển

        print(f"Chỉ số của từ '{query}' trong từ điển: {query_idx}")
        return tfidf_matrix[doc_idx][query_idx]  # Giá trị TF-IDF của từ được truy vấn trong tài liệu
    return 0