import json

import torch

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
        return tfidf_matrix[doc_idx][query_idx]  # Giá trị TF-IDF của từ được truy vấn trong tài liệu
    return 0

def ranknet_loss(s_i, s_j, P_ij):
    diff = s_i - s_j
    P_hat = torch.sigmoid(diff)  # Xác suất dự đoán P̂ᵢⱼ
    loss = -P_ij * torch.log(P_hat + 1e-10) - (1 - P_ij) * torch.log(1 - P_hat + 1e-10)
    return loss.mean()

def export_traning_pairs(pairs, query, output_path = "training_pairs.txt"):
    """
    Xuất các cặp tài liệu và nhãn vào file JSON.
    Args:
        pairs (List[Tuple]): Danh sách các cặp tài liệu và nhãn.
        query (str): Truy vấn dùng để tạo cặp dữ liệu.
        output_path (str): Đường dẫn file xuất ra.
    """
    data = {
        "query": query,
        "pairs": [
            {
                "pair_id": idx,
                "document_i": di,
                "document_j": dj,
                "pij": pij
            }
            for idx, ((di, dj), pij) in enumerate(pairs, 1)
        ]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)