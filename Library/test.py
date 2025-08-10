# =========================
# Ví dụ cách sử dụng TFIDF
# =========================
from TF_IDF import TFIDF


if __name__ == "__main__":
    # Danh sách các văn bản (chuỗi)
    documents = [
        "con mèo đen ngủ hả",
        "con chó trắng chạy ạ",
        "con mèo trắng chạy"
    ]

    # Đường dẫn file stopword
    stopword_path = "../../vietnamese-stopwords.txt"

    # Khởi tạo đối tượng TFIDF với stopword
    tfidf_model = TFIDF(documents, stopword_path=stopword_path)

    X = tfidf_model.fit_transform()
    print("TF-IDF matrix shape:", X)
    # Lấy danh sách các từ (feature names)
    print("Feature names:", tfidf_model.get_feature_names())

    # Tính vector TF-IDF cho một văn bản mới (chuỗi)
    # new_doc = "con mèo trắng"
    # vector = tfidf_model.transform(new_doc)
    # print("TF-IDF vector:", vector)
