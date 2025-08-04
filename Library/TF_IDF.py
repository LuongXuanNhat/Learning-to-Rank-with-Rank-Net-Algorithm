import math
from collections import Counter, defaultdict
import os


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
            with open(stopword_path, encoding="utf-8") as f:
                self.stopwords = set(line.strip() for line in f if line.strip())
        # Tiền xử lý: tách từ và lọc stopword
        self.documents = [self._preprocess(doc) for doc in documents]
        self.N = len(self.documents)
        self.idf = self._compute_idf()

    def _preprocess(self, text):
        """
        Tách từ đơn giản (theo khoảng trắng) và loại bỏ stopword.
        Args:
            text (str): Câu đầu vào.
        Returns:
            List[str]: Danh sách từ đã lọc stopword.
        """
        tokens = text.strip().lower().split()
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
        tf = Counter(doc)
        total_terms = len(doc)
        return {term: count / total_terms for term, count in tf.items()} if total_terms > 0 else {}

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
    
