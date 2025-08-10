# Learning to Rank with RankNet Algorithm

## Mô tả dự án

Dự án này triển khai hệ thống **Learning to Rank (L2R)** sử dụng thuật toán **RankNet** với **Sentence Transformers** để xếp hạng tài liệu tiếng Việt. Hệ thống được thiết kế để xếp hạng các câu ca dao, tục ngữ Việt Nam dựa trên mức độ liên quan với truy vấn của người dùng.

### Điểm nổi bật:

- ✅ Sử dụng **Sentence Transformers** thay vì TF-IDF để có biểu diễn ngữ nghĩa tốt hơn
- ✅ Tiền xử lý văn bản tiếng Việt với **PyVi** (tách từ + loại bỏ stopwords)
- ✅ Mô hình **RankNet** với 2 tầng ẩn và các kỹ thuật tối ưu (LayerNorm, Dropout, Early Stopping)
- ✅ Đánh giá hiệu suất với **Pairwise Accuracy**
- ✅ Giao diện web Flask để demo trực quan

## Kiến trúc hệ thống

### 1. **Tiền xử lý dữ liệu (Data Preprocessing)**

- **Input**: Dữ liệu JSON với format:
  ```json
  [
    { "id": 1, "value": "Ăn quả nhớ kẻ trồng cây" },
    { "id": 2, "value": "Anh em bốn bể là nhà" }
  ]
  ```
- **PyVi Tokenization**: Tách từ chính xác cho tiếng Việt
- **Stopwords Removal**: Loại bỏ từ dừng từ file `vietnamese-stopwords.txt`
- **Text Normalization**: Chuẩn hóa văn bản (lowercase, remove punctuation)

### 2. **Vector hóa với Sentence Transformers**

- **Model**: `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)
- **Query Embedding**: Vector biểu diễn truy vấn
- **Document Embedding**: Vector biểu diễn từng tài liệu
- **Feature Vector**: Concatenation [query_emb + doc_emb] = 768 dimensions

### 3. **Tạo dữ liệu Pairwise Training**

- **Relevance Score Assignment**: Dựa trên cosine similarity ranking
  - Top 10%: Highly relevant (score = 3)
  - Top 20%: Relevant (score = 2)
  - Top 33%: Somewhat relevant (score = 1)
  - Còn lại: Not relevant (score = 0)
- **Pairwise Labels**:
  - `1.0`: Document 1 liên quan hơn Document 2
  - `0.0`: Document 1 kém liên quan hơn Document 2
  - `0.5`: Hai documents có mức độ liên quan bằng nhau

### 4. **Mô hình RankNet**

```
Input (768) → FC1 (256) → LayerNorm → ReLU → Dropout
            → FC2 (128) → LayerNorm → ReLU → Dropout
            → FC3 (1) → Score
```

- **Loss Function**: RankNet Cross-Entropy Loss
- **Optimization**: Adam optimizer với learning rate scheduling
- **Regularization**: Dropout + Weight Decay + Gradient Clipping

### 5. **Đánh giá và Ranking**

- **Pairwise Accuracy**: Đo độ chính xác trên cặp documents
- **Top-K Ranking**: Xuất top 10 documents liên quan nhất
- **Comparison**: So sánh với cosine similarity baseline

## Cách sử dụng

1. Cài đặt các thư viện cần thiết:

   ```bash
   pip install numpy torch
   ```

2. Đảm bảo các file dữ liệu:

   - `FetchData/output/cadao_tucngu_mini.json`: Dữ liệu ca dao, tục ngữ.
   - `FetchData/vietnamese-stopwords.txt`: Danh sách stopword tiếng Việt.

3. Chạy file chính:

   ```bash
   python main.py
   ```

4. Kết quả:
   - In ra số tài liệu, số từ vựng, kích thước ma trận TF-IDF.
   - In điểm liên quan của từng tài liệu với truy vấn mẫu.
   - In số cặp dữ liệu huấn luyện.
   - In kết quả xếp hạng tài liệu sau khi huấn luyện.
     Ví dụ với từ truy vấn `ăn`
     ```bash
     Số tài liệu: 10
     Số từ trong từ điển: 71
     Kích thước ma trận TF-IDF: 10
     Điểm liên quan của từ 'ăn': [0.33526681861307994, 0.0, 0.33526681861307994, 0.50290022791962, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
     Số cặp tài liệu: 23
     Epoch 0, Loss: 0.6946
     Epoch 10, Loss: 0.5505
     Epoch 20, Loss: 0.4979
     Epoch 30, Loss: 0.4970
     Epoch 40, Loss: 0.4927
     Epoch 50, Loss: 0.4926
     Epoch 60, Loss: 0.4922
     Epoch 70, Loss: 0.4921
     Epoch 80, Loss: 0.4920
     Epoch 90, Loss: 0.4920
     Kết quả xếp hạng:
     1.5658 - Ăn quả nhớ kẻ trồng cây
     1.5066 - Anh em bốn bể là nhà Người dưng khác họ vẫn là anh em
     1.0139 - Ăn trông nồi, ngồi trông hướng
     0.5692 - Ăn vóc học hay.
     -0.5608 - Áo năng thay năng mới, người năng tới năng thân
     -0.5612 - Anh đi anh nhớ quê nhà Nhớ canh rau muống nhớ cà dầm tương
     -0.5632 - Anh em nào phải người xa, Cùng chung bác mẹ một nhà cùng thân.
     -0.5635 - Ai giàu ba họ, ai khó ba đời
     -0.5643 - Anh em như thể chân tay Rách lành đùm bọc, dở hay đỡ đần
     -0.5677 - Ao sâu tốt cá
     ```

## Cấu trúc thư mục

- `FetchData/`: Chứa dữ liệu và stopword.
- `Library/TF-IDF/`: Thư viện TF-IDF tự xây dựng.
- `helper.py`: Các hàm tiện ích (đọc dữ liệu, tính điểm liên quan, ...).
- `main.py`: Chạy toàn bộ pipeline từ tiền xử lý đến huấn luyện và xếp hạng.
- `Library/rank_net.py`: Định nghĩa mô hình RankNet.

## Liên hệ

- Tác giả: LuongXuanNhat
- Repo: https://github.com/LuongXuanNhat/Learning-to-Rank-with-Rank-Net-Algorithm

```

```
