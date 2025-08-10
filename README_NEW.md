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

## Cài đặt và sử dụng

### 1. **Cài đặt môi trường**

```bash
# Clone repository
git clone https://github.com/LuongXuanNhat/Learning-to-Rank-with-Rank-Net-Algorithm.git
cd Learning-to-Rank-with-Rank-Net-Algorithm

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. **Chuẩn bị dữ liệu**

Đảm bảo có các file sau:

- `FetchData/output/cadao_tucngu_medium.json`: Dữ liệu ca dao, tục ngữ
- `vietnamese-stopwords.txt`: Danh sách stopwords tiếng Việt

### 3. **Chạy training**

#### Option 1: Sử dụng script Python

```bash
python main_f2.py
```

#### Option 2: Sử dụng Jupyter Notebook

```bash
jupyter notebook notebook/L2R_RankNet_V3.ipynb
```

### 4. **Demo web interface**

```bash
python app.py
# Mở browser tại http://localhost:5000
```

## Kết quả mẫu

### Training Process:

```
Epoch 1/40, Average Loss: 0.6932
Epoch 5/40, Average Loss: 0.5234
Epoch 10/40, Average Loss: 0.4567
...
New best model saved with loss: 0.4123
```

### Ranking Results:

```
Query: 'gia đình hạnh phúc'

Top 10 documents:
 1. [ID: 15] Score: 2.3456
    Text: Gia đình hòa thuận thì mọi việc đều thành công...

 2. [ID: 42] Score: 2.1234
    Text: Nhà có vợ hiền là phúc lớn...

Pairwise Accuracy: 0.8542 (85.42%)
```

## Cấu trúc dự án

```
Learning-to-Rank-with-Rank-Net-Algorithm/
├── notebook/
│   ├── L2R_RankNet_V1.ipynb      # Version 1 (TF-IDF)
│   ├── L2R_RankNet_V2.ipynb      # Version 2 (Cải tiến)
│   └── L2R_RankNet_V3.ipynb      # Version 3 (Sentence Transformers)
├── FetchData/
│   ├── output/
│   │   ├── cadao_tucngu_mini.json    # Dataset nhỏ (10 documents)
│   │   ├── cadao_tucngu_medium.json  # Dataset trung bình (50+ documents)
│   │   └── cadao_tucngu_complete.json # Dataset đầy đủ
│   └── crawl_data.py                 # Script thu thập dữ liệu
├── Library/
│   ├── rank_net.py              # RankNet implementation
│   └── TF_IDF.py               # TF-IDF implementation (version cũ)
├── templates/
│   └── index.html              # Web interface template
├── main_f2.py                  # Script chính (Sentence Transformers)
├── app.py                      # Web application
├── requirements.txt            # Dependencies
├── vietnamese-stopwords.txt    # Vietnamese stopwords
└── README.md                   # Documentation
```

## API và Classes chính

### 1. **RankNet Class**

```python
class RankNet(nn.Module):
    def __init__(self, input_size, hidden_size1=256, hidden_size2=128, dropout=0.2)
    def forward(self, x)
    def predict_rank(self, x1, x2)  # So sánh 2 documents
```

### 2. **DataPreparator Class**

```python
class DataPreparator:
    def __init__(self, sentence_model, stopwords_path=None)
    def preprocess_text(self, text) -> str
    def calculate_relevance_scores(self, query, documents) -> List[Tuple]
    def generate_pairwise_data(self, queries, documents) -> List[Dict]
    def save_pairwise_data(self, pairwise_data, file_path)
```

### 3. **RankNetDataset Class**

```python
class RankNetDataset(Dataset):
    def __init__(self, pairwise_data, sentence_model)
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]
```

### 4. **Utility Functions**

```python
def train_ranknet(model, dataloader, optimizer, device, num_epochs=20)
def evaluate_ranking(model, documents, queries, sentence_model, device, top_k=10)
def calculate_pairwise_accuracy(model, test_data, sentence_model, device)
def test_specific_document_pair(model, sentence_model, preparator, documents, query, doc_id1, doc_id2, device)
```

## Hiệu suất và đánh giá

### Metrics:

- **Pairwise Accuracy**: Tỷ lệ dự đoán đúng trên các cặp documents
- **Top-K Precision**: Chất lượng kết quả trong top K
- **Loss Convergence**: Hội tụ của hàm mất mát trong quá trình training

### Benchmark với Cosine Similarity:

Mô hình RankNet thường đạt hiệu suất tốt hơn 10-15% so với cosine similarity baseline trên dataset ca dao, tục ngữ.

## Tùy chỉnh và mở rộng

### 1. **Thay đổi mô hình Sentence Transformer**

```python
# Trong main_f2.py, thay đổi model name:
sentence_model = SentenceTransformer('your-preferred-model-name')
```

### 2. **Điều chỉnh kiến trúc RankNet**

```python
model = RankNet(
    input_size=input_size,
    hidden_size1=512,      # Tăng kích thước hidden layer
    hidden_size2=256,
    dropout=0.3            # Điều chỉnh dropout rate
)
```

### 3. **Thêm dataset mới**

```python
# Format dữ liệu JSON:
[
  {"id": 1, "value": "Nội dung tài liệu 1"},
  {"id": 2, "value": "Nội dung tài liệu 2"}
]
```

## Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory**: Giảm batch_size trong DataLoader
2. **Slow training**: Sử dụng GPU hoặc giảm kích thước dataset
3. **Poor accuracy**: Tăng số epochs hoặc điều chỉnh learning rate
4. **Overfitting**: Tăng dropout rate hoặc thêm weight decay

### Performance Tips:

- Sử dụng GPU cho training nhanh hơn
- Cache embeddings để tránh tính lại
- Sử dụng mixed precision training cho efficiency

## Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Liên hệ

- **Tác giả**: LuongXuanNhat
- **Repository**: [https://github.com/LuongXuanNhat/Learning-to-Rank-with-Rank-Net-Algorithm](https://github.com/LuongXuanNhat/Learning-to-Rank-with-Rank-Net-Algorithm)
- **Email**: luongxuannhat.dev@gmail.com

## Tài liệu tham khảo

1. Burges, C. et al. (2005). "Learning to rank using gradient descent"
2. Liu, T.Y. (2009). "Learning to rank for information retrieval"
3. Sentence Transformers Documentation: https://www.sbert.net/
4. PyVi - Vietnamese NLP Toolkit: https://github.com/trungtv/pyvi
