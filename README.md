# Learning to Rank with RankNet Algorithm

## Mô tả dự án

Đây là một dự án minh họa cách xây dựng hệ thống xếp hạng tài liệu sử dụng mô hình học sâu RankNet kết hợp với biểu diễn đặc trưng TF-IDF cho tiếng Việt. Dữ liệu đầu vào là các câu ca dao, tục ngữ, được xử lý loại bỏ stopword và biểu diễn thành vector TF-IDF. Mô hình RankNet sẽ học xếp hạng các tài liệu dựa trên mức độ liên quan với truy vấn.

## Quy trình hoạt động

1. **Tiền xử lý dữ liệu**

   - Đọc dữ liệu từ file JSON (mỗi mục có trường 'value' là một câu).
   - Lọc stopword (từ file `vietnamese-stopwords.txt`).
   - Tách từ đơn giản theo khoảng trắng.

2. **Biểu diễn TF-IDF**

   - Sử dụng lớp `TFIDF` để chuyển đổi toàn bộ tập văn bản thành ma trận TF-IDF (tương tự sklearn TfidfVectorizer).
   - Lưu lại danh sách từ vựng (feature names).

3. **Tính điểm liên quan**

   - Với một truy vấn (query), tính điểm liên quan của từng tài liệu dựa trên vector TF-IDF.

4. **Tạo cặp dữ liệu huấn luyện**

   - Sinh các cặp tài liệu (i, j) với nhãn 1 nếu tài liệu i liên quan hơn j với truy vấn, ngược lại là 0.

5. **Huấn luyện mô hình RankNet**

   - Sử dụng PyTorch để xây dựng và huấn luyện mô hình mạng neural RankNet trên các cặp dữ liệu.

6. **Dự đoán và xếp hạng**
   - Sau khi huấn luyện, mô hình dự đoán điểm số cho từng tài liệu và sắp xếp theo thứ tự giảm dần.

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
