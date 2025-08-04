# Hướng dẫn chạy ứng dụng Web tìm kiếm Ca dao - Tục ngữ

## Cài đặt Dependencies

1. Mở Command Prompt (cmd) trong thư mục dự án
2. Cài đặt các thư viện cần thiết:

```cmd
pip install -r requirements.txt
```

## Chạy ứng dụng

1. Trong Command Prompt, chạy lệnh:

```cmd
python app.py
```

2. Mở trình duyệt web và truy cập: http://localhost:5000

## Tính năng

- **Giao diện tìm kiếm**: Giống Google với ô tìm kiếm và nút tìm kiếm
- **Xếp hạng thông minh**: Sử dụng thuật toán RankNet để xếp hạng kết quả
- **Hiển thị điểm số**: Mỗi kết quả hiển thị điểm số liên quan
- **Highlight từ khóa**: Làm nổi bật từ khóa tìm kiếm trong kết quả
- **Gợi ý tìm kiếm**: Các từ khóa phổ biến để thử nghiệm
- **Responsive**: Giao diện thích ứng với màn hình di động

## Cách sử dụng

1. Nhập từ khóa vào ô tìm kiếm (ví dụ: "ăn", "học", "làm")
2. Nhấn "Tìm kiếm" hoặc Enter
3. Xem kết quả được xếp hạng từ cao đến thấp với điểm số hiển thị
4. Có thể click vào các gợi ý để tìm kiếm nhanh

## Cấu trúc file

- `app.py`: Server Flask backend
- `templates/index.html`: Giao diện web frontend
- `requirements.txt`: Danh sách dependencies
- `main.py`: Code huấn luyện RankNet gốc
- `helper.py`: Các hàm hỗ trợ
- `Library/`: Thư viện TF-IDF và RankNet

## Troubleshooting

Nếu gặp lỗi khi chạy:

1. Kiểm tra đường dẫn file dữ liệu trong `app.py`
2. Đảm bảo file `cadao_tucngu_mini.json` tồn tại trong `FetchData/output/`
3. Kiểm tra file `vietnamese-stopwords.txt` ở thư mục gốc
4. Cài đặt lại dependencies: `pip install -r requirements.txt --force-reinstall`

## API Endpoints

- `GET /`: Trang chủ với giao diện tìm kiếm
- `POST /search`: API tìm kiếm (nhận JSON với field "query")
- `GET /health`: Kiểm tra trạng thái hệ thống
