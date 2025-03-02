# FilterFace

FilterFace là một dự án ứng dụng học sâu, sử dụng mô hình học máy để áp dụng các bộ lọc hoạt họa lên khuôn mặt trong hình ảnh. Dự án này được xây dựng dựa trên nền tảng [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template), kết hợp sức mạnh của [PyTorch Lightning](https://www.pytorchlightning.ai/) và [Hydra](https://hydra.cc/) để tạo ra một cấu trúc linh hoạt và dễ dàng mở rộng cho các dự án học sâu.

## Tính Năng

- **Tiền xử lý dữ liệu:** Cung cấp các công cụ để xử lý và chuẩn bị dữ liệu hình ảnh cho quá trình huấn luyện.
- **Huấn luyện mô hình:** Sử dụng PyTorch Lightning để huấn luyện các mô hình học sâu với cấu trúc rõ ràng và hiệu quả.
- **Tùy chỉnh cấu hình:** Sử dụng Hydra để quản lý cấu hình, cho phép dễ dàng thay đổi tham số và cấu trúc mô hình.
- **Đánh giá và kiểm thử:** Tích hợp các phương pháp đánh giá hiệu suất mô hình và kiểm thử để đảm bảo chất lượng.

## Công nghệ sử dụng và hướng tiếp cận:
- Tiền xử lý dữ liệu bằng các công cụ xử lý ảnh
- Sử dụng 1 mô hình ResNet18 để tách khuôn mặt ra khỏi khung hình, kết quả thu được là bounding box bao quanh khuôn mặt đó
- Cắt khuôn mặt có trong khung hình và tạo thành dữ liệu mới, với đầu ra là các điểm định vị trên khuôn mặt đó
- Sử dụng 1 mô hình ResNet54 để thực hiện định vị 68 điểm landmark trên khuôn mặt trên
- Áp dụng thuật toán Kalman filter để ổn định 68 điểm trên khuôn mặt khi người dùng di chuyển
- Áp dụng kỹ thuật Delaunay Triangula để gắn ảnh kỹ thuật số lên khuôn mặt dựa trên 68 điểm trên

## Demo:
[Demo Video](video3.gif)
  
## Cài Đặt
1. **Sao chép kho lưu trữ:**
   ```bash
   git clone https://github.com/chuquangcan/FilterFace.git
   cd FilterFace
   ```

2. **Tạo và kích hoạt môi trường ảo:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Trên macOS/Linux
   venv\Scripts\activate  # Trên Windows
   ```

3. **Cài đặt các gói phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

## Sử Dụng

- **Huấn luyện mô hình:**
   ```bash
   python main.py
   ```
   Lệnh này sẽ bắt đầu quá trình huấn luyện với cấu hình mặc định được xác định trong `configs/train.yaml`.

- **Tùy chỉnh cấu hình:**
   Bạn có thể thay đổi các tham số huấn luyện bằng cách chỉnh sửa tệp cấu hình hoặc sử dụng dòng lệnh:
   ```bash
   python main.py trainer.max_epochs=20 model.learning_rate=0.001
   ```
   Lệnh trên sẽ thiết lập số epoch tối đa là 20 và learning rate là 0.001.

## Cấu Trúc Thư Mục

- `configs/`: Chứa các tệp cấu hình cho Hydra.
- `src/`: Mã nguồn chính của dự án, bao gồm các mô-đun về mô hình, dữ liệu và huấn luyện.
- `scripts/`: Các script hữu ích cho việc tiền xử lý và các tác vụ khác.


