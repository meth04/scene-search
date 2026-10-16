# Scene Search

Tìm đoạn video bằng mô tả văn bản, xem và xuất video đã che mặt/biển số.
Ứng dụng chạy local bằng Streamlit + FastAPI; **EG2 là mặc định**, V6 là
tùy chọn thử nghiệm. Không cần API key hay server GPU.

**[Tải bộ cài công khai](https://github.com/meth04/scene-search/releases/latest)**
· **[Hướng dẫn cài từ máy mới](INSTALL.md)**

Bản cài Windows 10/11 64-bit tự chuẩn bị Python, môi trường riêng,
model đúng checksum và FFmpeg. Hỗ trợ CPU hoặc NVIDIA; CPU xử lý chậm hơn.
Lần đầu cần mạng, sau đó tìm kiếm và xử lý chạy offline.

```powershell
# Trong thư mục mã đã giải nén hoặc clone:
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
powershell -NoProfile -File run_demo.ps1
```

Mở **http://127.0.0.1:8501**. Tải MP4, chờ xử lý, chọn bộ sưu tập,
tìm bằng văn bản, lọc nguồn/thời gian và xuất CSV/JSON hoặc MP4 riêng tư.
Giữ terminal mở; Ctrl+C dừng các dịch vụ do launcher khởi chạy.

- [Cài đặt, tải model, CPU/GPU và xử lý lỗi](INSTALL.md)
- [Sử dụng, API và hợp đồng privacy](docs/FINAL_LOCAL_PRODUCT_GUIDE.md)
- [Trạng thái sản phẩm/nghiên cứu](docs/PRODUCT_STATUS.md)
- [Kiến trúc](ARCHITECTURE.md)
- [Giấy phép tài nguyên đi kèm](THIRD_PARTY_NOTICES.md)

Model bundle tại Releases chỉ chứa các model công khai cần thiết.
Không cần dữ liệu benchmark, index mẫu hay checkpoint huấn luyện Stage3A.
V6.1/V6.2 không được cài vào sản phẩm.

Ứng dụng dành cho một người dùng tin cậy trên localhost.
Che tự động có thể bỏ sót đối tượng nhỏ; kiểm tra bản riêng tư trước khi chia sẻ.
