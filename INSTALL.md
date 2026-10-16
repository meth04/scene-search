# Cài Scene Search trên máy mới

Bản phát hành công khai: **https://github.com/meth04/scene-search/releases/latest**.
Không cần quyền vào repo làm việc, tài khoản Hugging Face hay API key.

## 1. Chuẩn bị

Windows 10/11 **64-bit**, khuyến nghị RAM **16 GB** và ít nhất **20 GB trống**
cho môi trường/model/cache cài đặt. Video và cache xử lý cần thêm dung lượng.
Có thể chạy bằng CPU; GPU NVIDIA từ 4 GB VRAM giúp xử lý nhanh hơn.
GTX1650 và CPU là các đường được kiểm thử trực tiếp. macOS/Linux chưa được
kiểm thử cho bộ cài PowerShell này.

Cần mạng ở lần cài đầu để tải thư viện, model khoảng **1,54 GB** và FFmpeg.
Sau khi cài xong, xử lý và tìm kiếm chạy local. Không cần Docker hay server GPU.

## 2. Lấy mã và cài

Tải **scene-search-windows-source-v1.zip** ở trang Releases, giải nén rồi mở
PowerShell trong thư mục có `install.ps1`. Hoặc dùng Git nếu đã cài Git:

```powershell
git clone https://github.com/meth04/scene-search.git
cd scene-search
```

Chạy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
```

Installer tìm Python **3.13 64-bit**; nếu thiếu sẽ thử cài theo tài khoản người
dùng bằng Windows Package Manager (`winget`). Nếu máy không có winget, cài
[Python 3.13 từ python.org](https://www.python.org/downloads/windows/) rồi chạy lại.
Installer tạo ba môi trường riêng, tải đúng model/FFmpeg, kiểm SHA-256 và chạy
kiểm tra cài đặt. Không cần chép `outputs/`, dữ liệu hay checkpoint huấn luyện
từ máy tác giả. Lần đầu có thể mất nhiều phút tùy mạng và máy.

Mặc định chọn NVIDIA nếu có; để dùng CPU kể cả trên máy có GPU:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 -Device cpu
```

CPU giữ model và FP32, nhưng indexing chậm hơn. Không dùng FP16 cho EG2.

## 3. Khởi chạy hằng ngày

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File run_demo.ps1
```

Mở **http://127.0.0.1:8501**. Giữ terminal mở; **Ctrl+C** dừng các dịch vụ
do launcher khởi chạy. Lệnh hằng ngày không cài lại thư viện hay tải model.

Tải MP4 → chờ xử lý → chọn bộ sưu tập → nhập mô tả → xem kết quả và xuất
video đã che mặt/biển số hoặc CSV/JSON. **EG2 là mặc định**; V6 shadow/active
vẫn là tùy chọn thử nghiệm. V6.1/V6.2 không nằm trong sản phẩm.

## 4. Gói model tải một lần

Tại Releases còn có **scene-search-models-v1.zip** và **SHA256SUMS.txt**.
Có thể tải gói model bằng trình duyệt hoặc mang sang máy khác, rồi cài bằng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 -ModelBundle "D:\scene-search-models-v1.zip"
```

Installer vẫn kiểm SHA-256 từng model. Gói này giúp khỏi tải lại model;
thư viện Python và FFmpeg vẫn cần mạng ở lần cài đầu. Không giải nén model
vào thư mục bất kỳ hoặc sửa hash để bỏ qua kiểm tra.

| Tài nguyên | Nguồn / mục đích |
|---|---|
| EmbeddingGemma 2, revision `914f7f8…` | Google/Hugging Face; embedding text/video |
| YOLO11n | Ultralytics; hình học và hỗ trợ che đối tượng |
| YuNet | OpenCV Zoo; gương mặt |
| YOLO license-plate | Koushim/Hugging Face; biển số |
| FFmpeg + ffprobe | Bản Windows BtbN có checksum; đọc/xuất media |

Không cần Stage3A learned checkpoint, dữ liệu benchmark, index mẫu, Qwen hay Laya.
Nguồn, phiên bản, kích thước và SHA đầy đủ nằm ở `configs/product/downloads.json`.
Xem [thông tin giấy phép](THIRD_PARTY_NOTICES.md) khi phân phối hoặc tích hợp lại.

## 5. Kiểm tra, cập nhật và xử lý lỗi

```powershell
# Chỉ kiểm tra; không mở UI hay xử lý video:
powershell -NoProfile -ExecutionPolicy Bypass -File run_demo.ps1 -Verify

# Cài lại dependency đã ghim hoặc sửa tài nguyên hỏng:
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 -Repair
```

Nếu tải bị ngắt, chạy lại installer: tệp đúng hash được giữ lại và tệp lớn đang
tải được tiếp tục nếu máy chủ hỗ trợ. Nếu báo CUDA/driver, thử `-Device cpu`.
Không xóa model đang dùng, không tăng ngưỡng hay bỏ qua lỗi kiểm tra tính toàn vẹn.

Nếu dùng Git, cập nhật bằng `git pull --ff-only`, rồi chạy installer với `-Repair`.
Nếu dùng ZIP, giải nén bản mới vào thư mục mới và cài; giữ bản cũ để bảo toàn thư
viện video của bạn. Dữ liệu ứng dụng nằm ở `outputs/product/`, model ở `data/`.
Không chép các thư mục này đè lên một cài đặt khác khi chưa kiểm tra tính tương thích.

Ứng dụng dành cho một người dùng tin cậy trên localhost. Che tự động có thể bỏ
sót đối tượng nhỏ; kiểm tra bản riêng tư trước khi chia sẻ. Chi tiết API, giới hạn
video và cơ chế khôi phục nằm trong [hướng dẫn sản phẩm](docs/FINAL_LOCAL_PRODUCT_GUIDE.md).
