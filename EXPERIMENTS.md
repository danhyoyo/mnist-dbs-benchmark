# DBS: layer-balanced quantized search

## Thuật toán

`layer_balanced` kết hợp hai thay đổi so với `legacy`:

1. **Biểu diễn theo lớp.** Mã `uint8` dùng zero point ở giữa dải để biểu diễn đúng 0. Scale cố định theo fan-in của từng lớp; hai ma trận weight được khởi tạo ngẫu nhiên với độ lệch chuẩn phù hợp, còn bias bắt đầu bằng 0. Không có shadow FP weights; chỉ `q` và `evidence` là trạng thái học trên từng tham số.
2. **Lịch block theo lớp.** Mỗi minibatch chia block theo căn bậc hai số tham số trong từng tensor. Con trỏ riêng của từng tensor bảo đảm các vị trí được thăm tuần tự. Lớp nhỏ, nhất là lớp đầu ra, nhận nhiều lượt probe hơn lịch block phẳng.

Quy tắc paired probe, cộng evidence, threshold và move một bin vẫn giống bản gốc. Có thể dùng `layer_scaled` và `layer_scheduled` để tách riêng hai thay đổi.

## Kết quả ban đầu

### Toàn bộ MNIST, MLP 784 → 256 → 10

Cùng seed 1, 10 epoch, batch 256, 8 paired probes, block 4096, perturb fraction 0.25, threshold 3, 256 bins. Bản `legacy` lấy từ `runs/gpu16gb/results.json`; bản mới ở `runs/dbs-layer-balanced-full/results.json`.

| Epoch | Legacy test accuracy | Layer-balanced test accuracy |
|---:|---:|---:|
| 1 | 34.71% | 40.93% |
| 2 | 37.52% | 63.69% |
| 3 | 44.75% | 73.01% |
| 4 | 51.71% | 77.95% |
| 5 | 60.57% | 80.53% |
| 6 | 67.58% | 82.90% |
| 7 | 73.72% | 84.29% |
| 8 | 77.43% | 85.00% |
| 9 | 79.42% | 86.12% |
| 10 | 80.23% | 86.39% |

Cả hai dùng 37,600 training forward evaluations. Wall time tương ứng 190.0 và 213.4 giây, nhưng hai lượt chạy diễn ra ở thời điểm GPU có tải khác nhau nên không đủ cơ sở để quy khác biệt thời gian cho thuật toán.

### Ablation nhỏ trên CPU

MNIST thật nhưng chỉ 4,096 mẫu train đầu tiên và 2,000 mẫu test đầu tiên; MLP 784 → 64 → 10, seed 1, 10 epoch, batch 256, 8 probes, block 4096, các tham số còn lại như trên.

| Variant | Test accuracy epoch 10 | Training forwards |
|---|---:|---:|
| `legacy` | 28.0% | 2,560 |
| `layer_scaled` | 26.8% | 2,560 |
| `layer_scheduled` | 27.7% | 2,560 |
| `layer_balanced` | 42.5% | 2,560 |

Trong phép thử nhỏ này, hai thay đổi phối hợp tạo lợi ích rõ hơn từng thay đổi riêng lẻ. Đây là kết quả một seed trên tập con, chỉ dùng để sàng lọc thiết kế.

## Giới hạn và bước thử tiếp

- Chạy 3–5 seed với cùng ngân sách để có trung bình và độ phân tán.
- Dùng validation tách từ train để chọn cấu hình; chỉ dùng test set cho báo cáo cuối.
- Lặp ablation trên toàn bộ MNIST để xác nhận vai trò riêng của scale và lịch block.
- Theo dõi accuracy theo wall time và forward budget. `forward_evaluations` không đo FLOPs hoặc năng lượng; bản hiện tại vẫn giải mã weight thành float để chạy PyTorch dense kernels.
- Thử số probe, perturb fraction, threshold và tỷ lệ phân bổ theo lớp sau khi có validation split.
