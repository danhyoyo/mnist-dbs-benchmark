# DBS với bộ tích lũy phiếu có residual

## Quy tắc và cơ sở

Trong mỗi minibatch, DBS nhận tổng phiếu nguyên `v_j` từ các cặp loss đối xứng. Với threshold `T`, bản `reset` đặt evidence về 0 sau một lần dịch bin. Bản `residual` thực hiện:

```text
e_j ← clamp(e_j + v_j, -32767, 32767)
if |e_j| ≥ T and bin còn dịch được:
    q_j ← q_j + sign(e_j)
    e_j ← e_j - T × sign(e_j)
```

Nếu bin đã chạm biên và evidence muốn đẩy ra ngoài, evidence được xóa. Mỗi lần thăm vẫn dịch tối đa một bin, nên residual lớn có thể tạo các bước dịch ở nhiều minibatch kế tiếp. Trạng thái persistent vẫn chỉ là `uint8 q` và `int16 evidence`; số forward mỗi step giữ nguyên `2 × probes`.

Đây là một biến thể tích lũy sai số sau lượng tử hóa có cơ sở từ [quantized evolution strategies của Xu et al. (2026)](https://arxiv.org/html/2602.03120v2) và nguyên lý residual/error feedback. Cách lấy phiếu từ hai loss có quan hệ gần với [sign-m-SPSA của Feng et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ac6c452efdf063644d330caba2ab1e57-Abstract-Conference.html) và [ZOQO của Bar & Giryes (2025)](https://arxiv.org/html/2501.06736). Do đó đây là cải tiến thực nghiệm của tổ hợp DBS, **không phải tuyên bố một lõi tối ưu zero-order mới**.

## Kết quả sàng lọc trên MNIST thật

Hai cách dùng cùng MLP 784→64→10, 4.096 ảnh train đầu tiên, 2.000 ảnh test đầu tiên, 10 epoch, batch 256, block 4096, 8 probe, threshold 3, 256 bin và cùng seed trong từng cặp. Đây là phép thử sàng lọc, đã quan sát test nên không dùng nó như kiểm chứng độc lập.

| Seed | Reset accuracy | Residual accuracy | Chênh lệch |
|---:|---:|---:|---:|
| 1 | 42,50% | 46,60% | +4,10 điểm % |
| 2 | 49,45% | 51,60% | +2,15 điểm % |

Mỗi run dùng 2.560 training forward evaluations. Dữ liệu gốc nằm ở `runs/dbs-{reset,residual}-cpu-seed{1,2}/results.json`.

### Đối chứng tập con với validation riêng

Một phép thử khác dùng 4.096 ảnh đầu của train MNIST, chia ngẫu nhiên theo từng seed thành 3.584 train + 512 validation; test là 2.000 ảnh đầu. MLP 784→64→10, 10 epoch, 3 seed, cùng thứ tự batch trong từng cặp. Chọn epoch theo validation rồi mới tính test. Vì split và số ảnh train khác phép sàng lọc trên, không so trực tiếp hai bảng.

| Seed | Reset test accuracy | Residual test accuracy | Chênh lệch |
|---:|---:|---:|---:|
| 1 | 40,25% | 41,65% | +1,40 điểm % |
| 2 | 45,55% | 47,50% | +1,95 điểm % |
| 3 | 42,85% | 46,05% | +3,20 điểm % |
| Trung bình | 42,88% | 45,07% | +2,18 điểm % |

Độ lệch chuẩn giữa seed của test accuracy là 2,65 điểm % cho reset và 3,05 điểm % cho residual. Cả sáu run dùng 2.240 training forward evaluations. Kết quả ở `runs/compare-subset-3seeds/`. Tập con nhỏ có phương sai và độ khó khác toàn bộ MNIST; bằng chứng chính cho hiệu quả trên toàn bộ dữ liệu vẫn mới chỉ có một seed.

## Đối chứng toàn bộ MNIST, seed 1

`train_compare.py` lấy ngẫu nhiên 5.000 mẫu từ 60.000 ảnh train làm validation, dùng 55.000 ảnh còn lại để học, và đánh giá 10.000 ảnh test **sau khi** chọn checkpoint tốt nhất theo validation accuracy. Cùng MLP 784→256→10, 10 epoch và cùng thứ tự minibatch trong từng phương pháp. Hai DBS dùng 8 probe, block 4096, threshold 3, 256 bin.

| Phương pháp | Val accuracy tốt nhất | Test accuracy | Training forwards | Thời gian |
|---|---:|---:|---:|---:|
| SGD | 98,04% | 98,03% | 2.150 | 98,8 s |
| DBS reset | 85,54% | 86,10% | 34.400 | 197,6 s |
| DBS residual | 85,76% | 86,55% | 34.400 | 200,1 s |

Residual hơn reset **0,45 điểm phần trăm test** ở seed này, và giảm test cross-entropy từ 0,4794 xuống 0,4625. Lợi thế này nhỏ; một seed chưa đủ khẳng định ổn định thống kê. SGD vẫn hơn residual 11,48 điểm phần trăm và dùng ít lượt forward train hơn 16 lần. Thời gian là số đo của một lượt chạy trên máy này, gồm đánh giá validation/test và ảnh hưởng tải hệ thống; không phải kết luận tốc độ tổng quát. Dữ liệu đầy đủ và checkpoint ở `runs/compare-full-seed1/`.

## Giao thức lặp tiếp

Chạy 3–5 seed trên toàn bộ MNIST để ước lượng biến thiên của chênh lệch, rồi kiểm tra trên Fashion-MNIST hoặc tập khác chưa dùng trong quá trình thiết kế. So `sgd`, `dbs_reset`, `dbs_residual` trên cùng split, thứ tự batch và seed. Báo accuracy, loss, epoch chọn, số training forward, thời gian và số bin move. Chênh lệch nhỏ trên một seed chưa chứng minh cải tiến ổn định.

## Giới hạn cần giữ rõ

- Cả hai bản DBS vẫn giải mã weight sang float để dùng PyTorch và clone vector mã khi probe. Không có bằng chứng tiết kiệm điện hay thời gian so với SGD.
- Kết quả 86,55% ở đây dùng 55.000 ảnh để train và validation riêng; kết quả 86,39% của benchmark trước dùng cả 60.000 ảnh train và đánh giá test mỗi epoch. Không trừ hai con số đó để suy ra tác dụng của residual; đối chứng đúng là 86,55% so với 86,10% trong cùng giao thức mới.
- Phép thử cũ trên MNIST đã dùng test trong quá trình phát triển. Giao thức validation mới tốt hơn cho so sánh giữa các biến thể, nhưng một công bố học thuật cần thêm tập dữ liệu độc lập và baseline zero-order gần nhất.
- Residual có thể lưu phiếu nhiễu lâu hơn; lợi ích phụ thuộc threshold, probe, block và bài toán. Nên xem `reset` như đối chứng bắt buộc.
