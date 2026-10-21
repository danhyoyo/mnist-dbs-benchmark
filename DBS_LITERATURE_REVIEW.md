# DBS trong bối cảnh nghiên cứu

**Mốc tra cứu:** 04/10/2026. **Phạm vi:** DBS đang được cài trong `benchmark.py`, nhất là chế độ `layer_balanced`; đây là đánh giá tài liệu khoa học, không phải xác nhận độc quyền sáng chế. Nguồn sử dụng là bài gốc công khai từ hội nghị, tạp chí, trang tác giả hoặc bản thảo arXiv. Kết quả cục bộ lấy từ `runs/gpu16gb/results.json` và `runs/dbs-layer-balanced-full/results.json`.

**Cập nhật thực nghiệm:** `benchmark.py` nay có thêm `--dbs-evidence-rule residual`, giữ phiếu vượt ngưỡng sau một lần dịch bin. Cơ chế này gần với residual accumulation của QES [8], nên không làm thay đổi kết luận về tính mới của lõi DBS. Đối chứng mới trên cùng validation split ghi trong [RESIDUAL_EXPERIMENT.md](RESIDUAL_EXPERIMENT.md): 86,55% test so với 86,10% của `reset` ở một seed toàn bộ MNIST; ba seed trên tập con đều nghiêng về residual. Đây là lợi ích bước đầu, chưa xác nhận tính ổn định trên nhiều seed toàn bộ dữ liệu.

## 1. DBS hiện đang làm gì?

Trạng thái học trên mỗi tham số gồm mã trọng số `q` kiểu `uint8` và evidence `e` kiểu `int16`. Trong một minibatch, DBS chọn block tham số, lấy nhiều hướng thưa `r` trong `{-1,0,+1}`, so sánh hai loss `L(q+r)` và `L(q-r)` trên cùng minibatch, rồi cộng lá phiếu `sign(L(q-r)-L(q+r))r` vào evidence. Khi `|e_j|` đạt ngưỡng, `q_j` dịch đúng một bin theo dấu của `e_j`. Chế độ `layer_balanced` bổ sung scale/khởi tạo theo lớp và quota block tỷ lệ với căn bậc hai kích thước tensor.

## 2. Tiền lệ gần nhất

| Thành phần của DBS | Công trình gốc | Kết luận đối chiếu |
|---|---|---|
| Hai đánh giá đối xứng | Spall, 1992 [1]; Saha et al., ICML 2021 [2] | SPSA dùng hai phép đo; tối ưu bằng so sánh cặp đã dùng một bit dấu của chênh loss. Không mới. |
| Nhiều probe, cộng `sign(loss+ − loss−) × direction` | Feng et al., NeurIPS 2024 [3], công thức sign-m-SPSA | Gần trùng lõi toán học của một minibatch DBS, với dấu đảo cho bước descent. Không mới. |
| Trọng số lượng tử, perturb lượng tử, chỉ forward | Zhao et al., 2024/2025 [4]; Bar & Giryes, ZOQO 2025 [5]; Zhou et al., EMNLP 2025 [6] | Đều đã công bố cách huấn luyện lượng tử bằng đánh giá forward. ZOQO đặc biệt gần: hai loss đối xứng, dấu chênh loss, bước cập nhật lượng tử trực tiếp. |
| Không giữ shadow FP weight, tích lũy tín hiệu và ngưỡng đổi trạng thái | Helwegen et al., Bop, NeurIPS 2019 [7]; Xu et al., QES 2026 [8] | Bop dùng momentum FP và gradient để quyết định flip; QES tích lũy residual FP16 cho tới ngưỡng lượng tử. Tích lũy rồi đổi mức rời rạc không phải ý tưởng hoàn toàn mới. |
| Block thưa hoặc phân bổ theo lớp | Zhang et al., ICML 2024 [9]; Feng et al. 2024 [3] | Block-wise/sparse zeroth-order đã có. Chưa xác minh một lịch **tuần tự theo từng tensor với quota căn bậc hai** trùng hệt. |

Một đối chiếu trực tiếp: nếu đặt một probe, evidence threshold bằng 1 và bỏ giới hạn block, bước của DBS có cùng dạng dấu như ZOQO [5]. Với nhiều probe, tổng các lá phiếu trong một minibatch có dạng sign-m-SPSA của Feng et al. [3]. DBS hiện khác chủ yếu ở **evidence `int16` tích lũy qua minibatch, ngưỡng dịch một bin và lịch block theo lớp**. QES 2026 đã làm yếu thêm khả năng tuyên bố riêng về phần “tích lũy đến ngưỡng”, dù residual của họ là FP16 và bài toán khác [8].

**Đánh giá tính mới:** chưa tìm thấy bản sao đúng toàn bộ vòng lặp DBS hiện tại trong các nguồn đã kiểm tra, nhưng điều đó không chứng minh tính mới. Tuyên bố an toàn là một *biến thể/tổ hợp* của quantized zeroth-order optimization, với giả thuyết đóng góp hẹp ở cách mã hóa evidence và điều phối lượt probe. Không nên tuyên bố đã phát minh huấn luyện không backward, paired probe, sign voting hoặc huấn luyện trực tiếp trọng số lượng tử.

Tên viết tắt **DBS** cũng đã xuất hiện trong học máy với nghĩa khác, chẳng hạn [“Dynamic Batch Size” (2020)](https://arxiv.org/abs/2007.11831). Nếu viết bài, nên định nghĩa tên đầy đủ rõ ràng và chọn tên ít gây nhầm lẫn.

## 3. Lợi ích hiện được dữ liệu hỗ trợ

- **So với DBS cũ:** trên một seed, cùng MNIST/MLP 784→256→10 và cùng 37.600 lượt forward train sau 10 epoch, accuracy tăng từ **80,23% lên 86,39%** (+6,16 điểm phần trăm). Bản mới đạt 80% ở epoch 5. Ablation nhỏ 4.096 mẫu train cho thấy chỉ kết hợp scale/khởi tạo theo lớp và lịch block theo lớp mới có lợi lớn; chưa có ablation toàn bộ MNIST hoặc nhiều seed.
- **Trạng thái tham số nhỏ về mặt biểu diễn:** 203.530 tham số × (1 byte `q` + 2 byte `evidence`) = **610.590 byte ≈ 0,582 MiB**, chưa tính scale và metadata nhỏ. SGD FP32 có momentum trong mã dùng khoảng 12 byte/tham số cho weight, gradient và momentum khi train, tức khoảng 2,33 MiB chỉ riêng các mảng đó. Đây là so sánh cấu trúc dữ liệu, chưa phải peak RAM/VRAM.
- **Chỉ cần phản hồi loss từ forward:** cập nhật DBS không gọi backward, nên về nguyên tắc dùng được khi đạo hàm không sẵn có hoặc phần cứng chỉ hỗ trợ inference. Đây là khả năng của cả họ zeroth-order, không phải lợi ích độc quyền của DBS [3–6].

## 4. Lợi ích chưa được chứng minh và chi phí

- So với SGD cùng benchmark, DBS mới **86,39% vs 98,13%**, kém 11,74 điểm phần trăm. Nó dùng 37.600 vs 2.350 lượt forward train. Wall time ghi 213,4 vs 116,2 giây, nhưng hai lần chạy khác thời điểm và GPU có tải khác nhau, nên chỉ nên xem là số đo của từng run, không phải hệ số tốc độ ổn định.
- `values()` giải mã trọng số sang `float`, `qp/qm` sao chép toàn bộ vector mã, và PyTorch chạy phép nhân ma trận/loss FP. Vì vậy hiện **chưa chứng minh tính toán integer-only**, tiết kiệm điện, hay lợi ích triển khai MCU/NPU.
- `peak_cuda_allocated_mb` của DBS mới là 35,36 MB, SGD cũ là 67,90 MB; đây là allocator peak từ các run riêng, không bao gồm toàn bộ VRAM/driver và chưa được kiểm soát môi trường. Cột năng lượng của cả hai là `null`.
- Độ chính xác được kiểm tra trên test mỗi epoch; cấu hình đã được phát triển với phản hồi từ benchmark MNIST. Muốn có kết luận nghiên cứu cần validation riêng và test cuối một lần.

## 5. Đường đi nghiên cứu có thể kiểm chứng

1. Giữ một phát biểu đóng góp hẹp: **integer vote accumulator + one-bin threshold transition + deterministic layer-balanced sparse probing**. Viết công thức và đối chiếu trực tiếp với [3], [5], [7], [8], [9].
2. Cài các baseline gần nhất trong cùng harness: sign-m-SPSA lượng tử [3], ZOQO [5], và một biến thể không có evidence. So ở cùng số forward, thời gian và ngân sách trạng thái; không chỉ so SGD/QAT.
3. Ablation toàn bộ dữ liệu, 3–5 seed: evidence `int16` vs không evidence; ngưỡng 1/2/3/5; lịch flat vs quota theo lớp; khởi tạo/scale; 4/8 bit. Báo trung bình, độ lệch và đường accuracy theo số forward.
4. Nếu mục tiêu là lợi ích phần cứng, thay phép giải mã/clone FP bằng kernel integer hoặc prototype MCU/NPU, rồi đo peak RAM, năng lượng và latency thực. Nếu mục tiêu là tối ưu black-box, đánh giá trên bài toán không có backward.

## 6. Kiểm tra ngược kết luận

- Không thấy bản sao 1:1 **không có nghĩa** tổ hợp là mới; phạm vi tra cứu chưa phải rà soát toàn bộ bài, mã nguồn và sáng chế.
- Lợi ích +6,16 điểm phần trăm có thể do khởi tạo/scale và lịch lớp, chưa cô lập evidence threshold. Một seed chưa cho biết độ ổn định.
- Dung lượng state 3 byte/tham số không suy ra peak bộ nhớ hay năng lượng vì implementation tạo buffer FP tạm. Các định lý cho mục tiêu lồi/trơn [2] không tự áp dụng cho mạng phi lồi và lưới lượng tử có clipping của DBS.

## Nguồn gốc và trạng thái

Tất cả nguồn dưới đây là **public**, do tác giả/đơn vị hội nghị hoặc kho bản thảo công bố. `[1]–[3], [6], [7], [9]` là bài hội nghị/tạp chí; `[4], [5], [8]` dùng bản công khai của tác giả, với `[4]` có bản tạp chí sau đó. Bản thảo mới hơn hoặc nguồn không bình duyệt được đọc như tiền lệ kỹ thuật, không như xác nhận chất lượng thực nghiệm.

1. [Spall, “Multivariate stochastic approximation using a simultaneous perturbation gradient approximation,” IEEE TAC, 1992](https://www.jhuapl.edu/spsa/pdf-spsa/spall_tac92.pdf) — bản tác giả công khai, nền tảng SPSA.
2. [Saha, Koren & Mansour, “Dueling Convex Optimization,” ICML 2021](https://proceedings.mlr.press/v139/saha21b.html) — tối ưu từ phản hồi so sánh cặp một bit.
3. [Feng et al., “Stepping Forward on the Last Mile,” NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ac6c452efdf063644d330caba2ab1e57-Abstract-Conference.html) — sign-m-SPSA, forward gradient lượng tử và cập nhật thưa.
4. [Zhao et al., “Poor Man’s Training on MCUs,” arXiv 2024](https://arxiv.org/abs/2411.05873) — zero-order lượng tử và sparse training trên thiết bị hạn chế tài nguyên.
5. [Bar & Giryes, “ZOQO: Zero-Order Quantized Optimization,” arXiv 2025](https://arxiv.org/html/2501.06736) — quy tắc paired loss/sign/update lượng tử rất gần DBS.
6. [Zhou et al., “QuZO,” EMNLP 2025](https://aclanthology.org/2025.emnlp-main.271/) — fine-tuning lượng tử bằng hai forward và stochastic rounding.
7. [Helwegen et al., “Latent Weights Do Not Exist,” NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/9ca8c9b0996bbf05ae7753d34667a6fd-Paper.pdf) — Bop, không latent FP weights, ngưỡng flip dựa trên momentum.
8. [Xu et al., “Quantized Evolution Strategies,” arXiv 2026](https://arxiv.org/html/2602.03120v2) — residual tích lũy qua bước tới ngưỡng chuyển mức lượng tử.
9. [Zhang et al., “Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning,” ICML 2024](https://proceedings.mlr.press/v235/zhang24ad.html) — block-wise descent và gradient sparsity.
