# Đề cương nghiên cứu

**Tên đề tài đề xuất (tiếng Việt):** Kiến trúc thời gian thực, tiết kiệm năng lượng và bền vững với ánh sáng cho nhận diện mệnh giá và hư hỏng (rách) tiền polymer Việt Nam trên thiết bị Android hỗ trợ người khiếm thị

**Tên đề xuất (tiếng Anh, để nộp JRTIP):** *An Energy-Aware, Illumination-Robust Real-Time Framework for Joint Denomination and Tear Detection of Polymer Banknotes on Mobile Devices for Visually Impaired Users*

> Lưu ý: tên đề tài **cố tình không đặt "banknote recognition" làm từ khóa chính** vì mảng này đã bão hòa (xem mục 2). Trọng tâm được chuyển sang 3 cụm từ khóa mà JRTIP thực sự tìm: *real-time*, *energy-aware*, *resource-constrained device*.

---

## 1. Bối cảnh và động lực

Người khiếm thị gặp khó khăn khi tự xác định mệnh giá tiền giấy polymer, đặc biệt khi tờ tiền đã cũ, bị rách, hoặc điều kiện ánh sáng chụp không lý tưởng (ngược sáng, cháy sáng ngoài trời). Hai vấn đề — **(a) sai lệch dự đoán do ánh sáng** và **(b) chi phí tính toán/năng lượng khi chạy liên tục trên điện thoại để hướng dẫn người dùng bằng giọng nói** — hiếm khi được giải quyết đồng thời trong cùng một hệ thống.

## 2. Khoảng trống nghiên cứu (Research gap)

Nhận diện tiền cho người khiếm thị bằng deep learning đã được nghiên cứu khá nhiều: tiền Ấn Độ (MobileNetV2 + TFLite trên Raspberry Pi), tiền Libya (YOLOv11 + app di động), tiền Bangladesh (CNN thời gian thực), tiền Ai Cập (benchmark YOLOv8/v9/v10), tiền Uganda (phát hiện tiền giả trên smartphone), và bộ dữ liệu IPCD quy mô lớn (~50.000 ảnh) dùng Faster R-CNN 3 giai đoạn cho ứng dụng "Roshni". Vì vậy:

- **Không nên** định vị đóng góp chính là "nhận diện mệnh giá" — điều này gần như chắc chắn bị reviewer chỉ ra là đã có nhiều tiền lệ.
- **Khoảng trống thật sự**: chưa có công trình nào kết hợp đồng thời (i) phát hiện **rách + định vị vùng rách**, (ii) module **bền vững với ánh sáng có thể học được** (không phải CLAHE/Retinex tĩnh), và (iii) kiến trúc **real-time tiết kiệm năng lượng có đo đạc thực tế** (mJ/inference) trên thiết bị Android thật. Đây là "khe hở 3 chiều" cần nhấn mạnh xuyên suốt bài báo.

## 3. Mục tiêu & 3 đóng góp chính

| Mã | Đóng góp | Câu hỏi nghiên cứu (RQ) |
|---|---|---|
| **C1** | Định lượng hoá vấn đề (problem characterization) | Ánh sáng ảnh hưởng khác nhau thế nào đến (a) phân loại mệnh giá và (b) phát hiện/định vị vết rách? |
| **C2** | Module bền vững ánh sáng học được + cổng chất lượng (quality-gate) | Một sub-network hiệu chỉnh ánh sáng học end-to-end + loss consistency có cải thiện độ chính xác dưới điều kiện xấu mà không tốn nhiều chi phí tính toán không? |
| **C3** | Kiến trúc real-time tiết kiệm năng lượng | Multi-task backbone dùng chung + cascade/early-exit + quantization giảm được bao nhiêu % latency/năng lượng so với chạy 3 model riêng lẻ, mà vẫn giữ được độ chính xác? |

## 4. Kiến trúc hệ thống đề xuất (đặt tên tạm)

```
Giọng nói hướng dẫn khung hình
   → [Bước A] Quality-Gate siêu nhẹ (vài chục KB)
        phát hiện: mờ / cháy sáng / ngược sáng / lệch khung
        chạy liên tục theo thời gian thực (mỗi frame)
      ├─ Chưa đạt → phản hồi giọng nói ("xoay tờ tiền", "tránh ánh sáng")
      └─ Đạt điều kiện → trigger Bước B (chỉ chạy 1 lần/lượt)
   → [Bước B] Illumination Correction Module (IC-Net, học được, vài lớp conv)
   → [Bước C] Backbone dùng chung (multi-task, ví dụ MobileNetV3-based)
        ├─ Head 1: Phân loại mệnh giá (6 lớp)
        ├─ Head 2: Nhị phân rách / không rách
        └─ Head 3: Bounding box vùng rách
   → Text-to-Speech: "Đây là tờ 50.000, bị rách góc dưới bên phải"
```

Đặt tên gợi ý cho 2 module mới để bài có "tên riêng" (thường được JRTIP đánh giá cao vì dễ trích dẫn): **IC-Net** (illumination correction) và **TG-Cascade** (torn/gate cascade). Có thể đổi tên tuỳ ý khi viết bài.

## 5. Bộ dữ liệu hiện có và kế hoạch bổ sung

Dữ liệu hiện tại (theo bảng bạn cung cấp), tính trên **1 mệnh giá**, nhân 6 mệnh giá:

| Điều kiện chụp | Nguồn tờ | Train | Val | Test |
|---|---|---|---|---|
| Trong nhà (Indoor) bình thường | Nguyên vẹn | 45 | 10 | 10 |
| Ngoài trời (Outdoor) | Nguyên vẹn | 45 | 10 | 10 |
| Ngược sáng (Backlight) | Nguyên vẹn | 35 | 8 | 8 |
| Cháy sáng khó nhìn | Nguyên vẹn | 35 | 8 | 8 |
| Rách — ảnh sạch | Rách | 25 | 5 | 5 |
| Rách — ảnh cháy sáng | Rách | 25 | 5 | 5 |
| **Tổng/1 mệnh giá** | | **210** | **46** | **46** |
| **Tổng/6 mệnh giá** | | **1260** | **276** | **276** |

**Rủi ro cần xử lý:** 302 ảnh/mệnh giá là nhỏ so với chuẩn deep learning, và tập test chỉ 46 ảnh/mệnh giá (~8 ảnh/điều kiện) khiến các con số accuracy/mAP dễ bị dao động lớn và khó thuyết phục reviewer. Kế hoạch giảm rủi ro:

1. **Transfer learning** bắt buộc — không train from scratch.
2. **Augmentation mô phỏng ánh sáng** (gamma jitter, exposure, ngược sáng tổng hợp) để tăng biến thể mà không cần thu thêm ảnh thật — đồng thời phục vụ trực tiếp cho loss consistency ở C2.
3. **K-fold cross-validation** (đề xuất 5-fold) thay vì 1 split cố định duy nhất, báo cáo mean ± std cho mọi bảng kết quả.
4. Nếu còn thời gian: thu thêm ảnh thật (mục tiêu +50% mỗi điều kiện) trong Tuần 1–3 của lịch trình bên dưới, chạy song song với các việc khác.
5. Nêu rõ **cỡ dữ liệu nhỏ là một giới hạn đã biết** trong phần Discussion/Limitation, không né tránh — reviewer đánh giá cao sự trung thực hơn là bị bắt lỗi.

## 6. Kế hoạch thí nghiệm chi tiết

| # | Thí nghiệm | Mục đích (gắn với đóng góp) | Đầu ra cho bài báo |
|---|---|---|---|
| E1 | Train baseline chỉ trên "Trong nhà bình thường", test riêng trên từng điều kiện còn lại | C1 | Bảng accuracy/mAP/IoU theo điều kiện (motivation table) |
| E2 | Grad-CAM / feature map trên các case sai | C1 | Hình minh hoạ vùng model "nhìn nhầm" |
| E3 | So sánh mức độ ảnh hưởng của ánh sáng lên classification vs. tear detection/localization | C1 | Biểu đồ lệch giữa 2 task |
| E4 | Ablation IC-Net: có/không illumination correction module | C2 | Bảng accuracy trước/sau, chi phí thêm (ms, KB) |
| E5 | Ablation consistency loss: có/không loss ràng buộc embedding bất biến ánh sáng | C2 | Bảng accuracy + t-SNE/embedding visualization |
| E6 | Ablation quality-gate: có/không cổng lọc chất lượng trước khi chạy model chính | C2 + C3 | Tỷ lệ frame bị chặn, năng lượng tiết kiệm/phiên sử dụng |
| E7 | Multi-task backbone (3 head) vs. 3 model riêng biệt | C3 | Bảng accuracy, latency, energy (mJ), model size (MB) |
| E8 | Cascade/early-exit bật/tắt, đo trên **1 phiên sử dụng mô phỏng** (nhiều frame trước khi người dùng chỉnh đúng khung) | C3 | Tổng năng lượng/phiên, không chỉ 1 lần inference |
| E9 | Quantization-aware training (INT8) + so sánh FP32 gốc | C3 | Bảng accuracy drop vs. giảm size/latency |
| E10 | (tuỳ chọn) Knowledge distillation: teacher lớn → student nhẹ | C3 | Bảng giữ accuracy nhưng giảm chi phí |
| E11 | So sánh với backbone SOTA nhẹ khác (MobileNetV3, EfficientNet-Lite, YOLOv8n/v11n, ShuffleNet) — cùng chạy trên **cùng một thiết bị Android thật** | C3 | Bảng so sánh tổng hợp |
| E12 | Đồ thị Pareto: accuracy vs. latency vs. energy | C3 (tổng hợp) | Hình trọng tâm của bài (JRTIP rất thích dạng này) |
| E13 | Đo năng lượng thực tế bằng Android Batterystats / power monitor (không ước lượng) | C3 | Số liệu mJ/inference đáng tin cậy, tăng độ thuyết phục |

## 7. Chỉ số đánh giá (metrics)

- **Độ chính xác**: Accuracy, mAP@0.5 (rách), IoU trung bình vùng rách, F1 (rách/không rách)
- **Thời gian thực**: FPS, latency trung bình/tail latency (p95)
- **Năng lượng**: mJ/inference, tổng mJ/phiên sử dụng
- **Tài nguyên**: kích thước model (MB), RAM peak
- **Thống kê**: mean ± std qua k-fold, kiểm định ý nghĩa (paired t-test) khi so sánh các cấu hình

## 8. Lịch trình thực hiện (đề xuất ~26 tuần / 6 tháng đến khi nộp)

| Giai đoạn | Tuần | Công việc chính | Kết quả cần có |
|---|---|---|---|
| **P0 – Chuẩn bị** | 1–2 | Rà soát/hoàn thiện nhãn dữ liệu; viết related work đối chiếu các bài đã nêu ở mục 2; dựng pipeline đánh giá chuẩn (accuracy/mAP/IoU) và pipeline đo năng lượng trên điện thoại Android thật | Bộ dữ liệu sạch, related-work draft, pipeline benchmark chạy được |
| **P0.5 – (song song, tuỳ chọn)** | 1–3 | Thu thêm ảnh thật nếu có nguồn lực | Dataset mở rộng |
| **P1 – Chứng minh vấn đề (C1)** | 3–5 | Chạy E1, E2, E3 | Bảng + hình motivation, xác nhận giả thuyết ban đầu |
| **P2 – Module ánh sáng (C2)** | 6–10 | Thiết kế và huấn luyện IC-Net; thiết kế loss consistency; huấn luyện quality-gate; chạy E4, E5, E6 | Module C2 hoàn chỉnh + bảng ablation |
| **P3 – Kiến trúc real-time (C3), phần 1** | 11–14 | Thiết kế multi-task backbone 3 head; huấn luyện; chạy E7 | Backbone multi-task hoạt động, so sánh với 3-model-riêng |
| **P3 – Kiến trúc real-time (C3), phần 2** | 15–16 | Thiết kế cascade/early-exit; chạy E8 | Số liệu tiết kiệm năng lượng theo phiên |
| **P4 – Nén mô hình & triển khai Android** | 17–19 | QAT INT8 (E9), tuỳ chọn distillation (E10), convert TFLite/NNAPI, đo bằng Batterystats/power monitor (E13) | App/prototype chạy thật trên điện thoại, số liệu năng lượng tin cậy |
| **P5 – So sánh & tổng hợp** | 20–21 | Chạy E11 (so sánh backbone SOTA), dựng E12 (Pareto plot) | Bảng so sánh tổng hợp + hình Pareto trung tâm |
| **P6 – Kiểm thử tích hợp** | 22 | Test end-to-end toàn pipeline (giọng nói → gate → IC-Net → backbone → TTS), thử với vài người dùng thật/mô phỏng | Video/demo, ghi nhận phản hồi định tính |
| **P7 – Viết bài** | 23–24 | Viết bản thảo đầy đủ, hoàn thiện hình/bảng, tự đối chiếu với "Note to authors" của JRTIP về real-time | Bản thảo hoàn chỉnh |
| **P8 – Rà soát & nộp** | 25–26 | Đọc lại nội bộ / nhờ đồng nghiệp góp ý, chỉnh sửa, nộp JRTIP | Bài nộp chính thức |

> Có thể rút gọn còn ~16–18 tuần nếu bỏ E10 (distillation, tuỳ chọn) và làm P2–P3 song song nếu có 2 người phụ trách 2 module độc lập.

## 9. Rủi ro & phương án dự phòng

| Rủi ro | Ảnh hưởng | Phương án |
|---|---|---|
| Dataset quá nhỏ, kết quả không ổn định | Reviewer nghi ngờ generalization | K-fold, augmentation mạnh, nêu rõ giới hạn, cân nhắc thu thêm dữ liệu |
| Không có thiết bị đo năng lượng chuyên dụng (Monsoon) | Số liệu mJ kém tin cậy | Dùng Android Batterystats/`dumpsys batterystats` làm phương án thay thế, nêu rõ sai số ước lượng |
| Module ánh sáng làm tăng latency, phản tác dụng với mục tiêu real-time | Mâu thuẫn nội tại giữa C2 và C3 | Thiết kế IC-Net cực nhẹ (vài lớp conv), luôn báo cáo chi phí thêm (ms) song song với lợi ích accuracy — biến đây thành một trade-off phân tích được, không phải nhược điểm giấu đi |
| Reviewer cho rằng cascade/quantization/multi-task không mới (đã có trong literature) | Bị đánh giá thấp về novelty | Nhấn mạnh **novelty nằm ở tổ hợp + bối cảnh ứng dụng cụ thể (tiền rách + người khiếm thị) + đo đạc thực nghiệm nghiêm túc trên thiết bị thật**, không claim novelty thuật toán thuần túy |

## 10. Cấu trúc bài báo dự kiến (map với JRTIP)

1. Introduction (nêu bài toán, khoảng trống 3 chiều ở mục 2, đóng góp C1–C3)
2. Related Work (so sánh trực diện với các bài banknote-for-blind đã liệt kê + các kỹ thuật real-time/embedded liên quan)
3. Dataset (mô tả bảng ở mục 5, quy trình gán nhãn rách)
4. Proposed Method (kiến trúc mục 4: Quality-Gate, IC-Net, multi-task backbone, cascade)
5. Experimental Setup (thiết bị thật dùng để đo, metrics ở mục 7)
6. Results — theo đúng thứ tự E1→E13 ở mục 6
7. Discussion (trade-off, giới hạn dataset, hướng mở rộng)
8. Conclusion

---

*Tài liệu này là bản đề cương làm việc — có thể điều chỉnh tên module, thứ tự thí nghiệm, hoặc rút gọn lịch trình tuỳ theo nguồn lực thực tế của nhóm.*
