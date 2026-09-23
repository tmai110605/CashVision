# Hướng dẫn thực thi: Tách nội dung sang Supplementary Material
## Bản thảo: CashVision (nộp Expert Systems with Applications)

Đây là chỉ thị làm việc cho AI/agent có quyền chỉnh sửa trực tiếp file nguồn (.tex nếu dùng LaTeX elsarticle, hoặc .docx nếu dùng Word). Mục tiêu: giảm ~20–30% độ dài bản thảo chính bằng cách đẩy các nội dung liệt kê ở Phần 3 sang một file Supplementary Material riêng, **mà không làm mất bất kỳ số liệu nào đang được văn bản chính trích dẫn tường minh**, không làm hỏng cross-reference, và không làm gãy mạch lập luận.

---

## 0. Trước khi bắt đầu

1. Tạo bản backup/commit git của toàn bộ thư mục nguồn trước khi sửa bất kỳ dòng nào.
2. Compile bản gốc trước, lưu lại PDF gốc để đối chiếu (đếm số trang, số từ — dùng `texcount main.tex` nếu có, hoặc đếm qua PDF).
3. Xác định kiến trúc nguồn:
   - Nếu LaTeX: xác nhận class là `elsarticle` (tên file gốc là `elsarticle-template-harv`), citation style Harvard (`\usepackage{harvard}` hoặc `natbib` với `agsm`/`dcu`).
   - Nếu Word (.docx): áp dụng cùng nguyên tắc ở Phần 1–3 nhưng thay thao tác LaTeX bằng: cắt nội dung → dán sang file `Supplementary.docx` riêng → chèn cross-reference bằng field `REF` hoặc ghi chú thủ công "(xem Supplementary Table S1)".
4. Không tự đặt tên `\label{}` mới trùng với label đã tồn tại. Nếu không chắc tên label hiện có, dùng `\caption{...}` (nội dung caption gần như nguyên văn, liệt kê ở Phần 3 dưới đây) để định vị đúng bảng/hình cần thao tác, tránh nhầm.

---

## 1. QUY TRÌNH BẮT BUỘC — áp dụng cho MỌI hạng mục ở Phần 3

Không được xóa/di chuyển bất kỳ bảng, hình, hay đoạn văn nào nếu chưa hoàn thành đủ 5 bước sau, theo đúng thứ tự:

### Bước 1 — LOCATE (định vị)
Tìm chính xác khối nội dung cần chuyển bằng caption/tiêu đề gần nguyên văn được cung cấp ở Phần 3. Ghi lại `\label{...}` gắn với nó (nếu có).

### Bước 2 — GREP-AUDIT (bắt buộc, quan trọng nhất)
Grep toàn bộ file .tex (kể cả Abstract, Highlights, Introduction, Discussion, Conclusion) tìm:
- Mọi con số cụ thể xuất hiện trong bảng/hình sắp chuyển (ví dụ nếu bảng có ô "56.35%", grep `56.35` trong toàn văn).
- Mọi tên phương pháp/baseline được nêu trong bảng đó (ví dụ "Zero-DCE++", "RetinexNet", "YOLO11n") — kiểm tra xem có đoạn văn nào đang dùng tên đó kèm số liệu cụ thể để lập luận không.
- Mọi chỗ gọi `\ref{...}` tới label của bảng/hình đó.
- **Riêng biệt quan trọng**: grep cụm `Table [0-9]` và `Fig\(ure\)\? [0-9]` dạng chữ số cứng (hardcoded) thay vì `\ref{}` — nếu tác giả gốc gõ tay "Table 4" thay vì `\ref{tab:mqtone}`, việc đánh số lại do xóa/di chuyển bảng phía trước sẽ khiến các câu này trỏ sai bảng mà không có lỗi compile nào cảnh báo. Đây là lỗi âm thầm nguy hiểm nhất — phải rà thủ công toàn bộ danh sách kết quả grep này.

### Bước 3 — DECIDE (quyết định xử lý từng câu bị ảnh hưởng)
Với mỗi câu văn tìm thấy ở Bước 2 mà phụ thuộc vào nội dung sắp chuyển, chọn đúng MỘT trong hai cách:
- **(A) Giữ nguyên số liệu trong câu văn**, chỉ thêm hậu tố dẫn nguồn, ví dụ: "...falls below the unenhanced baseline under severe overexposure (56.35% vs. 61.90%; full comparison in Supplementary Table S1)."
- **(B) Viết lại câu văn ở dạng tổng quát hơn**, bỏ số liệu cụ thể, chỉ giữ kết luận định tính, ví dụ: "several low-light curve estimators underperform under severe overexposure (see Supplementary Table S1 for full numeric comparison)."

Ưu tiên (A) cho các câu là lập luận cốt lõi của bài (vd. so sánh MQTone với đối thủ mạnh nhất); dùng (B) cho các câu liệt kê phụ.

### Bước 4 — EDIT (thực hiện)
- Copy nguyên khối `\begin{table}...\end{table}` hoặc `\begin{figure}...\end{figure}` sang `supplementary.tex`.
- Đổi `\label{tab:xxx}` gốc thành `\label{tab:S-xxx}` (thêm tiền tố S) để tránh trùng label giữa 2 file nếu sau này gộp compile chung hoặc dùng gói `xr` để tham chiếu chéo.
- Trong `main.tex`, tại vị trí bảng cũ: hoặc (i) xóa hẳn + chèn câu dẫn "(see Supplementary Table Sx)" tại đúng chỗ logic văn bản đang nhắc tới nó, hoặc (ii) thay bằng bảng rút gọn (xem hướng dẫn riêng từng mục ở Phần 3).
- Không xóa `\label{}` nếu label đó còn được `\ref{}` ở nơi khác trong `main.tex` — trong trường hợp cần giữ nhưng nội dung đã chuyển đi, cân nhắc dùng gói `xr` (`\usepackage{xr}` + `\externaldocument{supplementary}`) để `\ref{}` từ main.tex vẫn trỏ đúng sang số bảng bên supplementary.tex nếu muốn liên kết số tự động; nếu không dùng `xr`, ghi tay "Supplementary Table Sx" (không dùng `\ref`).

### Bước 5 — VERIFY (bắt buộc trước khi coi là xong)
- Compile lại cả `main.tex` và `supplementary.tex` — không được có warning "undefined reference" hoặc "multiply defined labels".
- Đọc lại toàn bộ đoạn văn xung quanh mỗi chỗ vừa sửa — kiểm tra mạch văn còn trôi chảy, không còn câu cụt hoặc câu nhắc "as shown below" trỏ vào chỗ trống.
- Đối chiếu lại danh sách Bước 2: mọi con số/tên phương pháp tìm được đã có xử lý (A) hoặc (B), không sót cái nào.

---

## 2. Kiến trúc file Supplementary

Tạo `supplementary.tex` với preamble tương tự `main.tex` (cùng class `elsarticle`, cùng gói citation), nhưng đổi hệ đánh số sang tiền tố S:

```latex
\documentclass[review,harvard]{elsarticle} % giữ đúng option như bản chính
\usepackage{...} % copy đúng các gói main.tex đang dùng cho bảng/hình

\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\theequation}{S\arabic{equation}}
\setcounter{table}{0}
\setcounter{figure}{0}
\setcounter{equation}{0}

\begin{document}
\title{Supplementary Material for: CashVision: An Energy-Aware Cascade for
Polymer Banknote Recognition and Tear Detection under Adverse Illumination}
\author{Anonymous Authors} % giữ ẩn danh giống bản chính, KHÔNG điền tên thật ở giai đoạn review

\maketitle

% Mỗi bảng/hình chuyển sang đây PHẢI có 1 câu caption-note ngay dưới caption gốc,
% ghi rõ nó hỗ trợ mục nào ở bản chính, ví dụ:
% "This table supports the discussion in Section 3.3 of the main text;
%  the main text (Table 4) reports the YOLOv8n subset only."

\end{document}
```

Quy tắc đặt tên: Table 4 (bản chính) → nếu tách phần mở rộng, gọi là **Table S1**; Table 6 mở rộng → **Table S2**; v.v. Đánh số S liên tục theo thứ tự xuất hiện trong supplementary.tex, không theo số thứ tự bảng gốc ở bản chính (để tránh gây hiểu lầm S ứng với đúng số bảng cũ).

---

## 3. Chi tiết từng hạng mục cần xử lý

### 3.1 Table 4 — "Cross-condition benchmark of photometric restoration methods under Protocol B..." (Panel A + Panel B)

**Rủi ro cao nhất trong toàn bộ việc này** — Section 3.3 trích số liệu từ gần như mọi ô trong bảng. Danh sách các câu đã xác định cần xử lý (không loại trừ còn sót, vẫn phải grep lại theo Bước 2):

- "MQTone attains the highest overall mean accuracy on both backbones (79.58% on YOLOv8n, 85.01% on YOLO11n)..." — trích cả 2 backbone.
- "...+1.59 and +1.39 pp over Gamma Correction, the strongest competitor."
- "Zero-DCE++ falls below the unenhanced baseline... (56.35% and 60.50% vs. 61.90% on YOLOv8n)"
- "...IAT, RetinexNet, EnlightenGAN, Afifi et al.; 91k–10.2M parameters) collapse... 26.94%–54.69% on YOLOv8n and 35.68%–56.60% on YOLO11n"
- "Afifi et al... strongest of the four under severe overexposure after fine-tuning (65.10% and 71.90%)..."
- Đoạn "MQTone vs. Gamma Correction": "(70.03% vs. 69.63% on YOLOv8n; 78.10% vs. 77.71% on YOLO11n)", "(Gamma: 82.63% on YOLO11n... MQTone: 86.43%, +3.80 pp over Gamma, and +2.70 pp on YOLOv8n)"
- Đoạn "Defect localization": "RetinexNet (47.50% and 47.01% from scratch; 46.88% and 45.71% fine-tuned)...", "MQTone brings no localization gain over the baseline (37.19% vs. 37.50%; 41.34% vs. 41.86%)"
- Đoạn "MQTone vs Gamma": chi tiết tham số/tốc độ ("19,686 parameters, 4.0–4.6× smaller...")

**Cách xử lý khuyến nghị** (an toàn hơn phương án cắt tự do ban đầu):
- **Giữ nguyên Panel A (YOLOv8n) đầy đủ ở bản chính** — bài đã tự gọi đây là "primary" evaluation, nên đây là bảng cốt lõi không nên cắt.
- **Chuyển toàn bộ Panel B (YOLO11n) sang Supplementary Table S1**, kèm ghi chú "Cross-backbone verification; see main-text Section 3.3 for discussion."
- Với mọi câu ở danh sách trên có trích số YOLO11n (ví dụ "85.01% on YOLO11n", "82.63% on YOLO11n", "78.10% vs. 77.71% on YOLO11n", "37.19% vs. 37.50%; 41.34% vs. 41.86%"): áp dụng cách (A) ở Bước 3 — giữ số liệu, thêm "(Supplementary Table S1)" ngay sau, KHÔNG xóa số.
- Không xóa bất kỳ hàng phương pháp nào (Zero-DCE, IAT, RetinexNet, EnlightenGAN, Afifi...) khỏi Panel A vì tất cả đều được nêu tên + số liệu cụ thể trong prose của Panel A. Nếu buộc phải rút gọn thêm, chỉ nên gộp cặp "From Scratch"/"Pretrained + Fine-Tuned" thành 1 dòng ("range: X–Y%") CHỈ với các phương pháp không bị nêu tên riêng lẻ với số liệu cụ thể trong văn bản — cần double-check qua Bước 2 cho từng phương pháp trước khi gộp.

### 3.2 Table 6 — "Rule Base Sensitivity Analysis... (32 configurations)"

- Figure 6 đã là bản tóm tắt trực quan của đúng dữ liệu này — rủi ro thấp hơn Table 4 vì phần lớn kết luận trong Section 3.5 dùng ngôn ngữ định tính ("broad, stable operational plateau", "Pareto elbow") hơn là trích từng số một.
- Vẫn cần grep các câu có số cụ thể, ví dụ: "inflating trigger rate from 16.4% up to 31.4% and session energy to 336.5 J (+44.2%)", "11.1% valuation hazard rate (4/36 sessions)", "hazard rate to 2.8%", "TTC = 3.12 s and session energy to 118.5 J" — đây đều là số xuất hiện y hệt trong Table 6, giữ nguyên trong văn bản (cách A), chỉ chuyển **bảng thô 32 dòng** sang Supplementary Table S2, không xóa các con số đã được diễn giải thành văn trong 5 đoạn (i)–(v).
- Có thể rút gọn 5 đoạn (i)–(v) hiện tại (khá dài) thành 1 đoạn tổng hợp ngắn hơn, miễn giữ đủ các số liệu đã bị các câu khác trong bài tham chiếu chéo (kiểm tra xem Section 4.4 hay Conclusion có nhắc lại số nào từ Table 6 không — grep `16.4%`, `61.3%` v.v. vì các số này trùng cả với Table 10/11, cần phân biệt nguồn).

### 3.3 Section 3.7 (Protocol C1/C2) — Table 8, Table 9, và phần thảo luận scope

- Table 9 nên **giữ nguyên ở bản chính** (gọn, không bị trích số chi tiết ở nơi khác, và là bằng chứng thuyết phục về việc VND không được app thương mại nào hỗ trợ — luận điểm này được nhắc lại ở Conclusion: "commercial assistive currency readers have historically targeted a small set of high-resource currencies" — không trích số cụ thể nên an toàn).
- Table 8 (kèm 3 footnote a/b/c rất dài) → chuyển toàn bộ sang Supplementary Table S3, **kèm cả 3 footnote nguyên văn** (đây là phần diễn giải phương pháp luận của từng hệ thống được so sánh, không thể tách rời bảng).
- Đoạn văn "Protocol C1: Literature-Normalized Comparison..." dài, có thể rút còn 3–4 câu tóm tắt: các hệ thống nào được so sánh, vì sao không so sánh trực tiếp được, và một câu định vị CashVision (accuracy/on-device/energy-aware/defect-detect) — giữ câu này vì nó là câu kết luận định vị bài, không phải số liệu thô.
- Đoạn "Scope boundary and future work" (phần liệt kê "No statistically tested superiority... No live accuracy... No cross-currency validation") nên **giữ ở bản chính**, không chuyển — đây là tuyên bố về giới hạn khoa học (limitation disclosure), reviewer/editor thường đánh giá cao và mong đợi thấy ngay trong bản chính, không nên giấu vào Supplementary.

### 3.4 Table 12 — breakdown theo từng điều kiện của video benchmark

- Bản thân bài đã ghi chú bảng này "are therefore descriptive and do not support condition-level inferential claims" → rủi ro thấp, không có lập luận thống kê nào của bài dựa vào từng con số riêng lẻ trong bảng này ngoài đoạn mô tả ngay phía trên nó.
- Grep riêng cụm "Backlight" trong đoạn văn mô tả (đã có nhắc "In directional backlighting, Cascade is correct in 4/6 sessions (66.7%...) versus 5/6 for B0 (83.3%...)" — đây là số lấy trực tiếp từ Table 12, cần giữ theo cách (A) vì đây là ngoại lệ duy nhất được bài dùng để mở hướng "future work" (adaptive τ thresholding).
- Có thể chuyển an toàn: 5/6 dòng điều kiện còn lại (Indoor, Outdoor, Overexposed, Torn Bright, Torn Clean) sang Supplementary Table S4, chỉ giữ dòng Backlight (hoặc giữ nguyên toàn bảng nếu muốn tối giản công sức — đây là mục rủi ro thấp nhất trong 8 mục, có thể làm nhanh).

### 3.5 Section 3.6.1–3.6.4 — công thức hình thức hoá (Eq. 21–27)

- Giữ nguyên Table 7 và các câu diễn giải kết quả ("Findings (i)(ii)(iii)") ở bản chính — đây là kết quả cốt lõi (Contribution liên quan đến specimen-disjoint generalization).
- Chuyển sang Supplementary: định nghĩa tập hợp hình thức (Eq. 21–24), phần mô tả chi tiết K=13 specimen clusters/K=78 census (Section 3.6.2), và mô tả đầy đủ 2 partitioning scheme (Section 3.6.3) — đây là chi tiết phục vụ reproducibility, không có văn bản nào khác trong bài trích số liệu từ chính các công thức này (chỉ trích kết quả ở Table 7), nên rủi ro thấp.
- Giữ 1 đoạn tóm tắt ngắn ở bản chính (2–3 câu): định nghĩa Leave-Specimen-Out, Paired-Tear Non-Contamination Constraint, và con số K=78 specimens tổng — vì con số 78 specimens/12 test specimens được nhắc lại ở phần thống kê Wilcoxon (Section 3.6.4) ngay sau đó, không nên xóa.

### 3.6 Chi tiết augmentation pipeline trong Section 3.1 (Protocol B)

- Đoạn (i)-(iv) liệt kê range tham số cụ thể (Gaussian kernel ≥15, Δbrightness ∈[+0.20,+0.35], γ∈[0.60,0.85]...) — rủi ro thấp, không có câu nào khác trong bài trích lại các con số range này.
- An toàn để chuyển toàn bộ sang Supplementary "Implementation Details", thay bằng 1 câu: "an optical degradation augmentation pipeline synthesizes glare, occlusion, and exposure artifacts specific to BOPP polymer substrates (full parameter ranges in Supplementary Material)."
- Giữ nguyên ở bản chính công thức số lượng instance (336×5+120×3=2,040...) vì con số 2,040 được nhắc lại nhiều lần sau đó ("trained identically on the same 2,040-image augmented training pool" — xuất hiện ở Table 4's note, ở Section 3.3 nhiều chỗ) — đây là con số bị tham chiếu chéo nhiều, PHẢI giữ ở bản chính.

### 3.7 Figure 5 — so sánh định tính 10 panel

- Rủi ro thấp: đoạn caption/văn bản quanh hình chỉ trích số confidence của từng panel trong chính caption (không có đoạn prose riêng nào khác trích lại các số này).
- Có thể rút gọn ảnh còn 4-5 panel (Input, Gamma, RetinexNet hoặc phương pháp có confidence cao nhất ngoài MQTone, MQTone) — giữ nguyên caption format nhưng chỉ liệt kê panel còn giữ; chuyển bản đầy đủ 10 panel sang Supplementary Figure S1 với caption đầy đủ gốc.
- Lưu ý: nếu dùng subfigure trong LaTeX (`\subfloat` hoặc `subcaption`), phải xóa cả phần code định nghĩa subfigure của các panel bị bỏ, không chỉ ẩn ảnh, để tránh lỗi compile do thiếu file ảnh nếu ảnh gốc cũng bị dọn khỏi thư mục.

### 3.8 Table 3 — chỉ giữ YOLOv8n, đẩy YOLOv8s/YOLO11n

- Câu cần xử lý: "all architectures exhibit high recognition accuracy (99.67% for YOLOv8n and YOLOv8s, 100.0% for YOLO11n)" — giữ nguyên cả 3 số (cách A), vì đây là câu mở đầu thiết lập baseline chung.
- "accuracy drops by −9.54% on YOLOv8n and −10.26% on YOLO11n" — giữ nguyên, thêm "(Supplementary Table S5)" sau vế YOLO11n.
- Các đoạn còn lại của Section 3.2 (Asymmetric Task Vulnerability Analysis) chỉ trích YOLOv8n → an toàn, không cần sửa.
- Chuyển 2 hàng model (YOLOv8s, YOLO11n) của Table 3 sang Supplementary Table S5, giữ hàng YOLOv8n ở bản chính.

---

## 4. Checklist cuối cùng trước khi coi là hoàn tất

- [ ] Compile `main.tex` — không có warning "undefined reference" / "multiply defined labels" / "citation undefined".
- [ ] Compile `supplementary.tex` riêng — thành công, không lỗi.
- [ ] Grep lại toàn bộ `main.tex` tìm mọi số liệu phần trăm (`%`), tham số (`params`), hoặc tên phương pháp còn sót mà không xuất hiện trong bất kỳ bảng/hình nào của bản chính lẫn không có "(Supplementary...)" đi kèm — đây là dấu hiệu số liệu "mồ côi", phải sửa.
- [ ] Grep `Table [0-9]` và `Fig\(ure\)\? [0-9]` dạng chữ số cứng — xác nhận không còn tham chiếu tay bị lệch số sau khi bảng/hình bị xóa/di chuyển làm renumber.
- [ ] Đọc lại một lượt toàn bộ Section 3 và 4 của bản chính từ đầu đến cuối — kiểm tra mạch văn tự nhiên, không có câu cụt hay đoạn "as shown below" trỏ vào khoảng trống.
- [ ] Kiểm tra Highlights, Abstract, Graphical Abstract — các số liệu tóm tắt ở đây (vd. accuracy, FPS, SUS score) không phụ thuộc bảng nào bị chuyển đi (thường các số này đã có ở Table 10/13/14 — vẫn giữ nguyên ở bản chính nên an toàn, chỉ cần xác nhận lại).
- [ ] Supplementary.tex có tiêu đề "Anonymous Authors" giống bản chính — không lộ danh tính (nhất quán với yêu cầu double-anonymized review đã xử lý trước đó).
- [ ] Mỗi bảng/hình trong Supplementary có 1 dòng ghi chú nêu rõ nó hỗ trợ mục nào ở bản chính, để reviewer dễ tra cứu qua lại.
- [ ] Đo lại độ dài bản chính sau khi sửa (số trang/số từ) — xác nhận đã giảm so với bản gốc đã lưu ở Bước 0.
- [ ] So sánh PDF mới với PDF gốc (đối chiếu thủ công 1 lượt) để đảm bảo không mất nội dung nào ngoài ý muốn.
