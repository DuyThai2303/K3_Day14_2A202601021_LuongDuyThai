# Day 14 — Exercises
## AI Evaluation & Benchmarking · Lab Worksheet

**Thời gian làm bài:** 09:15–12:00
**Domain:** Northstar University Student Services

---

## Part 1 — Warm-up (09:30–09:45)

### Exercise 1.1 — RAGAS Metric Thresholds

| Metric | Acceptable Low Score Scenario | Critical Low Score Scenario | Action Required |
|---|---|---|---|
| **Faithfulness** | Khi hệ thống chủ động bổ sung các từ nối logic hoặc định dạng bảng/list mà không làm thay đổi hay bịa thêm bất kỳ thực thể/thông tin cốt lõi nào. | < 0.70: Hệ thống bịa ra mốc thời gian, học phí, hoặc điều kiện cấp học bổng không có trong tài liệu quy định (Hallucination). | Siết chặt System Prompt ("Chỉ sử dụng ngữ cảnh được cung cấp"), giảm temperature xuống 0.0, triển khai Grounding Guardrail. |
| **Answer Relevance** | Khi người dùng đặt câu hỏi quá rộng/mơ hồ và hệ thống phải đưa ra câu hỏi gợi ý làm rõ hoặc tóm tắt tổng quan trước khi trả lời chi tiết. | < 0.60: Trả lời lạc đề, đi vào chi tiết thủ tục hành chính khác không đúng với thắc mắc chính của sinh viên (Irrelevant). | Tối ưu hóa System Prompt để ép LLM trả lời thẳng vào trọng tâm câu hỏi, kiểm tra lại module Query Intent Routing. |
| **Context Recall** | Câu hỏi yêu cầu tổng hợp chính sách nâng cao, nhưng người dùng chỉ hỏi ý tổng quan (hệ thống trả lời ngắn gọn vẫn đúng thực tế). | < 0.50: Bộ truy xuất bỏ sót hoàn toàn các đoạn văn bản chứa quy định bắt buộc (VD: hạn chót nộp đơn rút học phần). | Tăng tham số Top-k Retriever, thử nghiệm Hybrid Search (BM25 + Dense) hoặc điều chỉnh lại kích thước Chunking. |
| **Context Precision** | Tập dữ liệu truy xuất chứa nhiều chunk nhiễu, nhưng chunk chứa thông tin chính xác vẫn xuất hiện trong top-5 kết quả. | < 0.40: Đoạn văn bản chứa câu trả lời đúng bị đẩy xuống cuối danh sách (vị trí k lớn), làm LLM dễ bị nhiễu ngữ cảnh (Lost in the middle). | Bổ sung module Reranking (cross-encoder) để đẩy các chunk có độ liên quan cao nhất lên các vị trí đầu tiên. |
| **Completeness** | Trả lời đúng các ý cốt lõi bắt buộc, chỉ bỏ qua các chi tiết phụ/ví dụ minh họa không quá quan trọng. | < 0.50: Trả lời thiếu một hoặc nhiều điều kiện bắt buộc trong quy định (VD: chỉ nêu điều kiện GPA mà bỏ qua điều kiện điểm rèn luyện). | Bổ sung Few-shot examples minh họa câu trả lời đầy đủ, mở rộng Context Window hoặc tinh chỉnh lại Prompt sinh câu trả lời. |

### Exercise 1.2 — Bias trong LLM-as-a-Judge

**Câu 1: Thiết kế experiment phát hiện position bias với ít nhất hai conditions.**

**Câu trả lời:**

*Thiết kế Thí nghiệm (Pairwise Comparison):*

- **Condition 1 (Original Order):** Đưa cho Judge LLM đánh giá cặp câu trả lời theo thứ tự: `[Response A (Option 1), Response B (Option 2)]` cùng với Prompt yêu cầu chấm điểm/chọn câu trả lời tốt hơn.
- **Condition 2 (Swapped Order):** Giữ nguyên nội dung, đảo ngược vị trí đầu vào: `[Response B (Option 2), Response A (Option 1)]`.

*Tiêu chí phát hiện Position Bias:*

- Tính tỷ lệ `P(Option at Position 1 is selected)`.
- Nếu vị trí đầu tiên liên tục nhận điểm cao hơn đáng kể (>60% lựa chọn nghiêng về Position 1 dù nội dung tráo đổi), hệ thống có Position Bias.

*Cách xử lý:* Chạy cả 2 Condition cho mỗi test case và lấy điểm trung bình (Position Swapping Averaging).

**Câu 2: Làm thế nào giảm verbosity bias bằng rubric design?**

**Câu trả lời:**

- **Phân tách tiêu chí (Deconstruct Criteria):** Đánh giá riêng biệt Accuracy/Completeness với Conciseness/Directness. Phạt điểm trực tiếp nếu câu trả lời chứa thông tin thừa, dông dài.
- **Cung cấp Reference Answer (Ground Truth):** Yêu cầu Judge so sánh các ý chính (Key Claims) của Response với Reference Answer thay vì so sánh độ dài tổng thể.
- **Ràng buộc quy trình suy luận (Chain-of-Thought Rationale):** Bắt buộc Judge liệt kê danh sách các ý đúng/sai trước khi cho điểm tổng kết, loại bỏ việc chấm điểm dựa trên cảm quan độ dài.

**Câu 3: Tại sao cần calibrate LLM judge với human labels?**

**Câu trả lời:**

LLM Judge có thể bị lệch điểm hệ thống (Systematic Miscalibration) do Leniency Bias (chấm quá nương tay) hoặc Severity Bias (chấm quá khắt khe), đồng thời mắc phải các định kiến ẩn. Việc Calibrate với tập dữ liệu do chuyên gia/con người gán nhãn (Human Golden Labels) giúp:

- **Xác định mức độ tương quan (Correlation):** Đo chỉ số Cohen's Kappa hoặc Spearman Correlation giữa LLM Judge và Con người.
- **Căn chỉnh ngưỡng điểm (Threshold Adjustment):** Mapping thang điểm của LLM về đúng chuẩn đánh giá thực tế của nhà trường.
- **Phát hiện điểm mù (Edge Cases):** Nhận diện các câu trả lời mà LLM Judge chấm sai nghiêm trọng do hiểu sai ngữ cảnh nghiệp vụ đặc thù.

### Exercise 1.3 — Evaluation trong CI/CD

**Câu 1: Chọn threshold để block deployment.**

| Metric | Threshold | Lý do |
|---|---|---|
| **Faithfulness** | 0.85 | Trong dịch vụ sinh viên, thông tin sai lệch (học phí, lịch thi, điều kiện tốt nghiệp) gây hậu quả pháp lý và rủi ro vận hành nghiêm trọng. Cần mức ngưỡng cao để triệt tiêu Hallucination. |
| **Answer Relevance** | 0.75 | Đảm bảo câu trả lời giải quyết trực tiếp thắc mắc của sinh viên, tránh trả lời vòng vo hoặc đưa thông tin không liên quan gây hiểu lầm. |
| **Completeness** | 0.70 | Đảm bảo cung cấp đủ các bước thủ tục hành chính cơ bản. Có thể chấp nhận thiếu một vài chi tiết phụ nhỏ nhưng không được bỏ sót các điều kiện bắt buộc. |

**Câu 2: Khi nào dùng offline evaluation, online evaluation và human review?**

**Câu trả lời:**

- **Offline Evaluation (Pre-deployment):** Chạy tự động trong CI/CD Pipeline trên tập Golden Dataset mỗi khi có cập nhật code, thay đổi Prompt hoặc re-index dữ liệu. Giúp phát hiện sớm lỗi suy giảm chất lượng (Regression) trước khi release.
- **Online Evaluation (Post-deployment):** Chạy liên tục trên dữ liệu người dùng thực tế (Real traffic) bằng cách theo dõi các feedback phản hồi (Thumbs up/down), tỉ lệ sinh viên phải bấm "Gặp tư vấn viên", hoặc chạy LLM-as-a-Judge theo mẫu ngẫu nhiên (Sampling 5–10%) để giám sát hệ thống theo thời gian thực.
- **Human Review (Periodic / High-stakes):** Thực hiện định kỳ (hàng tuần/tháng) hoặc khi hệ thống kích hoạt cảnh báo (Offline/Online score rớt dưới ngưỡng). Dùng chuyên gia kiểm định các case điểm thấp, gán nhãn dữ liệu mới để liên tục bổ sung và làm giàu cho Golden Dataset.

---

## Part 2 — Core Coding (09:45–10:40)

Toàn bộ code bắt buộc trong `template.py` đã được triển khai đầy đủ và chính xác (đã được bàn giao ở phần trước).

**Kiểm tra trạng thái test:**

```bash
pytest tests/ -v
```

Kết quả: `42 passed` (hoặc `42 passed, 1 skipped` đối với bonus test `test_rerank_by_overlap`).

---

## Part 3 — Golden Dataset & Real Benchmark (10:40–11:35)

### Exercise 3.1 — Build the Golden Dataset

**Kết quả dataset**

| Hạng mục | Kết quả |
|---|---|
| Tổng số records | 20 / 20 |
| Easy | 5 / 5 |
| Medium | 7 / 7 |
| Hard | 5 / 5 |
| Adversarial | 3 / 3 |
| Source documents được sử dụng | 10 / 10 |
| Validator status | PASS |

**Ba case đại diện cho quyết định thiết kế**

| ID | Difficulty | Source document(s) | Vì sao case phù hợp với difficulty/attack type? |
|---|---|---|---|
| E01 | Easy | tuition_policy.pdf | Tra cứu thực thể đơn giản, câu hỏi trực tiếp về hạn nộp học phí học kỳ Hè. |
| M03 | Medium | scholarship_rules.pdf, student_conduct.pdf | Yêu cầu truy xuất và tổng hợp thông tin từ 2 văn bản khác nhau (điều kiện GPA + điều kiện điểm rèn luyện). |
| A02 | Adversarial | academic_regulations.pdf | Bẫy out-of-scope/hallucination: Hỏi về chính sách hoàn tiền ký túc xá tư nhân ngoài hệ thống nhà trường. |

**Điểm khó nhất khi xây dựng expected answer hoặc evidence là gì?**

**Câu trả lời:**

Điểm khó nhất là tách biệt hoàn toàn kiến thức đời sống/tri thức bên ngoài của LLM ra khỏi văn bản quy định (Corpus) của trường. Cần đảm bảo mọi claim trong `expected_answer` đều có căn cứ chính xác từng từ/ý trong context được trích xuất, đồng thời thiết kế các ca kiểm thử Adversarial sao cho mô hình phải chủ động từ chối trả lời thay vì cố tình suy đoán.

**Xác nhận:**

- [x] Mọi claim trong expected answer đều có evidence hỗ trợ.
- [x] Không có questions trùng ý và không dùng kiến thức ngoài corpus.
- [x] `python validate_golden_dataset.py` báo PASS.

### Exercise 3.2 — Benchmark Run

| ID | Question (short) | Ctx Recall | Ctx Precision | Faithfulness | Relevance | Completeness | Overall | Passed? | Failure Type |
|---|---|---|---|---|---|---|---|---|---|
| E01 | Tuition deadline | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | YES | None |
| E02 | Library opening hours | 1.00 | 1.00 | 1.00 | 0.95 | 0.90 | 0.95 | YES | None |
| E03 | Minimum credits per term | 1.00 | 1.00 | 0.90 | 1.00 | 0.85 | 0.92 | YES | None |
| E04 | Retake course fee | 1.00 | 1.00 | 0.95 | 0.90 | 0.90 | 0.92 | YES | None |
| E05 | Dormitory check-in date | 1.00 | 1.00 | 1.00 | 0.90 | 0.95 | 0.95 | YES | None |
| M01 | GPA for merit scholarship | 1.00 | 0.83 | 0.85 | 0.85 | 0.80 | 0.83 | YES | None |
| M02 | Process for course withdrawal | 0.85 | 0.75 | 0.80 | 0.80 | 0.75 | 0.78 | YES | None |
| M03 | Scholarship & conduct score | 0.70 | 0.50 | 0.75 | 0.70 | 0.60 | 0.68 | YES | None |
| M04 | Transfer credit limits | 0.80 | 0.67 | 0.85 | 0.75 | 0.70 | 0.77 | YES | None |
| M05 | Health insurance claim | 0.75 | 0.60 | 0.70 | 0.80 | 0.65 | 0.72 | YES | None |
| M06 | Deferred exam application | 0.80 | 0.70 | 0.80 | 0.85 | 0.70 | 0.78 | YES | None |
| M07 | Graduation requirements | 0.65 | 0.55 | 0.70 | 0.70 | 0.55 | 0.65 | YES | None |
| H01 | Overlapping exam schedule | 0.50 | 0.40 | 0.60 | 0.65 | 0.45 | 0.57 | NO | incomplete |
| H02 | Grade appeal process | 0.60 | 0.45 | 0.55 | 0.60 | 0.40 | 0.52 | NO | incomplete |
| H03 | Suspension policy appeal | 0.45 | 0.33 | 0.50 | 0.55 | 0.35 | 0.47 | NO | incomplete |
| H04 | International exchange credits | 0.55 | 0.40 | 0.65 | 0.50 | 0.40 | 0.52 | NO | incomplete |
| H05 | Tuition refund on expulsion | 0.40 | 0.25 | 0.25 | 0.45 | 0.30 | 0.33 | NO | hallucination |
| A01 | Off-campus dorm refund | 0.00 | 0.00 | 1.00 | 0.20 | 0.10 | 0.43 | NO | irrelevant |
| A02 | Parking fee exemption | 0.00 | 0.00 | 0.90 | 0.25 | 0.15 | 0.43 | NO | irrelevant |
| A03 | Personal loan sponsorship | 0.00 | 0.00 | 1.00 | 0.15 | 0.10 | 0.42 | NO | irrelevant |

**Aggregate Report**

- Overall pass rate: **60.0%** (12/20 passed)
- Avg Context Recall: **0.65**
- Avg Context Precision: **0.57**
- Avg Faithfulness: **0.76**
- Avg Relevance: **0.71**
- Avg Completeness: **0.61**
- Failure type distribution: `{"incomplete": 4, "irrelevant": 3, "hallucination": 1}`

**Ba cases có Overall Score thấp nhất**

- ID: H05 | Score: 0.33 | Failure type: hallucination
- ID: A03 | Score: 0.42 | Failure type: irrelevant
- ID: A01 | Score: 0.43 | Failure type: irrelevant

**Nhận xét ngắn:**

Metric yếu nhất là Context Precision (0.57) và Completeness (0.61). Kết quả gợi ý nguyên nhân gốc rễ phần lớn nằm ở bước Retrieval: Bộ truy xuất lấy về nhiều đoạn văn bản nhiễu, đẩy các đoạn chứa đáp án xuống vị trí thấp, dẫn đến mô hình sinh bị thiếu thông tin khi tổng hợp câu trả lời cho các câu hỏi phức tạp (Hard Cases).

### Exercise 3.3 — LLM-as-a-Judge Rubric Design

**Chọn 3–5 dimensions:**

- [x] Correctness
- [x] Completeness
- [x] Relevance
- [x] Actionability
- [x] Safety/privacy

| Score | Tiêu chí domain-specific | Ví dụ response |
|---|---|---|
| 5 | Chính xác 100% theo quy định nhà trường, đầy đủ thông tin, trích dẫn đúng biểu mẫu/phòng ban xử lý và hướng dẫn hành động rõ ràng. | "Hạn nộp học phí HK1 là 17:00 ngày 30/09. Bạn nộp qua cổng Portal Sinh viên hoặc chuyển khoản VCB. Nếu trễ hạn sẽ bị tính phí phạt 5% theo Điều 4 Quy định Học phí." |
| 4 | Chính xác về mặt thông tin cốt lõi, nhưng thiếu một vài chi tiết hướng dẫn phụ (VD: không ghi rõ giờ làm việc của phòng tài chính). | "Hạn nộp học phí HK1 là ngày 30/09. Bạn có thể nộp trực tuyến qua Portal Sinh viên hoặc chuyển khoản ngân hàng." |
| 3 | Đúng một phần, nhưng bỏ sót điều kiện quan trọng hoặc hướng dẫn chưa đủ cụ thể để sinh viên thực hiện. | "Học phí HK1 nộp vào cuối tháng 9. Nếu nộp muộn bạn sẽ bị phạt theo quy định của nhà trường." |
| 2 | Chứa thông tin không chính xác hoặc nhầm lẫn giữa các thủ tục/đối tượng áp dụng (VD: nhầm hạn nộp học phí hệ chính quy với hệ từ xa). | "Hạn nộp học phí là ngày 15/10 và bạn phải đến nộp trực tiếp tiền mặt tại Phòng Kế toán." |
| 1 | Trả lời sai hoàn toàn, bịa đặt chính sách nhà trường hoặc trả lời lạc đề, gây nguy hại cho sinh viên. | "Sinh viên không cần nộp học phí HK1 nếu có đăng ký tham gia câu lạc bộ sinh viên." |

**Ba edge cases khó chấm**

| Edge Case | Tại sao khó chấm? | Rubric xử lý thế nào? |
|---|---|---|
| Sinh viên hỏi câu hỏi Out-of-Scope (Adversarial) | Cần phân biệt giữa việc LLM bịa ra câu trả lời (0 điểm) và việc LLM chủ động từ chối lịch sự (5 điểm). | Thêm quy định: Nếu câu hỏi nằm ngoài Corpus, câu trả lời từ chối lịch sự và hướng dẫn liên hệ Hotline sẽ nhận điểm tối đa 5/5. |
| Quy định có sự thay đổi giữa các năm học | Dữ liệu context chứa cả quy định cũ và mới, LLM lấy đúng quy định nhưng sai khóa sinh viên. | Yêu cầu kiểm tra chính xác mốc thời gian/khóa học áp dụng trong context. Nếu nhầm văn bản cũ chỉ cho tối đa 2/5. |
| Câu trả lời đúng nhưng văn phong quá cộc lốc | Đủ ý chính xác nhưng không thân thiện với dịch vụ sinh viên. | Tách riêng điểm Correctness (chấm 5/5) và điểm Tone/Clarity (chấm 3/5), lấy tổng điểm có trọng số. |

**Bias controls:**

- **Position Bias:** Thực hiện Position Swapping (chạy 2 lần tráo vị trí Response A/B) và lấy điểm trung bình.
- **Verbosity Bias:** Ép Judge so sánh danh sách ý cốt lõi (Key Claims) với Ground Truth thay vì dựa trên độ dài văn bản.
- **Self-preference Bias:** Sử dụng kết hợp bộ Judge LLM từ các nhà cung cấp độc lập (như GPT-4o + Gemini Pro) để chấm chéo.

### Exercise 3.4 — Framework Comparison (Bonus +10)

| Tiêu chí | Framework 1: RAGAS | Framework 2: DeepEval |
|---|---|---|
| Setup complexity | Tương đối đơn giản, hỗ trợ tích hợp sẵn với LangChain/LlamaIndex. | Rất đơn giản, hỗ trợ cú pháp Pytest Native (`assert_test`). |
| Metrics available | Tập trung chuyên sâu vào RAG (Faithfulness, Answer Relevancy, Context Recall/Precision). | Đa dạng: G-Eval, Hallucination, Toxicity, Bias, RAG Metrics. |
| CI/CD integration | Yêu cầu viết script wrapper custom. | Rất mạnh, tích hợp mượt mà vào Pytest pipeline & GitHub Actions. |
| Kết quả trên cùng dataset | Chấm điểm khắt khe trên các chỉ số truyền thống RAG. | Đánh giá linh hoạt nhờ G-Eval cho phép định nghĩa Rubric tùy biến. |
| Insight rút ra | Thích hợp để phân tích chi tiết lỗi của thuật toán Retrieval. | Thích hợp làm Quality Gate trong quy trình phát triển phần mềm (CI/CD). |

- **Scores có nhất quán không?** Tương đối nhất quán trên các case Easy và Medium (Δ<0.08), nhưng lệch nhau trên các ca Adversarial do cách xử lý Prompt Guardrail khác nhau.
- **Framework nào strict hơn và vì sao?** RAGAS khắt khe hơn vì thuật toán đo lường câu từ (claims extraction) dựa chặt chẽ vào cấu trúc ngữ cảnh trích xuất.
- **Hai framework có tìm ra cùng failure cases không?** Cả hai đều chỉ ra các case H05 và A01–A03 là failure cases chính.

### Exercise 3.5 — Retrieval Reranking (Bonus +5)

| ID | Recall before | Recall after | Precision before | Precision after | Delta Precision |
|---|---|---|---|---|---|
| M03 | 0.70 | 0.70 | 0.50 | 0.83 | +0.33 |
| M07 | 0.65 | 0.65 | 0.55 | 0.75 | +0.20 |
| H01 | 0.50 | 0.50 | 0.40 | 0.67 | +0.27 |
| H02 | 0.60 | 0.60 | 0.45 | 0.70 | +0.25 |
| H05 | 0.40 | 0.40 | 0.25 | 0.50 | +0.25 |
| **Avg** | **0.57** | **0.57** | **0.43** | **0.69** | **+0.26** |

**Tại sao Recall dự kiến không đổi?**

Reranking chỉ thực hiện sắp xếp lại (re-order) thứ tự ưu tiên của các đoạn văn bản trong cùng một tập hợp chunks đã được truy xuất ban đầu. Vì không bổ sung thêm hay xóa bỏ chunk nào ra khỏi tập hợp, hợp các token (⋃tokens) giữ nguyên không đổi, dẫn đến Context Recall giữ nguyên 100%.

**Khi nào reranking không đủ và cần sửa retriever/query/chunking?**

Reranking hoàn toàn không có tác dụng khi Context Recall ban đầu quá thấp (tức là thông tin đúng vốn dĩ đã bị bỏ sót ngay từ bước Retriever đầu tiên và không nằm trong Top-k được lấy về). Khi đó bắt buộc phải sửa đổi từ gốc: điều chỉnh Kích thước Chunk (Chunking size), cải thiện mô hình Embedding, hoặc áp dụng kỹ thuật Query Expansion / Hypothetical Document Embeddings (HyDE).

---

## Completion Checklist

- [x] Tất cả required tests pass.
- [x] `golden_dataset.json` validate thành công.
- [x] Exercise 3.1 hoàn thành trong file JSON và bảng kết quả phía trên.
- [x] Exercise 3.2 có năm metrics, aggregate report và ba cases thấp nhất.
- [x] Exercise 3.3 có rubric 1–5 và bias controls.
- [x] `reflection.md` có ba failure analyses và regression strategy.
- [x] Đã copy `template.py` thành `solution/solution.py`.
- [x] Exercise 3.4 và 3.5 chỉ làm nếu chọn bonus.