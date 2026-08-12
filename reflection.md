# 📄 Evaluation Summary & Failure Reflection Report (Checkpoint CP4)

**Project:** Northstar Student Services RAG Evaluation  
**Dataset:** `golden_dataset.json` (20 QA Pairs)  
**Evaluator Run Date:** 2026-08-12  
**Target Model:** Gemini 1.5 Flash (`gemini-flash-latest`)  

---

## 1. Evaluation Summary Report (Báo cáo tổng hợp)

* **Overall Pass Rate:** `0.0%` (0/20 cases passed standard strict criteria)
* **Avg Context Recall:** `0.700` (70.0%)
* **Avg Context Precision:** `0.000` (0.0%)
* **Avg Faithfulness:** `0.000` (0.0%)
* **Avg Relevance:** `0.553` (55.3%)
* **Avg Completeness:** `0.700` (70.0%)
* **Dominant Failure Type:** `hallucination` (20/20 cases)

### Nhận xét chung:
1. **Retrieval Performance (BM25):** Bộ truy xuất BM25 hoạt động khá tốt trên các câu hỏi Easy và Multi-context, đạt **Context Recall 70%**. Tuy nhiên, Context Precision bị 0% do danh sách Chunk trả về ($k=5$) chứa nhiều đoạn nhiễu không nằm trong Gold Context.
2. **Generation & Faithfulness:** Gemini sinh câu trả lời mượt mà, đúng trọng tâm (**Completeness 70%**, **Relevance 55.3%**). Dù vậy, do LLM diễn giải bằng ngôn ngữ tự nhiên (Paraphrasing) thay vì trích dẫn nguyên văn (Verbatim Quotes) theo kỳ vọng kiểm tra của Evaluator, chỉ số **Faithfulness** bị đánh giá về 0.000.

---

## 2. 5 Whys Root Cause Analysis (3 Case điểm thấp nhất)

Dựa trên kết quả benchmark từ `evaluate_answers.py`, 3 case có điểm Overall thấp nhất là **M01**, **M05**, và **M04**.

---

### 🔍 Case 1: M01 — Attendance and Grading Evaluation
* **Question:** *"How are attendance and grading evaluated?"*
* **Overall Score:** `0.167` | **Failure Type:** `hallucination`
* **Metrics breakdown:** Recall: `0.000` | Precision: `0.000` | Faithfulness: `0.000` | Relevance: `0.500` | Completeness: `0.000`

#### Phân tích 5 Whys:
1. **Why 1 (Symptom):** Tại sao điểm Overall của M01 chỉ đạt 0.167?  
   $\rightarrow$ Vì `Context Recall` và `Completeness` đều bằng 0, LLM không đưa ra thông tin chính xác về quy định điểm danh và chấm điểm.
2. **Why 2:** Tại sao LLM không đưa ra được thông tin chính xác?  
   $\rightarrow$ Vì bộ truy xuất không lấy được đúng các Chunk chứa thông tin trong file `05_attendance_and_grading.md`.
3. **Why 3:** Tại sao BM25 lại không truy xuất được đoạn văn bản thuộc file `05_attendance_and_grading.md`?  
   $\rightarrow$ Vì câu hỏi M01 dùng từ khóa chung chung ("attendance", "grading", "evaluated"), làm cho BM25 bị nhiễu do các từ này xuất hiện rải rác ở nhiều document khác.
4. **Why 4:** Tại sao thuật toán BM25 dễ bị nhiễu từ khóa chung chung?  
   $\rightarrow$ Vì hệ thống hiện tại chỉ sử dụng **Lexical Search (BM25)** đơn thuần trên đoạn văn (Paragraph level) mà chưa kết hợp Semantic Vector Search (Dense Retrieval) hoặc Hybrid Search.
5. **Why 5 (Root Cause có thể hành động):**  
   $\rightarrow$ **Thiếu chiến lược Hybrid Retrieval (BM25 + Dense Embeddings) và Prompt System chưa ép LLM trích dẫn nguyên văn nguồn tài liệu.**

* **So sánh với `find_root_cause()`:** Khớp hoàn toàn. Evaluator xác định `Retrieval Miss` dẫn đến `Generation Hallucination`.
* **Fix đề xuất:** Chuyển sang **Hybrid Search (BM25 + Sentence Transformers / OpenAI Embeddings)** với Reciprocal Rank Fusion (RRF).
* **Metric verify fix:** `Context Recall` tăng từ `0.000` lên $\ge 0.850$.

---

### 🔍 Case 2: M05 — Privacy, Security, and Policy Updates
* **Question:** *"How does the university handle privacy, security, and policy updates?"*
* **Overall Score:** `0.208` | **Failure Type:** `hallucination`
* **Metrics breakdown:** Recall: `0.000` | Precision: `0.000` | Faithfulness: `0.000` | Relevance: `0.625` | Completeness: `0.000`

#### Phân tích 5 Whys:
1. **Why 1 (Symptom):** Tại sao case M05 chỉ đạt 0.208 điểm?  
   $\rightarrow$ `Context Recall` bằng 0.000, câu trả lời thực tế bị Evaluator đánh dấu là Hallucination.
2. **Why 2:** Tại sao Context Recall lại bằng 0?  
   $\rightarrow$ BM25 không ưu tiên trích xuất file `09_privacy_security_and_policy_updates.md` vào vị trí Top-$k$.
3. **Why 3:** Tại sao file `09_privacy_security_and_policy_updates.md` không vào được Top-5?  
   $\rightarrow$ Vì câu hỏi ghép 3 khía cạnh ("privacy", "security", "policy updates") làm loãng mật độ từ khóa (TF-IDF) trên từng Chunk ngắn.
4. **Why 4:** Tại sao phỏng vấn đa khía cạnh làm giảm mật độ từ khóa trên Chunk?  
   $\rightarrow$ Chiến lược Chunking hiện tại đang chia theo Paragraph cố định quá nhỏ, làm phân tán các ý liên quan trong cùng một tài liệu.
5. **Why 5 (Root Cause có thể hành động):**  
   $\rightarrow$ **Kích thước Chunking (Paragraph level) quá mảnh và thiếu kỹ thuật Query Expansion / Sub-query Decomposition đối với các câu hỏi Medium đa ý.**

* **So sánh với `find_root_cause()`:** Khớp với đánh giá `Multi-aspect Query Failure`.
* **Fix đề xuất:** Sử dụng **Hierarchical Chunking** (Parent-Child Retriever) hoặc **Query Decomposition** (tách 1 câu hỏi phức hợp thành 3 sub-queries trước khi retrieve).
* **Metric verify fix:** `Context Recall` cho các câu Medium tăng lên $\ge 0.800$.

---

### 🔍 Case 3: M04 — Student Support and Appeals
* **Question:** *"How can students request support or file an appeal?"*
* **Overall Score:** `0.238` | **Failure Type:** `hallucination`
* **Metrics breakdown:** Recall: `0.000` | Precision: `0.000` | Faithfulness: `0.000` | Relevance: `0.714` | Completeness: `0.000`

#### Phân tích 5 Whys:
1. **Why 1 (Symptom):** Tại sao M04 nhận điểm Overall 0.238?  
   $\rightarrow$ Câu trả lời bị đánh giá không Faithfulness và thiếu tính kiểm chứng nguyên văn.
2. **Why 2:** Tại sao Faithfulness bị đánh giá 0.000 dù thông tin trả lời khá phù hợp?  
   $\rightarrow$ LLM đã diễn giải lại bằng văn phong tổng hợp (Paraphrased summary) mà không giữ lại các cụm từ nguyên bản từ tài liệu `08_student_support_and_appeals.md`.
3. **Why 3:** Tại sao LLM lại tự do diễn giải văn bản thay vì trích dẫn nguyên văn?  
   $\rightarrow$ System Prompt trong `_build_prompt` chưa yêu cầu đủ nghiêm ngặt về việc trích dẫn nguyên văn (Verbatim quotation).
4. **Why 4:** Tại sao Prompt chưa ép trích dẫn nguyên văn?  
   $\rightarrow$ Kỹ thuật Prompting hiện tại tập trung vào việc bảo mật/chống Jailbreak nhiều hơn là định dạng chuẩn output cho Verbatim Faithfulness Evaluator.
5. **Why 5 (Root Cause có thể hành động):**  
   $\rightarrow$ **Thiếu System Instructions cụ thể hướng dẫn LLM trích dẫn nguyên văn câu/đoạn bằng dấu ngoặc kép hoặc giữ nguyên câu trúc từ ngữ liệu gốc.**

* **So sánh với `find_root_cause()`:** Evaluator chỉ ra lỗi `Ungrounded Paraphrasing`.
* **Fix đề xuất:** Cập nhật System Prompt: *"For every claim, copy exact verbatim sentences from context using quotes."*
* **Metric verify fix:** `Faithfulness` tăng từ `0.000` lên $\ge 0.700$.

---

## 3. Clustered Improvements & Action Plan (Improvement Log)

Thay vì sửa đổi đơn lẻ từng câu hỏi, các thất bại được gom nhóm (clustered) thành 3 hành động cải tiến hệ thống có tác động diện rộng:

| Priority | Improvement Cluster | Root Cause Addressed | Actionable Fix Description | Target Verification Metric |
| :--- | :--- | :--- | :--- | :--- |
| **P1 (Cao nhất)** | **Prompt Grounding & Citation Format** | Faithfulness = 0.000 trên toàn bộ 20 câu hỏi do LLM diễn giải tự do. | Cập nhật Prompt trong `domain_assistant.py` yêu cầu LLM trích dẫn nguyên văn cụm từ (Verbatim quotes) từ context. | **Faithfulness** tăng từ `0.0%` $\rightarrow$ `> 75.0%`, **Pass Rate** `> 50%`. |
| **P2** | **Hybrid Retrieval & Reranking** | Context Recall bị 0.000 ở các câu hỏi Medium (M01, M04, M05). | Kết hợp BM25 + Vector Embeddings và thêm Cohere/Cross-Encoder Reranker để lọc Top-5 Chunk chuẩn xác hơn. | **Context Recall** tăng từ `70%` $\rightarrow$ `> 90%`, **Context Precision** `> 60%`. |
| **P3** | **Parent-Child Chunking Strategy** | Phân mảnh thông tin ở các câu hỏi đa khía cạnh (Multi-aspect queries). | Chuyển từ Paragraph Chunking sang Parent-Child Chunking (Retrieval trên Small Chunks, gửi Parent Chunk vào Prompt). | **Context Precision** tăng từ `0.0%` $\rightarrow$ `> 50.0%`. |

---

## 4. Regression Strategy (Lịch trình & Chiến lược chống sụt giảm chất lượng)

Để đảm bảo các cải tiến không làm tụt giảm chất lượng ở các case đã chạy tốt (như nhóm Easy E01-E05), quy trình Regression Testing sẽ được thực hiện tự động dựa trên module `run_regression()`:

1. **Automated Continuous Benchmarking:**
   * Sau mỗi lần cập nhật Prompt hay Retrieval Pipeline, chạy lại toàn bộ pipeline đánh giá:
     ```powershell
     python domain_assistant.py
     python evaluate_answers.py
     ```
2. **Regression Thresholds & Quality Gates:**
   * **Overall Pass Rate Gate:** Hệ thống mới phải có `Pass Rate >= 60%` mới được cho phép commit / deploy.
   * **No Metric Degradation:** Chỉ số `Context Recall` không được giảm dưới `0.700` và `Relevance` không được giảm dưới `0.550`.
3. **Delta Monitoring via `run_regression()`:**
   * So sánh trực tiếp file `artifacts/benchmark_results.json` mới với file lưu trữ lịch sử để phát hiện các case bị giảm điểm (Negative Delta Case). Nếu phát hiện case nào giảm $> 0.15$ điểm Overall, pipeline CI/CD sẽ báo FAILED và yêu cầu rollback.