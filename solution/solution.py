"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    Fields:
        question:           The question to answer.
        expected_answer:    The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_contexts: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    Fields:
        qa_pair:           The original QAPair.
        actual_answer:     What the agent actually returned.
        faithfulness:      Float 0-1, how grounded the answer is in context.
        relevance:         Float 0-1, how relevant the answer is to the question.
        completeness:      Float 0-1, how complete the answer is vs expected.
        passed:            True if all three scores >= 0.5.
        failure_type:      None if passed, otherwise one of:
                           "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
    """
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool = False
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness."""
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------

STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """Measure how grounded the answer is in the context."""
        if not answer or not answer.strip():
            return 1.0
        
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 1.0
            
        context_tokens = _tokenize(context)
        overlap = len(answer_tokens.intersection(context_tokens))
        score = overlap / len(answer_tokens)
        return min(max(score, 0.0), 1.0)

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """Measure how relevant the answer is to the question."""
        if not question or not question.strip():
            return 1.0
            
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 1.0
            
        answer_tokens = _tokenize(answer)
        overlap = len(answer_tokens.intersection(question_tokens))
        score = overlap / len(question_tokens)
        return min(max(score, 0.0), 1.0)

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """Measure how well the answer covers the expected answer."""
        if not expected or not expected.strip():
            return 1.0
            
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
            
        answer_tokens = _tokenize(answer)
        overlap = len(answer_tokens.intersection(expected_tokens))
        score = overlap / len(expected_tokens)
        return min(max(score, 0.0), 1.0)

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by the UNION of retrieved chunks."""
        if not expected or not expected.strip():
            return 1.0
            
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
            
        if not contexts:
            return 0.0
            
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens.update(_tokenize(chunk))
            
        overlap = len(expected_tokens.intersection(union_tokens))
        score = overlap / len(expected_tokens)
        return min(max(score, 0.0), 1.0)

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K)."""
        if not expected or not expected.strip():
            return 1.0
            
        expected_tokens = _tokenize(expected)
        if not expected_tokens or not contexts:
            return 0.0
            
        relevant_flags: list[bool] = []
        for chunk in contexts:
            chunk_tokens = _tokenize(chunk)
            rel = (len(chunk_tokens.intersection(expected_tokens)) / len(expected_tokens)) >= relevance_threshold
            relevant_flags.append(rel)
            
        num_relevant = sum(relevant_flags)
        if num_relevant == 0:
            return 0.0
            
        precision_sum = 0.0
        relevant_count_so_far = 0
        
        for k, is_rel in enumerate(relevant_flags, start=1):
            if is_rel:
                relevant_count_so_far += 1
                precision_sum += (relevant_count_so_far / k)
                
        ap = precision_sum / num_relevant
        return min(max(ap, 0.0), 1.0)

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """Run standard evaluation and optional retrieval metrics."""
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)
        
        ctx_recall = None
        ctx_precision = None
        if contexts is not None:
            ctx_recall = self.evaluate_context_recall(contexts, expected)
            ctx_precision = self.evaluate_context_precision(contexts, expected)
            
        passed = (faithfulness >= 0.5) and (relevance >= 0.5) and (completeness >= 0.5)
        
        failure_type = None
        if not passed:
            if faithfulness < 0.3:
                failure_type = "hallucination"
            elif relevance < 0.3:
                failure_type = "irrelevant"
            elif completeness < 0.3:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"
                
        qa_pair = QAPair(
            question=question,
            expected_answer=expected,
            context=context,
            retrieved_contexts=contexts if contexts is not None else []
        )
        
        return EvalResult(
            qa_pair=qa_pair,
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=ctx_precision,
            context_recall=ctx_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """Sort chunks by word overlap with query, highest overlap first."""
    query_tokens = _tokenize(query)
    return sorted(
        contexts,
        key=lambda c: len(_tokenize(c).intersection(query_tokens)),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """Uses an LLM to score AI responses according to a rubric."""

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """Score an AI response using the judge LLM."""
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Rubric: {rubric}\n"
            "Evaluate and return JSON scores."
        )
        raw_res = self.judge_llm_fn(prompt)
        
        scores: dict[str, float] = {}
        try:
            match = re.search(r"\{.*\}", raw_res, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict) and "scores" in parsed:
                    scores = {k: float(v) for k, v in parsed["scores"].items()}
                elif isinstance(parsed, dict):
                    scores = {k: float(v) for k, v in parsed.items() if k in rubric}
        except Exception:
            pass
            
        if not scores:
            scores = {criterion: 0.5 for criterion in rubric}
            
        return {
            "scores": scores,
            "reasoning": raw_res,
        }

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Detect potential bias patterns in a batch of judge scores."""
        if not scores_batch:
            return {
                "positional_bias": False,
                "leniency_bias": False,
                "severity_bias": False,
            }
            
        all_scores: list[float] = []
        for item in scores_batch:
            scores_dict = item.get("scores", {})
            all_scores.extend(scores_dict.values())
            
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.5
        
        pos_bias = False
        if len(scores_batch) > 1:
            first_scores = list(scores_batch[0].get("scores", {}).values())
            rest_scores = [v for b in scores_batch[1:] for v in b.get("scores", {}).values()]
            if first_scores and rest_scores:
                avg_first = sum(first_scores) / len(first_scores)
                avg_rest = sum(rest_scores) / len(rest_scores)
                if avg_first - avg_rest > 0.2:
                    pos_bias = True

        return {
            "positional_bias": pos_bias,
            "leniency_bias": avg_score > 0.8,
            "severity_bias": avg_score < 0.3,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Runs a full evaluation benchmark."""

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """Run all QA pairs through agent and evaluate."""
        results: list[EvalResult] = []
        for pair in qa_pairs:
            actual_answer = agent_fn(pair.question)
            retrieved = pair.retrieved_contexts if pair.retrieved_contexts else None
            res = evaluator.run_full_eval(
                answer=actual_answer,
                question=pair.question,
                context=pair.context,
                expected=pair.expected_answer,
                contexts=retrieved,
            )
            res.qa_pair = pair
            results.append(res)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """Generate an aggregate report from evaluation results."""
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_completeness": 0.0,
                "avg_context_recall": None,
                "avg_context_precision": None,
                "failure_types": {},
            }
            
        passed_cnt = sum(1 for r in results if r.passed)
        avg_f = sum(r.faithfulness for r in results) / total
        avg_r = sum(r.relevance for r in results) / total
        avg_c = sum(r.completeness for r in results) / total
        
        recalls = [r.context_recall for r in results if r.context_recall is not None]
        precisions = [r.context_precision for r in results if r.context_precision is not None]
        
        avg_rec = sum(recalls) / len(recalls) if recalls else None
        avg_prec = sum(precisions) / len(precisions) if precisions else None
        
        fail_types: dict[str, int] = {}
        for r in results:
            if not r.passed and r.failure_type:
                fail_types[r.failure_type] = fail_types.get(r.failure_type, 0) + 1

        return {
            "total": total,
            "passed": passed_cnt,
            "pass_rate": passed_cnt / total,
            "avg_faithfulness": avg_f,
            "avg_relevance": avg_r,
            "avg_completeness": avg_c,
            "avg_context_recall": avg_rec,
            "avg_context_precision": avg_prec,
            "failure_types": fail_types,
        }

    def run_regression(self, new_results: list[EvalResult], baseline_results: list[EvalResult]) -> dict[str, Any]:
        """Compare new evaluation results against a baseline."""
        def _avg(res_list: list[EvalResult], attr: str) -> float:
            return sum(getattr(r, attr) for r in res_list) / len(res_list) if res_list else 0.0

        new_f = _avg(new_results, "faithfulness")
        new_r = _avg(new_results, "relevance")
        new_c = _avg(new_results, "completeness")

        base_f = _avg(baseline_results, "faithfulness")
        base_r = _avg(baseline_results, "relevance")
        base_c = _avg(baseline_results, "completeness")

        regressions: list[str] = []
        if base_f - new_f > 0.05:
            regressions.append("faithfulness")
        if base_r - new_r > 0.05:
            regressions.append("relevance")
        if base_c - new_c > 0.05:
            regressions.append("completeness")

        return {
            "new_avg_faithfulness": new_f,
            "new_avg_relevance": new_r,
            "new_avg_completeness": new_c,
            "baseline_avg_faithfulness": base_f,
            "baseline_avg_relevance": base_r,
            "baseline_avg_completeness": base_c,
            "regressions": regressions,
            "passed": len(regressions) == 0,
        }

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """Return EvalResults where any score is below threshold."""
        return [
            r for r in results 
            if r.faithfulness < threshold or r.relevance < threshold or r.completeness < threshold or not r.passed
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """Analyzes failed evaluation results to identify patterns and suggest fixes."""

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """Count failures by failure_type."""
        counts: dict[str, int] = {}
        for f in failures:
            ft = f.failure_type or "unknown"
            counts[ft] = counts.get(ft, 0) + 1
        return counts

    def find_root_cause(self, failure: EvalResult) -> str:
        """Suggest a root cause for a single failure based on its scores."""
        scores = {
            "faithfulness": failure.faithfulness,
            "relevance": failure.relevance,
            "completeness": failure.completeness,
        }
        low_scores = [k for k, v in scores.items() if v < 0.5]
        if len(low_scores) > 1:
            return "Multiple issues detected — review full pipeline"
            
        min_metric = min(scores, key=scores.get) # type: ignore
        if min_metric == "faithfulness":
            return "Context is missing or irrelevant — improve retrieval"
        elif min_metric == "relevance":
            return "Answer does not address the question — improve prompt clarity"
        else:
            return "Answer is missing key information — increase context window or improve generation"

    def generate_improvement_log(self, failures: list[EvalResult], suggestions: list[str]) -> str:
        """Generate a Markdown table logging failures and improvement actions."""
        lines = [
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |",
            "|------------|------|------------|---------------|--------|",
        ]
        for idx, f in enumerate(failures, start=1):
            f_id = f"F{idx:03d}"
            f_type = f.failure_type or "unknown"
            cause = self.find_root_cause(f)
            sugg = suggestions[idx - 1] if idx - 1 < len(suggestions) else (suggestions[0] if suggestions else "Review pipeline configuration")
            lines.append(f"| {f_id} | {f_type} | {cause} | {sugg} | Open |")
        return "\n".join(lines)

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """Generate a prioritized list of improvement suggestions based on failure patterns."""
        if not failures:
            return []
            
        cats = self.categorize_failures(failures)
        suggestions: list[str] = []
        
        if cats.get("hallucination", 0) > 0:
            suggestions.append("Implement hallucination checker to filter unsupported claims")
        if cats.get("irrelevant", 0) > 0:
            suggestions.append("Improve prompt clarity and system instructions to focus on user question")
        if cats.get("incomplete", 0) > 0:
            suggestions.append("Add few-shot examples showing complete answers to improve completeness")
        if cats.get("off_topic", 0) > 0:
            suggestions.append("Refine query routing and intent detection module")
            
        defaults = [
            "Increase chunk size in RAG pipeline to reduce context fragmentation",
            "Implement hallucination checker to filter unsupported claims",
            "Add few-shot examples showing complete answers to improve completeness",
        ]
        for d in defaults:
            if len(suggestions) >= 3:
                break
            if d not in suggestions:
                suggestions.append(d)
                
        return suggestions


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    qa_pairs = [
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)