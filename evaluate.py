import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from datetime import datetime

load_dotenv()

DOC_RELEVANCE_THRESHOLD = 1.0

class RAGEvaluator:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.load_local(
            "vectorstore",
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.llm = OllamaLLM(model="mistral")
        self.results = []

    def get_relevant_docs(self, question):
        results = self.vectorstore.similarity_search_with_score(question, k=5)
        return [(doc, score) for doc, score in results if score < DOC_RELEVANCE_THRESHOLD]

    def query_rag(self, question):
        """Query with docs-first fallback to web (mirrors production logic)."""
        if not question or not question.strip():
            return {"question": question, "answer": "[EMPTY QUERY — skipped]",
                    "source": "none", "retrieved_docs": 0, "sources": [], "context_length": 0}

        doc_results = self.get_relevant_docs(question)

        if doc_results:
            docs = [doc for doc, _ in doc_results]
            context = "\n\n".join([d.page_content for d in docs])
            source = "documents"
            sources = [d.metadata.get("source", "Unknown") for d in docs]
        else:
            from ddgs import DDGS
            with DDGS() as ddgs:
                web_results = list(ddgs.text(question, max_results=5))
            if web_results:
                context = "\n\n".join([f"{r['title']}: {r['body']}" for r in web_results])
                source = "web"
                sources = [r.get("href", "") for r in web_results]
            else:
                context = ""
                source = "none"
                sources = []

        if not context:
            answer = "I couldn't find relevant information in the documents or on the web for this question."
        else:
            prompt = f"""Answer the question based ONLY on the context below. Do not use your own knowledge.
If the context does not contain enough information, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""
            answer = self.llm.invoke(prompt)

        return {
            "question": question,
            "answer": answer,
            "source": source,
            "retrieved_docs": len(doc_results) if source == "documents" else 0,
            "sources": sources,
            "context_length": len(context)
        }

    def evaluate_groundedness(self, result):
        answer = result["answer"].lower()
        hallucination_markers = [
            "i don't know", "i'm not sure", "i cannot", "i don't have",
            "based on my training", "in general", "typically", "usually",
            "i believe", "it is likely",
        ]
        hallucination_score = sum(1 for m in hallucination_markers if m in answer) / len(hallucination_markers)
        return round(1 - hallucination_score, 2)

    def evaluate_relevance(self, result):
        if result["retrieved_docs"] == 0 and result["source"] != "web":
            return 0.0
        return 1.0 if result["source"] in ("documents", "web") else 0.0

    def evaluate_length(self, result):
        word_count = len(result["answer"].split())
        if word_count < 10:
            return {"score": 0.2, "feedback": "Too short"}
        elif word_count < 30:
            return {"score": 0.5, "feedback": "Somewhat short"}
        elif word_count <= 300:
            return {"score": 1.0, "feedback": "Appropriate length"}
        elif word_count <= 500:
            return {"score": 0.7, "feedback": "Somewhat long"}
        else:
            return {"score": 0.3, "feedback": "Too long"}

    def run_test(self, question, expected_in_answer=None, expect_fallback=False):
        print(f"\n{'='*80}")
        print(f"QUESTION: {repr(question)}")
        print("="*80)

        try:
            result = self.query_rag(question)

            if result["source"] == "none" and not question.strip():
                print("  → Empty query handled gracefully (skipped)")
                self.results.append({"question": question, "overall_score": 1.0, "note": "empty query handled"})
                return True

            groundedness = self.evaluate_groundedness(result)
            relevance = self.evaluate_relevance(result)
            length_eval = self.evaluate_length(result)

            correctness = 1.0
            if expected_in_answer:
                found = any(exp.lower() in result["answer"].lower() for exp in expected_in_answer)
                correctness = 1.0 if found else 0.3

            overall_score = round((groundedness + relevance + length_eval["score"] + correctness) / 4, 2)

            print(f"\nSOURCE USED: {result['source'].upper()}")
            print(f"ANSWER:\n{result['answer'][:500]}{'...' if len(result['answer']) > 500 else ''}\n")
            print(f"EVALUATION:")
            print(f"  Groundedness:  {groundedness}  (grounded in retrieved context?)")
            print(f"  Relevance:     {relevance}  (found useful context?)")
            print(f"  Length:        {length_eval['score']}  ({length_eval['feedback']})")
            print(f"  Correctness:   {correctness}  (contains expected keywords?)")
            print(f"  OVERALL:       {overall_score}/1.0")

            if expect_fallback:
                fell_back = result["source"] == "web"
                print(f"  Fallback to web: {'✅ YES' if fell_back else '❌ NO (stayed in docs)'}")

            self.results.append({
                "question": question,
                "answer": result["answer"],
                "source": result["source"],
                "groundedness": groundedness,
                "relevance": relevance,
                "length_score": length_eval["score"],
                "correctness": correctness,
                "overall_score": overall_score,
                "sources": result["sources"],
            })

            return overall_score >= 0.7

        except Exception as e:
            print(f"ERROR: {str(e)}")
            self.results.append({"question": question, "error": str(e), "overall_score": 0.0})
            return False


def run_test_suite():
    evaluator = RAGEvaluator()

    print("\n" + "="*80)
    print("INDUSTRIAL RAG ASSISTANT — EDGE CASE TEST SUITE")
    print("="*80)

    tests = {
        "Basic Facts (should answer from docs)": [
            {"question": "What is basic oxygen steelmaking?",
             "expected": ["oxygen", "steel"]},
            {"question": "What are the main steps in steel production?",
             "expected": ["steps", "production"]},
            {"question": "What are defects in steel?",
             "expected": ["defect"]},
        ],
        "Edge Cases — Empty / Null": [
            {"question": "",    "expected": []},
            {"question": "   ", "expected": []},
        ],
        "Edge Cases — Nonsense / Gibberish": [
            {"question": "xyz abc qwerty 12345", "expected": [], "expect_fallback": True},
            {"question": "asdfghjkl",            "expected": [], "expect_fallback": True},
        ],
        "Edge Cases — Off-topic (should fall back to web)": [
            {"question": "What is the meaning of life?",     "expected": [], "expect_fallback": True},
            {"question": "Who won the FIFA World Cup 2022?", "expected": [], "expect_fallback": True},
            {"question": "What is a Python decorator?",      "expected": [], "expect_fallback": True},
        ],
        "Edge Cases — Vague Queries": [
            {"question": "Tell me something", "expected": []},
            {"question": "What?",             "expected": []},
            {"question": "Explain",           "expected": []},
        ],
        "Edge Cases — Context Carryover (no memory between turns)": [
            # These test if the system handles follow-ups without prior context
            {"question": "What is hot rolling?",        "expected": ["rolling"]},
            {"question": "What about cold rolling?",    "expected": ["rolling"]},
            {"question": "How do they compare?",        "expected": []},  # no context — will struggle
        ],
        "Complex / Multi-part Queries": [
            {"question": "What are the different types of steel defects and how are they classified?",
             "expected": ["defect"]},
            {"question": "Compare hot rolling and cold rolling — what are the differences in temperature, process and product quality?",
             "expected": ["rolling"]},
        ],
    }

    total_tests = 0
    passed_tests = 0

    for category, test_list in tests.items():
        print(f"\n\n{'#'*80}")
        print(f"# {category.upper()}")
        print(f"{'#'*80}")

        for test in test_list:
            passed = evaluator.run_test(
                test["question"],
                test.get("expected"),
                test.get("expect_fallback", False)
            )
            total_tests += 1
            if passed:
                passed_tests += 1

    print(f"\n\n{'='*80}")
    print("SUMMARY REPORT")
    print(f"{'='*80}")
    print(f"Total Tests:   {total_tests}")
    print(f"Passed:        {passed_tests}")
    print(f"Failed:        {total_tests - passed_tests}")
    print(f"Pass Rate:     {round(passed_tests / total_tests * 100, 1)}%")

    valid = [r for r in evaluator.results if "overall_score" in r and "error" not in r and "note" not in r]
    if valid:
        print(f"\nAVERAGE METRICS (across {len(valid)} scored tests):")
        print(f"  Groundedness: {round(sum(r['groundedness'] for r in valid) / len(valid), 2)}/1.0")
        print(f"  Relevance:    {round(sum(r['relevance'] for r in valid) / len(valid), 2)}/1.0")
        print(f"  Correctness:  {round(sum(r['correctness'] for r in valid) / len(valid), 2)}/1.0")
        print(f"  OVERALL:      {round(sum(r['overall_score'] for r in valid) / len(valid), 2)}/1.0")

    by_source = {}
    for r in valid:
        src = r.get("source", "unknown")
        by_source.setdefault(src, []).append(r["overall_score"])
    print("\nSCORES BY SOURCE:")
    for src, scores in by_source.items():
        print(f"  {src}: avg {round(sum(scores)/len(scores), 2)} over {len(scores)} tests")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump({"timestamp": timestamp, "total_tests": total_tests,
                   "passed_tests": passed_tests, "individual_results": evaluator.results}, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    print("="*80)


if __name__ == "__main__":
    run_test_suite()
