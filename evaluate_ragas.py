"""
RAGAS evaluation for the Industrial RAG Assistant.
Uses Gemini as the LLM judge (requires GOOGLE_API_KEY in .env).

Metrics:
  - Faithfulness       : Is every claim in the answer supported by the retrieved chunks?
  - Response Relevancy : Does the answer address what was asked?
  - Context Recall     : Did FAISS retrieve chunks containing the right answer?
  - Context Precision  : Of the retrieved chunks, how many were actually needed?

Run: python evaluate_ragas.py
"""

import os
import json
from dotenv import load_dotenv
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall, LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

load_dotenv()

# --- Eval dataset: questions + ground truth answers ---
# Ground truths are concise reference answers for context recall/precision scoring
EVAL_DATASET = [
    {
        "question": "What is basic oxygen steelmaking?",
        "ground_truth": "Basic oxygen steelmaking (BOS) is a process where oxygen is blown into molten pig iron to reduce carbon content and produce steel. It uses a Basic Oxygen Furnace (BOF) with pig iron, scrap steel, and flux."
    },
    {
        "question": "What are the main steps in steel production?",
        "ground_truth": "The main steps are: iron ore and coal preparation, iron making in a blast furnace, steel making in a BOF or EAF, continuous casting to solidify steel, and rolling (hot or cold) to shape the final product."
    },
    {
        "question": "What defects can occur in steel?",
        "ground_truth": "Steel defects include surface defects like cracks, scratches, and scale, as well as internal defects like porosity, inclusions, and segregation. They can compromise structural integrity and aesthetic quality."
    },
    {
        "question": "What is the difference between hot rolling and cold rolling?",
        "ground_truth": "Hot rolling is done above the recrystallization temperature, making steel easier to shape but with less precise dimensions. Cold rolling is done at room temperature, producing tighter tolerances, smoother surfaces, and higher strength."
    },
    {
        "question": "What equipment is used in rolling mills?",
        "ground_truth": "Rolling mills use work rolls, backup rolls, roll stands, tension reels, coilers, cooling systems, and shape measurement devices to reduce and shape steel into plates, coils, or strips."
    },
    {
        "question": "What are quality standards for steel?",
        "ground_truth": "Steel quality standards include ISO, ASTM, DIN, and EN standards that define chemical composition, mechanical properties, surface quality, and dimensional tolerances for different steel grades and applications."
    },
    {
        "question": "What causes surface defects in steel?",
        "ground_truth": "Surface defects in steel are caused by factors like improper casting conditions, roll marks, scale inclusions, thermal stresses, chemical reactions, and handling damage during processing."
    },
    {
        "question": "How is steel hardened?",
        "ground_truth": "Steel is hardened through heat treatment processes such as quenching (rapid cooling) and tempering. The steel is heated above its critical temperature and then rapidly cooled to form martensite, increasing hardness."
    },
]


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)


def get_rag_response(vectorstore, question, threshold=1.0):
    """Run the RAG pipeline and return answer + retrieved context strings."""
    from langchain_ollama import OllamaLLM

    results = vectorstore.similarity_search_with_score(question, k=5)
    relevant_docs = [doc for doc, score in results if score < threshold]

    if not relevant_docs:
        return "I couldn't find relevant information in the documents for this question.", []

    contexts = [doc.page_content for doc in relevant_docs]
    context_text = "\n\n".join(contexts)

    llm = OllamaLLM(model="mistral")
    prompt = f"""Answer the question based ONLY on the context below. Do not use your own knowledge.
If the context does not contain enough information, say "I don't have enough information to answer this."

Context:
{context_text}

Question: {question}

Answer:"""

    answer = llm.invoke(prompt)
    return answer, contexts


def run_ragas_evaluation():
    print("\n" + "="*80)
    print("INDUSTRIAL RAG ASSISTANT — RAGAS EVALUATION")
    print("="*80)

    # Set up Gemini as the LLM judge
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in .env — needed for RAGAS LLM judge.")
        return

    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    )

    print("\nLoading vectorstore...")
    vectorstore = load_vectorstore()

    print(f"Running RAG pipeline on {len(EVAL_DATASET)} questions...\n")

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(EVAL_DATASET, 1):
        q = item["question"]
        print(f"  [{i}/{len(EVAL_DATASET)}] {q}")
        answer, ctx = get_rag_response(vectorstore, q)
        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(item["ground_truth"])

    # Build RAGAS dataset
    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    })

    print("\nRunning RAGAS metrics (Gemini as judge)...")
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    # Display results
    print("\n" + "="*80)
    print("RAGAS RESULTS")
    print("="*80)
    df = result.to_pandas()

    print(f"\n{'Question':<55} {'Faith':>6} {'Relev':>6} {'Recall':>7} {'Prec':>6}")
    print("-"*85)
    for _, row in df.iterrows():
        q = row["user_input"][:52] + "..." if len(row["user_input"]) > 52 else row["user_input"]
        print(f"{q:<55} {row.get('faithfulness', 0):>6.2f} {row.get('response_relevancy', 0):>6.2f} "
              f"{row.get('llm_context_recall', 0):>7.2f} {row.get('llm_context_precision_with_reference', 0):>6.2f}")

    print("\nAVERAGE SCORES:")
    for metric in ["faithfulness", "response_relevancy", "llm_context_recall", "llm_context_precision_with_reference"]:
        if metric in df.columns:
            print(f"  {metric:<45}: {df[metric].mean():.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"ragas_results_{timestamp}.json"
    result_dict = {
        "timestamp": timestamp,
        "num_questions": len(EVAL_DATASET),
        "averages": {m: float(df[m].mean()) for m in ["faithfulness", "response_relevancy",
                     "llm_context_recall", "llm_context_precision_with_reference"] if m in df.columns},
        "per_question": df.to_dict(orient="records"),
    }
    with open(out_file, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\nResults saved to: {out_file}")
    print("="*80)


if __name__ == "__main__":
    run_ragas_evaluation()
