"""Offline RAGAS evaluation of the RAG pipeline, judged by Claude.

Runs a small fixed test set through the full retrieve -> rerank -> generate
chain, then scores the results with RAGAS (faithfulness, answer relevancy,
context precision, context recall). Not part of the live query path.

See CLAUDE.md "Deviation from spec" for why the judge is Claude rather than
Gemini as originally specified.

Scores each row by calling the RAGAS metric objects directly inside a single
asyncio.run() rather than via ragas.evaluate(). ragas.evaluate()'s internal
Executor applies nest_asyncio and runs jobs through its own event-loop
wrapper, which hung indefinitely against Gemini's async client in this
environment (direct calls to the same client outside that executor worked
fine, and the hang persisted even at max_workers=1, so it's specific to
ragas's executor rather than concurrency or the model itself). Root cause
unresolved; calling metrics directly sidesteps it entirely.
"""

import asyncio
import sys
import warnings
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import os  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY is not set (required for the Claude judge model)", file=sys.stderr)
    sys.exit(1)

# The legacy top-level ragas.metrics classes are deprecated in favour of
# ragas.metrics.collections (a newer, instructor/litellm-based API not used
# here), but remain fully functional in ragas 0.4.x and are what pairs
# cleanly with LangchainLLMWrapper.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

import pandas as pd  # noqa: E402

from config import JUDGE_MODEL  # noqa: E402
from rag_chain import RAGChain  # noqa: E402
from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_community.embeddings import OllamaEmbeddings  # noqa: E402
from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness  # noqa: E402

# Test set spans all seven policy documents (IT_SECURITY_POLICY gets two,
# since it has both an access-control and an incident-reporting angle).
TEST_SET = [
    {
        "question": "Can I use my company email for personal purposes?",
        "reference": (
            "Limited personal use of company IT resources, including email, is permitted "
            "provided it does not interfere with job duties, violate other company policies, "
            "or incur additional costs for the company."
        ),
    },
    {
        "question": "What should I do if I suspect a policy violation?",
        "reference": (
            "Report it promptly to your manager, HR, or a designated reporting channel; "
            "retaliation against employees who report in good faith is prohibited."
        ),
    },
    {
        "question": "What must I do with confidential information when my employment ends?",
        "reference": (
            "Promptly return all tangible confidential information, including all copies, to "
            "the company, or destroy it and certify in writing that the destruction has occurred."
        ),
    },
    {
        "question": "Who should I notify if I suspect a data breach?",
        "reference": (
            "Report it to the Data Protection Officer (DPO) immediately; if the breach is "
            "notifiable, the company will inform the PDPC within the statutory timeframe and "
            "notify affected individuals."
        ),
    },
    {
        "question": "What types of flexible work arrangements does the company recognise?",
        "reference": (
            "Three types: Flexi-Place (working from a location other than the primary office), "
            "Flexi-Time (staggered hours or compressed work weeks), and Flexi-Load (part-time "
            "or job-sharing)."
        ),
    },
    {
        "question": "What should I do if I suspect my password has been compromised?",
        "reference": "Report the suspected compromise to IT immediately.",
    },
    {
        "question": "Can I raise a whistleblowing concern anonymously?",
        "reference": (
            "Yes, concerns may be raised anonymously, although providing contact details, where "
            "the reporter is comfortable doing so, can help the company investigate more effectively."
        ),
    },
    {
        "question": "Is multi-factor authentication required for accessing company systems?",
        "reference": (
            "Yes, MFA is required for access to email, cloud storage, and other systems "
            "designated by IT as sensitive."
        ),
    },
]


def run_pipeline(chain: RAGChain) -> list[dict]:
    rows = []
    for item in TEST_SET:
        result = chain.query(item["question"])
        rows.append(
            {
                "user_input": item["question"],
                "response": result["answer"],
                "retrieved_contexts": [s["text"] for s in result["sources"]] or [""],
                "reference": item["reference"],
            }
        )
    return rows


def build_metrics(judge_llm, judge_embeddings):
    return {
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_relevancy": AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        "context_precision": ContextPrecision(llm=judge_llm),
        "context_recall": ContextRecall(llm=judge_llm),
    }


async def score_row(metrics: dict, row: dict) -> dict:
    sample = SingleTurnSample(
        user_input=row["user_input"],
        response=row["response"],
        retrieved_contexts=row["retrieved_contexts"],
        reference=row["reference"],
    )

    async def score_one(name, metric):
        try:
            return name, await metric.single_turn_ascore(sample)
        except Exception as exc:
            print(f"  WARNING: {name} failed for {row['user_input']!r}: {exc}", file=sys.stderr)
            return name, float("nan")

    results = await asyncio.gather(*(score_one(name, m) for name, m in metrics.items()))
    return dict(results)


async def run_evaluation(rows: list[dict]) -> list[dict]:
    judge_llm = LangchainLLMWrapper(
        ChatAnthropic(model=JUDGE_MODEL, anthropic_api_key=os.environ["ANTHROPIC_API_KEY"])
    )
    judge_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
    metrics = build_metrics(judge_llm, judge_embeddings)

    scored_rows = []
    for i, row in enumerate(rows, 1):
        print(f"  Scoring {i}/{len(rows)}: {row['user_input']!r}")
        scores = await score_row(metrics, row)
        scored_rows.append({**row, **scores})
    return scored_rows


def main():
    chain = RAGChain()
    print(f"Running {len(TEST_SET)} test questions through the pipeline...")
    rows = run_pipeline(chain)

    print(f"Scoring with RAGAS (judge: {JUDGE_MODEL})...")
    scored_rows = asyncio.run(run_evaluation(rows))

    df = pd.DataFrame(scored_rows)
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print()
    print(df[["user_input"] + metric_cols].to_string(index=False))
    print()
    print("Mean scores:")
    for col in metric_cols:
        print(f"  {col}: {df[col].mean():.3f}")

    out_path = ROOT_DIR / "data" / "eval_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
