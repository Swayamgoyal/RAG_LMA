# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Two RAG pipelines are compared over the same chunked corpus, the **IFAB Laws of the Game 2026/27** (260-page football rulebook):

- **Pipeline 1 (baseline):** `all-MiniLM-L6-v2` bi-encoder → FAISS inner-product search → Top 10 → first 3 chunks → LLM
- **Pipeline 2 (re-ranking):** the same Top 10 → `cross-encoder/ms-marco-MiniLM-L-6-v2` → re-sorted Top 3 → LLM (identical prompt)
- **LLM:** `qwen2.5:7b`, run locally through Ollama

Everything (code, outputs and the report) is in **[RAG_Baseline_vs_ReRanking.ipynb](RAG_Baseline_vs_ReRanking.ipynb)**.

## Repository layout

| Path | Contents |
|---|---|
| `RAG_Baseline_vs_ReRanking.ipynb` | Notebook: data prep, both pipelines, failure-mode analysis, latency, scaling, report |
| `results/rank_shifts.csv` | Gold-chunk rank before/after re-ranking for the failure and limitation queries |
| `results/answers.json` | LLM Answer 1 vs Answer 2 for every failure/limitation query, with context chunk ids |
| `results/query_bank_eval.csv` | Retrieval-only evaluation over a 40-query bank (Hit@k / MRR) |
| `results/latency.csv` | Per-stage latency statistics |
| `results/rank_shift.png` | Rank-shift chart |
| `requirements.txt` | Python dependencies |

## Running it

1. Python 3.14 environment: `pip install -r requirements.txt` (for a GPU, install the CUDA build of torch first, as noted in the file).
2. Install [Ollama](https://ollama.com) and run `ollama pull qwen2.5:7b`.
3. Run all cells in the notebook. The rulebook PDF is downloaded from theifab.com into `data/` on first run.

Tested on Windows 11 with an RTX 4060 Laptop GPU (8 GB).
