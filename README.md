# Where to Inject: Collaborative Filtering Depth in LLM Rankers

**ECS 172 Final Project — University of California, Davis**

Adi Kumar · Dev Rathod · Saee Patil · Krishna Gupta

Repo: [https://github.com/dev-rathod/ECS172](https://github.com/dev-rathod/ECS172)

---

## Overview

Large language models are promising re-rankers for recommendation, but they lack the **collaborative filtering (CF)** knowledge captured by traditional sequential recommenders like SASRec. Prior work injects CF signals either at the **input layer** (CoLLM, LLaRA) or at the **output/objective level** — leaving open a basic question:

> **Where in the network is the CF signal most usefully received?**

This project studies **injection depth as a first-class design variable**. We propose a **mid-layer hidden-state injection** method that adds projected SASRec embeddings directly to the hidden states of candidate tokens at an intermediate transformer layer via a PyTorch forward hook — no LLM weights modified. The injection layer is **selected using XAI attribution analysis** (Integrated Gradients, layer-wise conductance, ALTI+) rather than arbitrary hyperparameter search.

**Key findings (MovieLens-1M, LLaMA-3.2-1B):**

| Backbone | Best method | HR@1 | vs. best input-level baseline |
|---|---|---|---|
| Frozen | Mid-layer injection @ **L13** | **0.382** | 0.320 (soft-token input) |
| LoRA fine-tuned | Mid-layer injection @ **L10** | **0.482** | 0.458 (text-only LoRA) |

The attribution analysis correctly **predicted the shift in optimal injection layer** (L13 → L10) between the frozen and fine-tuned backbones, validating attribution as a principled layer-selection criterion.

---

## Method

```
Text prompt (user history + candidates A–J)
        │
   Layers L0 … L(k−1)        SASRec (50-dim item embeddings)
        │                            │
        │                    MLP Adapter (50 → 1024 → SiLU → LayerNorm → 2048)
        ▼                            │
   Layer Lk  ◄── forward hook: h_new = h + α · z_movie  (at A–J token positions)
        │
   Layers L(k+1) … L15
        │
        ▼
  Log-probs over letter tokens A–J  →  ranked candidate list
```

Five CF-integration strategies are compared, holding the CF source (SASRec), dataset, and evaluation protocol fixed:

1. **Text-only zero-shot** — frozen LLaMA ranks lettered title candidates by letter-token log-probability.
2. **LoRA fine-tuning** — parameter-efficient fine-tuning on the ranking task (~11.3M params, ≈1.1% of model).
3. **Soft-token replacement (input layer)** — candidate titles replaced entirely by adapter-projected CF soft tokens.
4. **CoLLM-style (input layer)** — title text + CF soft token appended per candidate at Layer 0.
5. **Mid-layer hidden-state injection (ours)** — CF vectors added to candidate hidden states at an XAI-selected intermediate layer.

---

## Results

### Frozen LLaMA-3.2-1B backbone

| Model variant | HR@1 | HR@3 | HR@5 | NDCG@3 | NDCG@5 | MRR |
|---|---|---|---|---|---|---|
| Random | 0.082 | 0.272 | 0.496 | 0.188 | 0.279 | 0.275 |
| Text-only baseline | 0.134 | 0.374 | 0.558 | 0.267 | 0.342 | 0.328 |
| CoLLM-style (input layer) | 0.122 | 0.400 | 0.610 | 0.281 | 0.367 | 0.339 |
| Soft-token (input layer) | 0.320 | 0.640 | 0.790 | 0.504 | 0.566 | 0.521 |
| **Ours — mid-layer injection (L13)** | **0.382** | 0.626 | **0.792** | **0.523** | **0.591** | **0.554** |

### LoRA fine-tuned backbone

| Model variant | HR@1 | HR@3 | HR@5 | NDCG@3 | NDCG@5 | MRR |
|---|---|---|---|---|---|---|
| Random | 0.082 | 0.272 | 0.496 | 0.188 | 0.279 | 0.275 |
| Soft-token (input layer) | 0.268 | 0.536 | 0.726 | 0.423 | 0.501 | 0.464 |
| CoLLM-style (input layer) | 0.328 | 0.634 | 0.806 | 0.504 | 0.575 | 0.525 |
| Text-only (LoRA only) | 0.458 | 0.730 | 0.848 | 0.616 | 0.665 | 0.624 |
| **Ours — mid-layer injection (L10)** | **0.482** | **0.770** | **0.896** | **0.649** | **0.701** | **0.650** |

Layer ablation (L6–L15) confirms that the empirically optimal injection layers match the peaks of the layer-wise attribution profiles on both backbones.

---

## Repository structure

```
.
├── src/
│   ├── inject_llm.py            # InjectedLlamaRanker: Mode A/C prompts + injection logic
│   ├── adapter.py               # MLP adapter (50 → 1024 → SiLU → LayerNorm → 2048)
│   ├── sasrec/                  # SASRec model + layers (Kang & McAuley, 2018)
│   ├── data.py                  # MovieLens loading, ID maps, user sequences
│   ├── ranking_data.py          # 10-candidate ranking example construction
│   ├── eval_injection.py        # Injection evaluation utilities
│   ├── kv_adapter.py            # KV-level adapter (exploratory)
│   └── metrics.py               # HR@k, NDCG@k, MRR
├── scripts/
│   ├── train_sasrec.py              # Phase 1: train SASRec on MovieLens-1M
│   ├── train_adapter.py             # Phase 2a: reconstruction-trained adapter
│   ├── train_adapter_ranking.py     # Phase 2b: listwise-ranking-trained adapter
│   ├── train_hs_adapter_finetunned.py  # Adapter training for hidden-state injection
│   ├── finetune_lora_ranking.py     # LoRA fine-tune on the ranking task
│   ├── eval_ranking.py              # Evaluate Mode A (text) / Mode C (soft tokens)
│   ├── eval_hidden_state.py         # Evaluate mid-layer hidden-state injection
│   ├── build_ranking_prompts_json.py
│   ├── cache_projected_embeddings.py
│   └── build_bar_plots.py
├── Finetuning/                  # Colab notebooks for LoRA fine-tuning
├── results/                     # Metric bar plots (base & fine-tuned)
├── results.md                   # Detailed experimental notes & position-bias analysis
├── Colab_Pipeline.ipynb         # End-to-end Colab pipeline
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/dev-rathod/ECS172.git
cd ECS172
pip install -r requirements.txt
```

Core dependencies: `torch >= 2.4`, `transformers >= 4.45`, `peft >= 0.11`, `accelerate`, `pandas`, `numpy`, `tqdm`. The LLM backbone is `unsloth/Llama-3.2-1B-Instruct` (downloaded automatically via Hugging Face).

**Data:** download [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/). Users with fewer than 11 ratings are filtered; the remaining users are split 90/10 into train/test. For each test user, the 10 most recent interactions form the history and the 11th serves as the positive item, mixed with 9 sampled negatives into a shuffled A–J candidate set (604 held-out users).

---

## Usage

### 1. Train SASRec (Phase 1)

```bash
python scripts/train_sasrec.py --epochs 15
```

Implicit feedback (rating ≥ 4), BCE loss with negative sampling. Produces 50-dim item embeddings (validation HR@10 = 0.551, NDCG@10 = 0.303).

### 2. Train the CF adapter (Phase 2)

```bash
# Reconstruction-grounded warm start
python scripts/train_adapter.py

# Listwise ranking objective (used by all CF injection conditions)
python scripts/train_adapter_ranking.py --epochs 12 --recon-lambda 0.3
```

### 3. Baselines

```bash
# Text-only zero-shot (Mode A) and soft-token replacement (Mode C)
python scripts/eval_ranking.py --modes A C

# LoRA fine-tuning on the ranking task
python scripts/finetune_lora_ranking.py \
    --model unsloth/Llama-3.2-1B-Instruct \
    --epochs 3 --output ./llama31-1b-movielens-ranking-lora
```

### 4. Mid-layer hidden-state injection (ours)

```bash
# Frozen backbone, inject at L13
python scripts/eval_hidden_state.py --target-layer 13 --alpha 0.1

# Fine-tuned backbone, inject at L10
python scripts/eval_hidden_state.py --model llama31-1b-movielens-full-final --target-layer 10

# Layer sweep
for L in 6 7 8 9 10 11 12 13 14 15; do
    python scripts/eval_hidden_state.py --target-layer $L --alpha 0.1
done
```

Most scripts support `--smoke` for a quick sanity run.

---

## XAI layer selection

Injection layers are chosen from attribution profiles, not grid search:

- **Integrated Gradients** — token/segment-level attribution from a zero-embedding baseline to the prompt.
- **Layer-wise conductance** — per-layer attribution score for the target answer token, producing a depth-wise importance profile.
- **ALTI+** — token-to-token information flow across layers, accounting for attention and residual connections.

On the **base** model, attribution concentrates in mid-to-late blocks (peaks ≈ L6, L13, L14) → inject at **L13**. After **LoRA fine-tuning**, attribution mass shifts earlier (L10, L11 dominant) → inject at **L10**. Both predictions match the empirically optimal injection layers from the ablation sweep.

---

## Team contributions

- **Adi Kumar** — project ideation and framing; soft-token replacement and CoLLM-style input-injection conditions.
- **Dev Rathod** — dataset preprocessing and prompt construction; mid-layer hidden-state injection (forward hook, MLP adapter) and the full L6–L15 layer sweep.
- **Krishna Gupta** — text-only zero-shot baseline and LoRA fine-tuning pipeline; log-probability ranking and metric evaluation.
- **Saee Patil** — full XAI pipeline (Integrated Gradients, layer-wise conductance, ALTI+); attribution figures and injection-layer selection.

---

## Key references

- Kang & McAuley. *Self-Attentive Sequential Recommendation* (SASRec). ICDM 2018.
- Zhang et al. *CoLLM: Integrating Collaborative Embeddings Into Large Language Models for Recommendation.* IEEE TKDE 2025.
- Bao et al. *TALLRec: An Effective and Efficient Tuning Framework to Align LLMs with Recommendation.* RecSys 2023.
- Sundararajan et al. *Axiomatic Attribution for Deep Networks* (Integrated Gradients). ICML 2017.
- Ferrando et al. *ALTI+.* EMNLP 2022.
- Sakarvadia et al. *Memory Injections: Correcting Multi-Hop Reasoning Failures During Inference.* BlackboxNLP 2023.