# RAG Benchmark Results

Retrieval and answer quality on three multi-hop QA benchmarks and EnterpriseRAG-Bench, a synthetic company knowledge base of 512k documents.

## Results

| Dataset | Setup | R@2 | R@5 | R@10 | EM | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MuSiQue | Graph | 66.28 | **85.60** | 91.39 | 57.80 | 69.58 |
| MuSiQue | No graph | 64.26 | **81.94** | 87.10 | 54.60 | 65.73 |
| 2WikiMultiHopQA | Graph | 85.45 | **98.33** | 98.55 | 71.10 | 79.84 |
| 2WikiMultiHopQA | No graph | 84.33 | **95.60** | 95.75 | 67.30 | 74.93 |
| HotpotQA | Graph | 93.25 | **98.95** | 99.50 | 69.60 | 83.39 |
| HotpotQA | No graph | 92.65 | **98.30** | 98.90 | 69.00 | 81.95 |
| EnterpriseRAG-Bench | No graph | 73.06 | **80.41** | 84.12 | n/a | n/a |

Scores are percentages. Each row is one final run, with settings tuned on the first 300 MuSiQue questions and then fixed. Recall counts passages on the multi-hop datasets and documents on EnterpriseRAG-Bench, using each dataset's original labels. EM and F1 come from a frontier LLM answering from the top 5 passages.

See [how the scores were measured](EVALUATION.md).

## Check the scores

Every question's ranking and answer is in `results/`. To recompute the table:

```bash
git clone https://github.com/OSU-NLP-Group/HippoRAG
git -C HippoRAG checkout d5c8329422e0a0b834a15874545cb6a74b4f9b26

python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/hf download onyx-dot-app/EnterpriseRAG-Bench data/questions/test.parquet \
  --repo-type dataset --local-dir data/erb

.venv/bin/python scripts/verify.py \
  --multihop-dir HippoRAG/reproduce/dataset \
  --erb-questions data/erb/data/questions/test.parquet \
  --write-qrels qrels
```

This needs Python 3.11 or later. The script stops if any score differs from the table.

You can also score the rankings with `ir_measures`:

```bash
.venv/bin/python -c "
import ir_measures as m
from ir_measures import R
print(m.calc_aggregate([R@2, R@5, R@10],
    list(m.read_trec_qrels('qrels/musique.qrels')),
    list(m.read_trec_run('results/musique/graph.trec'))))"
```

## References

- Question and corpus files: [HippoRAG, commit `d5c8329`](https://github.com/OSU-NLP-Group/HippoRAG/tree/d5c8329422e0a0b834a15874545cb6a74b4f9b26/reproduce/dataset)
- MuSiQue: Trivedi et al., TACL 2022
- 2WikiMultiHopQA: Ho et al., COLING 2020
- HotpotQA: Yang et al., EMNLP 2018
- EnterpriseRAG-Bench: Sun et al., [arXiv:2605.05253](https://arxiv.org/abs/2605.05253)
- This work builds on ideas from HippoRAG 2 ([arXiv:2502.14802](https://arxiv.org/abs/2502.14802)) and SAG2 ([arXiv:2610.22588](https://arxiv.org/abs/2610.22588))
