# Evaluation

## Datasets

| Dataset | Questions | Scored for recall | Corpus |
| --- | ---: | ---: | --- |
| MuSiQue | 1,000 | 1,000 | 11,656 passages |
| 2WikiMultiHopQA | 1,000 | 1,000 | 6,119 passages |
| HotpotQA | 1,000 | 1,000 | 9,811 passages |
| EnterpriseRAG-Bench | 500 | 470 | 511,958 documents |

Dataset file checksums are in `results/scores.json`.

## Setups

Both setups were frozen on MuSiQue before the final runs on the other datasets.

- `graph`: run on the three multi-hop datasets.
- `no-graph`: run on all four datasets.

## Identifiers

- A multi-hop passage ID is `<dataset>-<position>`, where `position` is the passage's zero-based position in `<dataset>_corpus.json`, written with five digits.
- If the same passage text appears more than once in a corpus, its label uses the last position.
- An EnterpriseRAG-Bench ID is the dataset's `doc_id`.

## Metrics

- **Recall@k:** the share of a question's labeled passages or documents in the first k results, averaged over labeled questions.
- **EnterpriseRAG-Bench recall:** ranked results were reduced to distinct documents at each document's first position, and the top 10 are published. Labels are the original `expected_doc_ids`, with no LLM relabeling. The 30 questions without expected documents are not scored for recall.
- **EM and F1:** SQuAD v1.1 answer normalization, taking the best score across a question's accepted answers. A frontier LLM generated each answer from the top 5 passages. EnterpriseRAG-Bench answers were not generated.

## Files

`results/<dataset>/<setup>.jsonl` holds one question per line:

```json
{"question_id": "...", "ranked_ids": ["...", "..."], "answer": "..."}
```

`results/<dataset>/<setup>.trec` holds the same ranking in TREC run format, with the score column derived from rank.
