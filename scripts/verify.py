#!/usr/bin/env python3
"""Recompute the published scores from the run files and the public datasets."""

import argparse
import hashlib
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = (2, 5, 10)
TOLERANCE = 1e-6
MULTIHOP = {"musique": None, "2wikimultihopqa": " ", "hotpotqa": ""}
ERB = "enterprise-rag-bench"


def sha256(path):
    """Returns the SHA-256 of a file."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def check_file(path, expected):
    """Stops if a dataset file differs from the one the runs used."""
    actual = sha256(path)
    if actual != expected:
        sys.exit(f"{path} has SHA-256 {actual}; expected {expected}.")


def passage_key(title, text):
    """Returns a passage as it was indexed: its title, then its text."""
    return f"{title}\n{text}"


def load_multihop(name, data_dir, checksums):
    """Reads a multi-hop dataset's supporting passage IDs and answers."""
    questions_path = data_dir / f"{name}.json"
    corpus_path = data_dir / f"{name}_corpus.json"
    check_file(questions_path, checksums[questions_path.name])
    check_file(corpus_path, checksums[corpus_path.name])
    positions = {
        passage_key(passage["title"], passage["text"]): position
        for position, passage in enumerate(json.loads(corpus_path.read_bytes()))
    }
    gold, answers = {}, {}
    for sample in json.loads(questions_path.read_bytes()):
        if name == "musique":
            sample_id = sample["id"]
            passages = [
                (paragraph["title"], paragraph["paragraph_text"])
                for paragraph in sample["paragraphs"]
                if paragraph["is_supporting"]
            ]
            answers[sample_id] = [sample["answer"], *sample.get("answer_aliases", [])]
        else:
            sample_id = sample["_id"]
            titles = {title for title, _ in sample["supporting_facts"]}
            passages = [
                (title, MULTIHOP[name].join(sentences))
                for title, sentences in sample["context"]
                if title in titles
            ]
            answers[sample_id] = [sample["answer"]]
        gold[sample_id] = {
            f"{name}-{positions[passage_key(title, text)]:05d}"
            for title, text in passages
        }
    return gold, answers


def load_erb(path, checksums):
    """Reads EnterpriseRAG-Bench's original expected document IDs."""
    import pyarrow.parquet as pq

    check_file(path, checksums["enterprise-rag-bench/questions/test.parquet"])
    rows = pq.read_table(path, columns=["question_id", "expected_doc_ids"]).to_pylist()
    return {row["question_id"]: set(row["expected_doc_ids"] or []) for row in rows}, {}


def normalize(answer):
    """Normalizes an answer as SQuAD v1.1 does."""
    text = "".join(c for c in answer.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(references, response):
    """Returns 1 if the answer equals a reference after normalization."""
    normalized = normalize(response)
    return float(any(normalize(reference) == normalized for reference in references))


def token_f1(references, response):
    """Returns the answer's best token F1 against its references."""
    response_tokens = normalize(response).split()
    best = 0.0
    for reference in references:
        reference_tokens = normalize(reference).split()
        common = sum((Counter(response_tokens) & Counter(reference_tokens)).values())
        if common:
            precision = common / len(response_tokens)
            recall = common / len(reference_tokens)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def score(path, gold, answers):
    """Scores one run file."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(gold):
        sys.exit(f"{path} does not hold each dataset question exactly once.")
    labeled = [(gold[row["question_id"]], row["ranked_ids"]) for row in rows if gold[row["question_id"]]]
    result = {"questions": len(rows), "labeled_questions": len(labeled)}
    for k in CUTOFFS:
        result[f"recall@{k}"] = fmean(len(g.intersection(r[:k])) / len(g) for g, r in labeled)
    responses = [(answers[row["question_id"]], row["answer"]) for row in rows if row.get("answer") is not None]
    if responses:
        result["exact_match"] = fmean(exact_match(a, r) for a, r in responses)
        result["f1"] = fmean(token_f1(a, r) for a, r in responses)
    return result


def write_qrels(directory, name, gold):
    """Writes a dataset's labels in TREC qrels format."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{question_id} 0 {doc_id} 1\n"
        for question_id, doc_ids in gold.items()
        for doc_id in sorted(doc_ids)
    ]
    (directory / f"{name}.qrels").write_text("".join(lines), encoding="utf8")


def main():
    """Recomputes every published score and fails on any difference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multihop-dir", type=Path, required=True)
    parser.add_argument("--erb-questions", type=Path, required=True)
    parser.add_argument("--write-qrels", type=Path)
    arguments = parser.parse_args()

    expected = json.loads((ROOT / "results" / "scores.json").read_text(encoding="utf8"))
    checksums = expected["data_files"]
    labels = {name: load_multihop(name, arguments.multihop_dir, checksums) for name in MULTIHOP}
    labels[ERB] = load_erb(arguments.erb_questions, checksums)
    if arguments.write_qrels:
        for name, (gold, _) in labels.items():
            write_qrels(arguments.write_qrels, name, gold)

    failures = 0
    for run, published in expected["results"].items():
        name = run.split("/")[0]
        actual = score(ROOT / "results" / f"{run}.jsonl", *labels[name])
        mismatched = [
            key for key, value in published.items()
            if key not in actual or abs(actual[key] - value) > TOLERANCE
        ]
        failures += bool(mismatched)
        print(f"{'FAIL' if mismatched else 'ok':4} {run}" + (f" {mismatched}" if mismatched else ""))
    if failures:
        sys.exit(f"{failures} run(s) differ from the published scores.")
    print(f"All {len(expected['results'])} runs match the published scores.")


if __name__ == "__main__":
    main()
