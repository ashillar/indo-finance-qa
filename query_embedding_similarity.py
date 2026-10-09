"""Embedding (cosine) similarity between the queries of InternVL, Qwen and Gemma.

Same idea as query_similarity.py, but instead of asking an LLM judge, every query
is turned into a multilingual embedding vector and compared with cosine similarity.
It is cheap, deterministic and symmetric, so it is a good cross-check on the judge.

For each image present in all 3 models, every query of one model is compared with
every query of another model. Per model pair it reports:
  - avg best-match cosine: for each query, its highest cosine against the other model
  - avg one-to-one cosine: best assignment where each query is matched at most once
  - % queries with a similar match (best cosine >= --threshold)
  - % images with at least one similar match
If similarity_results/<source>_raw.jsonl (from query_similarity.py --all) exists,
the judge score is joined in and the correlation with cosine is reported.

Backends (pick one):
  gateway  OpenAI-compatible /embeddings on the LiteLLM gateway
           export LITELLM_API_KEY="sk-..."
           export EMBED_MODEL="<embedding model name on the gateway>"
  local    sentence-transformers on this machine (downloads the model once)
           pip install sentence-transformers
           default model: BAAI/bge-m3 (about 2.2 GB download on first run)
           smaller/faster: --model intfloat/multilingual-e5-small (about 470 MB)

Usage:
    python query_embedding_similarity.py --source bi
    python query_embedding_similarity.py --source ojk --backend local
    python query_embedding_similarity.py --limit 5 --threshold 0.7

Cosine values are not on the same scale as the 0-100 judge scores, and each embedding
model spreads them differently (e5 models sit high, roughly 0.7-1.0; bge-m3 is more
spread out). The script prints the distribution of best-match cosines; run once, look at
the median / percentiles, then choose --threshold. Use the same model for BI and OJK.
"""

import argparse
import json
import math
import os
from itertools import combinations, permutations
from statistics import mean, median

import requests

from query_similarity import (
    BASE_DIR, BASE_URL, MODELS, RESULTS_DIR, VERIFY_TLS, fmt, load_source, write_csv,
)

DEFAULT_LOCAL_MODEL = "BAAI/bge-m3"
DEFAULT_THRESHOLD = 0.80
BATCH = 64


# ---------------------------------------------------------------------------
# Embedding backends. Each returns one L2-normalised vector per input text.
# ---------------------------------------------------------------------------

def normalise(vec):
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def embed_gateway(texts, model, api_key):
    vectors = []
    for start in range(0, len(texts), BATCH):
        chunk = texts[start:start + BATCH]
        resp = requests.post(
            f"{BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": chunk},
            verify=VERIFY_TLS,
            timeout=180,
        )
        if not resp.ok:
            raise RuntimeError(f"Embeddings error {resp.status_code}: {resp.text[:500]}")
        rows = sorted(resp.json()["data"], key=lambda r: r["index"])
        if len(rows) != len(chunk):
            raise RuntimeError(f"Asked for {len(chunk)} embeddings, got {len(rows)}.")
        vectors.extend(normalise(r["embedding"]) for r in rows)
        print(f"  embedded {min(start + BATCH, len(texts))}/{len(texts)}")
    return vectors


def embed_local(texts, model):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("Local backend needs: pip install sentence-transformers")
    st = SentenceTransformer(model)
    # e5 models were trained with a "query: " prefix on short questions.
    if "e5" in model.lower():
        texts = [f"query: {t}" for t in texts]
    arr = st.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=BATCH)
    return [list(map(float, v)) for v in arr]


def embed_texts(texts, args):
    if args.backend == "gateway":
        api_key = os.environ.get("LITELLM_API_KEY")
        if not api_key:
            raise SystemExit("Set LITELLM_API_KEY (gateway backend).")
        model = args.model or os.environ.get("EMBED_MODEL")
        if not model:
            raise SystemExit("Give the gateway embedding model: --model NAME or EMBED_MODEL=NAME.")
        return embed_gateway(texts, model, api_key)
    return embed_local(texts, args.model or DEFAULT_LOCAL_MODEL)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def cosine(u, v):
    return sum(x * y for x, y in zip(u, v))


def one_to_one(matrix):
    """Mean cosine of the best assignment where each query is used at most once."""
    if not matrix or not matrix[0]:
        return None
    n, m = len(matrix), len(matrix[0])
    if n > m:  # make rows the smaller side
        matrix = [list(col) for col in zip(*matrix)]
        n, m = m, n
    if m > 7:  # brute force is fine for the usual 3x3; fall back to greedy if huge
        used, total = set(), 0.0
        for row in matrix:
            j = max((j for j in range(m) if j not in used), key=lambda j: row[j])
            used.add(j)
            total += row[j]
        return total / n
    best = max(sum(matrix[i][p[i]] for i in range(n)) for p in permutations(range(m), n))
    return best / n


def pearson(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def load_judge_scores(source):
    """(file, model_a, i, model_b, j) -> judge score, with model_a before model_b in MODELS order."""
    path = RESULTS_DIR / f"{source}_raw.jsonl"
    scores = {}
    if not path.exists():
        return scores
    order = {m: k for k, m in enumerate(MODELS)}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in row:
                continue
            for p in row.get("pairs", []):
                try:
                    a, i, b, j = p["model_a"], int(p["query_a"]), p["model_b"], int(p["query_b"])
                    score = float(p["score"])
                except (KeyError, TypeError, ValueError):
                    continue
                if order.get(a, 0) > order.get(b, 0):
                    a, i, b, j = b, j, a, i
                scores[(row["file"], a, i, b, j)] = score
    return scores


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["bi", "ojk"], default="bi")
    parser.add_argument("--backend", choices=["gateway", "local"],
                        help="Default: gateway if EMBED_MODEL is set, else local")
    parser.add_argument("--model", help="Embedding model name (gateway name, or Hugging Face id for local)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Cosine at or above which two queries count as similar (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--limit", type=int, help="Only the first N images")
    args = parser.parse_args()
    if not args.backend:
        args.backend = "gateway" if os.environ.get("EMBED_MODEL") else "local"

    data, common = load_source(args.source)
    targets = common[: args.limit] if args.limit else common
    pairs = list(combinations(MODELS, 2))

    # Embed every distinct query text once.
    texts = sorted({q for f in targets for m in MODELS for q in data[m][f] if q})
    print(f"{len(targets)} images, {len(texts)} distinct queries, backend={args.backend}")
    vectors = dict(zip(texts, embed_texts(texts, args)))
    judge = load_judge_scores(args.source)
    if judge:
        print(f"Joined {len(judge)} judge scores from {args.source}_raw.jsonl")

    totals = {p: {"images": 0, "queries": 0, "matched": 0, "images_with_match": 0,
                  "best": [], "o2o": [], "all": []} for p in pairs}
    per_image, long_rows = [], []
    judge_x, judge_y = {p: [] for p in pairs}, {p: [] for p in pairs}

    for f in targets:
        record = {"file": f}
        image_avgs = []
        for a, b in pairs:
            qa, qb = data[a][f], data[b][f]
            matrix = [[cosine(vectors[x], vectors[y]) if x and y else None for y in qb] for x in qa]
            valid_rows = [[c for c in row if c is not None] for row in matrix]
            best = [max(r) for r in valid_rows if r]
            t = totals[(a, b)]
            if best:
                matched = sum(c >= args.threshold for c in best)
                dense = [[c if c is not None else -1 for c in row] for row in matrix]
                o2o = one_to_one(dense)
                t["images"] += 1
                t["queries"] += len(best)
                t["matched"] += matched
                t["images_with_match"] += matched > 0
                t["best"].extend(best)
                t["o2o"].append(o2o)
                image_avgs.append(mean(best))
                record[f"{a}_vs_{b}_matched"] = f"{matched}/{len(best)}"
                record[f"{a}_vs_{b}_avg_best"] = round(mean(best), 3)
                record[f"{a}_vs_{b}_one_to_one"] = round(o2o, 3)
            else:
                record[f"{a}_vs_{b}_matched"] = ""
                record[f"{a}_vs_{b}_avg_best"] = ""
                record[f"{a}_vs_{b}_one_to_one"] = ""
            for i, row in enumerate(matrix, 1):
                for j, c in enumerate(row, 1):
                    if c is None:
                        continue
                    t["all"].append(c)
                    js = judge.get((f, a, i, b, j))
                    if js is not None:
                        judge_x[(a, b)].append(c)
                        judge_y[(a, b)].append(js)
                    long_rows.append({
                        "file": f, "model_a": a, "query_a_no": i, "query_a": qa[i - 1],
                        "model_b": b, "query_b_no": j, "query_b": qb[j - 1],
                        "cosine": round(c, 4),
                        "similar": int(c >= args.threshold),
                        "judge_score": "" if js is None else js,
                    })
        record["overall_avg_best"] = round(mean(image_avgs), 3) if image_avgs else ""
        per_image.append(record)
    per_image.sort(key=lambda r: -(r["overall_avg_best"] or 0))

    summary = []
    for a, b in pairs:
        t = totals[(a, b)]
        r = pearson(judge_x[(a, b)], judge_y[(a, b)])
        sb = sorted(t["best"])
        summary.append({
            "model_pair": f"{a} vs {b}",
            "images": t["images"],
            "avg_all_pairs_cosine": fmt(mean(t["all"]), 3) if t["all"] else "",
            "avg_best_match_cosine": fmt(mean(t["best"]), 3) if t["best"] else "",
            "avg_one_to_one_cosine": fmt(mean(t["o2o"]), 3) if t["o2o"] else "",
            "best_p25": fmt(percentile(sb, 0.25), 3),
            "best_median": fmt(median(sb), 3) if sb else "",
            "best_p75": fmt(percentile(sb, 0.75), 3),
            "pct_queries_with_similar_match": fmt(100 * t["matched"] / t["queries"]) if t["queries"] else "",
            "pct_images_with_any_match": fmt(100 * t["images_with_match"] / t["images"]) if t["images"] else "",
            "corr_with_judge": "" if r is None else round(r, 3),
        })

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"{args.source}_emb"
    outputs = [
        (RESULTS_DIR / f"{tag}_summary.csv", summary),
        (RESULTS_DIR / f"{tag}_per_image.csv", per_image),
        (RESULTS_DIR / f"{tag}_pairs.csv", long_rows),
    ]
    for path, table in outputs:
        write_csv(path, table)

    print(f"\n=== {args.source}_single: embedding similarity over {len(targets)} images "
          f"(similar = cosine >= {args.threshold}) ===\n")
    headers = ["Model pair", "Images", "Avg best", "1-to-1", "Median", "% queries", "% images", "r vs judge"]
    widths = [20, 8, 10, 9, 9, 11, 10, 11]
    print("".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))
    for r in summary:
        cells = [r["model_pair"], r["images"], r["avg_best_match_cosine"], r["avg_one_to_one_cosine"],
                 r["best_median"], r["pct_queries_with_similar_match"], r["pct_images_with_any_match"],
                 r["corr_with_judge"]]
        print("".join(str(c).ljust(w) for c, w in zip(cells, widths)))
    print("\n'% queries' and '% images' depend on --threshold; 'Avg best', '1-to-1' and 'Median' do not.")
    if not judge:
        print(f"(No judge scores found: run query_similarity.py --all --source {args.source} "
              "to fill 'r vs judge'.)")

    print("\nTables written:")
    for path, _ in outputs:
        print(f"  {path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
