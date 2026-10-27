"""Inspect per-page extracted facts and cross-page candidate pairs for a group (CPU only, reads extraction cache).

Usage:
    python debug_pairs.py --folder bi_multi --groups-file bi_multi_groups.txt \
        --key "KEKR Provinsi Bali Agustus 2017::p070-p071"
"""

import argparse
import itertools
import sys
from pathlib import Path

import multi_qwen_v2 as m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--groups-file", required=True)
    parser.add_argument("--key", action="append", required=True, help="Group key (repeatable)")
    parser.add_argument("--extract-cache", default=f"{Path(m.DEFAULT_OUT).with_suffix('')}.extract.jsonl")
    parser.add_argument("--model", default=m.DEFAULT_MODEL)
    parser.add_argument("--extract-max-pixels", type=int, default=1605632)
    parser.add_argument("--extract-passes", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=6)
    args = parser.parse_args()
    args.doc = None

    units, _ = m.resolve_units(args, Path(args.folder))
    by_key = {key: (doc, window) for key, doc, window in units}
    cache = m.load_extract_cache(Path(args.extract_cache))
    print(f"cache entries: {len(cache)} from {args.extract_cache}")

    for key in args.key:
        if key not in by_key:
            print(f"\n### {key}: not in groups file")
            continue
        doc, window = by_key[key]
        print(f"\n### {key}")
        infos = {}
        for page, path in window:
            entry = m.get_cached_entry(cache, path, args)
            if entry is None:
                sys.exit(f"no cache entry for {path}")
            raw = [len(p.get("facts") or []) for p in entry.get("passes") or []]
            trunc = [p.get("truncated") for p in entry.get("passes") or []]
            infos[page] = info = m.build_page_info(entry, page)
            print(f"  p{page}: raw facts per reading {raw} (truncated {trunc}) -> clean {info['raw_counts']} -> kept {len(info['facts'])}")
            for f in info["facts"]:
                val = f"{f['value']} {f['unit']}".strip() if f["value"] else "(narrative)"
                print(f"     [{f['type']}] {f['subject']} = {val} ({f['period']}) | {f['statement'][:80]}")

        job = {"doc": doc, "pages": list(infos)}
        cands = m.build_candidates(job, infos, args.max_candidates)
        print(f"  candidates q1/q2/q3: {[len(cands[s]) for s in (1, 2, 3)]}")
        for s in (1, 2, 3):
            for c in cands[s][:2]:
                print(f"    [q{s} | {c['category']} | p{c['pa']}-p{c['pb']}] {c['fa']['subject']} vs {c['fb']['subject']}")
                print(f"      -> {c['reference']}")

        doc_tokens = frozenset(m.topic_tokens(doc.replace("_", " ").replace("-", " ")))
        for pa, pb in itertools.combinations(job["pages"], 2):
            a, b = infos[pa], infos[pb]
            ua = [f for f in a["facts"] if m.unique_to_page(f, b)]
            ub = [f for f in b["facts"] if m.unique_to_page(f, a)]
            print(f"  p{pa}-p{pb}: unique to p{pa}: {len(ua)}/{len(a['facts'])}, unique to p{pb}: {len(ub)}/{len(b['facts'])}")


if __name__ == "__main__":
    main()
