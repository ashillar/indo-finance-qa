"""Write a page-groups file for multi_qwen.py --groups-file.

Reads the page images in a folder (default: bi_multi), groups them by
document, splits each document into runs of nearby pages, and cuts each run
into windows of 2-3 pages. Edit the output by hand before using it.

    python make_groups.py
    python make_groups.py --folder ojk_multi --out ojk_multi_groups.txt
    python multi_qwen.py --folder bi_multi --groups-file bi_multi_groups.txt
"""

import argparse
import re
import sys
import urllib.parse
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PAGE_PATTERN = re.compile(r"^(?P<doc>.+)_p(?P<page>\d+)$", re.IGNORECASE)


def normalize_name(name):
    """Same name repair as multi_qwen.py, so document names match."""
    if name.count(" 20") >= 2:
        name = name.replace(" 20", " ")
    return re.sub(r"\s+", " ", urllib.parse.unquote(name)).strip()


def group_documents(folder):
    docs = defaultdict(list)
    skipped = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        match = PAGE_PATTERN.match(normalize_name(path.stem))
        if not match:
            skipped.append(path.name)
            continue
        doc = match.group("doc").strip().rstrip("_").strip()
        docs[doc].append(int(match.group("page")))
    return {doc: sorted(set(pages)) for doc, pages in docs.items()}, skipped


def split_runs(pages, max_gap):
    runs = [[pages[0]]]
    for page in pages[1:]:
        if page - runs[-1][-1] <= max_gap:
            runs[-1].append(page)
        else:
            runs.append([page])
    return runs


def make_windows(run, size, overlap):
    if len(run) <= size:
        return [run]
    step = size - overlap
    windows = []
    for start in range(0, len(run), step):
        window = run[start : start + size]
        if len(window) < 2:
            window = run[-2:]
        windows.append(window)
        if start + size >= len(run):
            break
    return windows


def format_group(pages):
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"{pages[0]}-{pages[-1]}"
    return ",".join(map(str, pages))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", default="bi_multi")
    parser.add_argument("--out", default=None, help="default: <folder>_groups.txt")
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument(
        "--max-page-gap",
        type=int,
        default=2,
        help="Pages further apart than this start a new group",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"ERROR: folder does not exist: {folder}")
    if args.group_size < 2:
        sys.exit("ERROR: --group-size must be at least 2")
    overlap = max(0, min(args.overlap, args.group_size - 1))
    out = Path(args.out or f"{folder.name}_groups.txt")

    docs, skipped = group_documents(folder)
    for name in skipped:
        print(f"WARNING: skipped {name} (not '<document>_pNNN')")

    lines, single_docs, isolated = [], [], []
    for doc in sorted(docs):
        groups = []
        for run in split_runs(docs[doc], args.max_page_gap):
            if len(run) < 2:
                isolated.append(f"{doc} p{run[0]}")
                continue
            groups.extend(make_windows(run, args.group_size, overlap))
        if groups:
            lines.append(f"{doc}: " + " ".join(map(format_group, groups)))
        else:
            single_docs.append(doc)

    # multi_qwen.py matches documents by substring, so warn when one name
    # is contained in another.
    for doc in docs:
        for other in docs:
            if doc != other and doc.lower() in other.lower():
                print(
                    f"WARNING: '{doc}' is also part of '{other}'; its line "
                    "will match both documents"
                )

    with out.open("w", encoding="utf-8") as stream:
        stream.write(
            f"# Page groups for {folder}. One line per document:\n"
            "#   <document>: <group> <group> ...\n"
            "# A group is a range (45-47) or a list (38,39,41), 2-3 pages.\n"
            "# Merge, split or delete groups so each one holds pages whose\n"
            "# content is related. Lines starting with # are ignored.\n"
        )
        if isolated:
            stream.write("#\n# Pages with no neighbour (not grouped):\n")
            for item in isolated:
                stream.write(f"#   {item}\n")
        stream.write("\n" + "\n".join(lines) + "\n")

    total = sum(line.count(" ") - line.split(":")[0].count(" ") for line in lines)
    print(
        f"Documents: {len(docs)} | grouped: {len(lines)} | groups: {total} | "
        f"isolated pages: {len(isolated)}"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
