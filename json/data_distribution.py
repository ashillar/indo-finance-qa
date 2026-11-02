
import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path("/Users/ashillafryda/Downloads/indo-finance-qa/json")

# Expected filenames. If a filename differs, the script searches
# for a matching JSONL file in BASE_DIR.
FILES = {
    "BI": {
        "Gemma": "bi_multipage_gemma_v4.jsonl",
        "InternVL": "bi_multipage_internvl_v4.jsonl",
        "Qwen": "bi_multipage_qwen_v4_new.jsonl",
    },
    "OJK": {
        "Gemma": "ojk_multipage_gemma_v4.jsonl",
        "InternVL": "ojk_multipage_internvl_v4.jsonl",
        "Qwen": "ojk_multipage_qwen_vl_v4.jsonl",
    },
}


def resolve_file(filename, dataset, model):
    """Find the requested file, with fallback matching for filenames."""
    exact_path = BASE_DIR / filename

    if exact_path.is_file():
        return exact_path

    # Match dataset and model names while allowing different suffixes.
    dataset_prefix = dataset.lower() + "_multipage_"
    model_key = model.lower()

    candidates = [
        path for path in BASE_DIR.glob("*.jsonl")
        if path.name.lower().startswith(dataset_prefix)
        and model_key in path.name.lower()
    ]

    if len(candidates) == 1:
        print(f"Using detected file for {dataset}/{model}: {candidates[0].name}")
        return candidates[0]

    if len(candidates) > 1:
        # Prefer the expected version if possible, otherwise the most
        # likely versioned filename.
        preferred = [
            path for path in candidates
            if path.name.lower().endswith("_v4_new.jsonl")
        ]
        if len(preferred) == 1:
            print(f"Using detected file for {dataset}/{model}: {preferred[0].name}")
            return preferred[0]

        print(f"WARNING: Multiple files found for {dataset}/{model}:")
        for path in candidates:
            print(f"  - {path.name}")
        print("Please specify the correct filename in FILES.")

    else:
        print(f"WARNING: No file found for {dataset}/{model} (expected {filename})")

    return None


def get_category_counts(file_path):
    """Count QA pairs by category from nested qa.result lists."""
    counts = Counter()

    if file_path is None or not file_path.is_file():
        return counts

    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: Invalid JSON at "
                    f"{file_path.name}:{line_number}: {exc}"
                )
                continue

            qa = record.get("qa", {})

            if not isinstance(qa, dict):
                continue

            results = qa.get("result", [])

            if not isinstance(results, list):
                continue

            for item in results:
                if not isinstance(item, dict):
                    continue

                category = item.get("category") or "Unknown"
                counts[category] += 1

    return counts


def print_distribution(title, model_counts):
    """Print category counts for Gemma, InternVL, Qwen and their total."""
    models = ("Gemma", "InternVL", "Qwen")

    categories = sorted(
        set().union(
            *(model_counts[model].keys() for model in models)
        )
    )

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print(
        f"{'Category':45s}"
        f"{'Gemma':>12s}"
        f"{'InternVL':>12s}"
        f"{'Qwen':>12s}"
        f"{'Total':>12s}"
    )
    print("-" * 93)

    for category in categories:
        values = [
            model_counts[model].get(category, 0)
            for model in models
        ]

        print(
            f"{category[:45]:45s}"
            f"{values[0]:12,d}"
            f"{values[1]:12,d}"
            f"{values[2]:12,d}"
            f"{sum(values):12,d}"
        )

    model_totals = {
        model: sum(model_counts[model].values())
        for model in models
    }

    overall_total = sum(model_totals.values())

    print("-" * 93)
    print(
        f"{'TOTAL QA':45s}"
        f"{model_totals['Gemma']:12,d}"
        f"{model_totals['InternVL']:12,d}"
        f"{model_totals['Qwen']:12,d}"
        f"{overall_total:12,d}"
    )

    return overall_total


def main():
    # Correctly initialize a Counter for every dataset and model.
    all_counts = {
        dataset: {
            model: Counter()
            for model in models
        }
        for dataset, models in FILES.items()
    }

    found_files = 0

    for dataset, models in FILES.items():
        for model, filename in models.items():
            file_path = resolve_file(filename, dataset, model)

            if file_path is None:
                continue

            found_files += 1
            all_counts[dataset][model] = get_category_counts(file_path)

    if found_files == 0:
        print(f"\nNo dataset files found in: {BASE_DIR}")
        return

    # BI and OJK distributions.
    dataset_totals = {}

    for dataset, model_counts in all_counts.items():
        dataset_totals[dataset] = print_distribution(
            f"{dataset} — QA CATEGORY DISTRIBUTION",
            model_counts,
        )

    # Combine BI and OJK counts for each model.
    combined_models = {
        model: Counter()
        for model in ("Gemma", "InternVL", "Qwen")
    }

    for dataset_counts in all_counts.values():
        for model, counts in dataset_counts.items():
            combined_models[model].update(counts)

    combined_total = print_distribution(
        "BI + OJK — COMBINED CATEGORY DISTRIBUTION",
        combined_models,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for dataset, total in dataset_totals.items():
        print(f"{dataset + ' total QA':25s}: {total:,}")

    print(f"{'BI + OJK total QA':25s}: {combined_total:,}")
    print(f"{'Files found':25s}: {found_files}/{sum(len(m) for m in FILES.values())}")


if __name__ == "__main__":
    main()
