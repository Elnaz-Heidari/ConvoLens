"""
ConvoLens - Dataset Probe
Script: probe_datasets.py

Run this BEFORE collect_curated_v4.py to check which datasets
actually load on your machine. Takes ~2-3 minutes.

Run from your ConvoLens folder (venv activated):
    python src/probe_datasets.py
"""

from datasets import load_dataset

CANDIDATES = [
    # ETHICAL alternatives
    {
        "name": "declare-lab/HarmfulQA",
        "topic": "ethical",
        "split": "train",
        "kwargs": {},
        "fields": ["question", "response"]
    },
    {
        "name": "allenai/prosocial-dialog",
        "topic": "ethical",
        "split": "train",
        "kwargs": {"trust_remote_code": True},
        "fields": ["context", "response"]
    },
    {
        "name": "allenai/moral_stories",
        "topic": "ethical",
        "split": "train",
        "kwargs": {},
        "fields": ["situation", "moral"]
    },
    {
        "name": "demelin/moral_stories",
        "topic": "ethical",
        "split": "train",
        "kwargs": {},
        "fields": ["situation", "moral"]
    },
    # CULTURAL alternatives
    {
        "name": "Anthropic/llm_global_opinions",
        "topic": "cultural",
        "split": "train",
        "kwargs": {},
        "fields": ["question", "options"]
    },
    {
        "name": "social_i_qa",
        "topic": "cultural",
        "split": "train",
        "kwargs": {},
        "fields": ["context", "question", "answerA"]
    },
    {
        "name": "hendrycks/ethics",
        "topic": "cultural",
        "split": "train",
        "kwargs": {"name": "commonsense", "trust_remote_code": True},
        "fields": ["input", "label"]
    },
    {
        "name": "worldbank/world_development_indicators",
        "topic": "cultural",
        "split": "train",
        "kwargs": {},
        "fields": []
    },
    {
        "name": "Helsinki-NLP/europarl_st",
        "topic": "cultural",
        "split": "train",
        "kwargs": {"name": "en"},
        "fields": []
    },
]

def probe():
    print("\n" + "="*60)
    print("  ConvoLens — Dataset Availability Probe")
    print("="*60)
    print("\nTesting each dataset (this may take 2-3 minutes)...\n")

    working = []
    failed = []

    for c in CANDIDATES:
        name = c["name"]
        print(f"  Testing {name}...", end=" ", flush=True)
        try:
            ds = load_dataset(name, split=c["split"], **c["kwargs"])
            sample = ds[0]
            fields_found = list(sample.keys())
            print(f"✅  ({len(ds):,} rows) | fields: {fields_found[:5]}")
            working.append({**c, "rows": len(ds), "actual_fields": fields_found})
        except Exception as e:
            short_err = str(e)[:80]
            print(f"❌  {short_err}")
            failed.append(name)

    print("\n" + "="*60)
    print(f"  Results: {len(working)} working | {len(failed)} failed")
    print("="*60)

    print("\n✅ WORKING DATASETS:")
    for w in working:
        print(f"   [{w['topic'].upper()}] {w['name']}")
        print(f"           rows: {w['rows']:,} | fields: {w['actual_fields'][:5]}")

    print(f"\n❌ FAILED: {', '.join(failed)}")
    print("\n📋 Copy and send this output to Claude.")
    print("   Claude will write collect_curated_v4.py using only the working ones.\n")

if __name__ == "__main__":
    probe()
