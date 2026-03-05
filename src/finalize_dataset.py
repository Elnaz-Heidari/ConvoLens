"""
ConvoLens - Week 1: Finalize Dataset
Script: finalize_dataset.py

Run this AFTER you have manually reviewed curation_pool.csv in Excel
and marked your favourite 20 per topic with YES in the 'selected' column.

This script:
  1. Reads your selections from data/raw/curation_pool.csv
  2. Validates you have exactly 20 per topic (warns if not)
  3. Re-assigns clean IDs
  4. Saves your final 60-conversation dataset to data/processed/final_dataset.csv

Run from your ConvoLens folder (venv activated):
    python src/finalize_dataset.py
"""

import pandas as pd
import os

POOL_PATH  = "data/raw/curation_pool_v4_selected.csv"
OUTPUT_PATH = "data/processed/final_dataset.csv"
TARGET_PER_TOPIC = 20

def finalize():
    print("\n" + "="*60)
    print("  ConvoLens — Finalizing Dataset from Your Selections")
    print("="*60)

    # ── Load the curation pool ──
    if not os.path.exists(POOL_PATH):
        print(f"\n❌ Cannot find {POOL_PATH}")
        print("   Please run collect_curated_v3.py first.")
        return

    df = pd.read_csv(POOL_PATH)

    # ── Check for selected column ──
    if "selected" not in df.columns:
        print("\n❌ No 'selected' column found.")
        print("   Please open the CSV in Excel and add YES to your chosen rows.")
        return

    # ── Filter selected rows ──
    # Fill NaN with empty string before string operations
    df["selected"] = df["selected"].fillna("").astype(str)
    selected = df[df["selected"].str.upper().str.strip() == "YES"].copy()

    if len(selected) == 0:
        print("\n❌ No rows marked as YES in the 'selected' column.")
        print("   Open data/raw/curation_pool.csv in Excel and mark your choices.")
        return

    # ── Validate counts per topic ──
    print("\n📊 Your selections:")
    all_good = True
    for topic in ["medical", "ethical", "cultural"]:
        count = len(selected[selected["topic_category"] == topic])
        status = "✅" if count == TARGET_PER_TOPIC else "⚠️ "
        print(f"   {status} {topic:<10} {count} selected (target: {TARGET_PER_TOPIC})")
        if count != TARGET_PER_TOPIC:
            all_good = False

    if not all_good:
        print("\n⚠️  Warning: Some topics don't have exactly 20 selections.")
        print("   You can continue anyway, or go back and adjust your selections.")
        answer = input("   Continue anyway? (yes/no): ").strip().lower()
        if answer != "yes":
            print("   Okay! Go back to Excel, adjust your YES marks, and run again.")
            return

    # ── Re-assign clean IDs ──
    os.makedirs("data/processed", exist_ok=True)
    selected = selected.reset_index(drop=True)
    selected["id"] = [
        f"{row['topic_category'][:2]}_{i+1:03d}"
        for i, row in selected.iterrows()
    ]

    # Drop the 'selected' column from final output (not needed anymore)
    final = selected.drop(columns=["selected"], errors="ignore")

    final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    # ── Summary ──
    print(f"\n✅ Final dataset saved!")
    print(f"   {len(final)} conversations → {OUTPUT_PATH}")
    print(f"\n📋 Breakdown:")
    for topic in ["medical", "ethical", "cultural"]:
        subset = final[final["topic_category"] == topic]
        print(f"\n   {topic.upper()} ({len(subset)} rows):")
        for sub, grp in subset.groupby("sub_category"):
            print(f"     • {sub}: {len(grp)}")

    print(f"""
🎉 You're ready for Week 2: Manual Annotation!

   Open data/processed/final_dataset.csv and start filling in:
   - epistemic_positioning
   - pragmatic_appropriateness
   - ethical_handling
   - factual_accuracy
   - conversational_quality
   - overall_score
   - annotator_notes

   Come back to Claude and we'll build your annotation schema together!
""")

if __name__ == "__main__":
    finalize()
