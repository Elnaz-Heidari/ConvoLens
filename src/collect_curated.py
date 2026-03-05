"""
ConvoLens - Week 1: Curated Data Collection (Improved)
Script: collect_curated.py

WHY THIS IS BETTER THAN collect_sharegpt.py:
  Instead of filtering a general dataset with keyword matching (which gave us
  coding questions labeled as "ethical"!), we now pull from 3 purpose-built
  datasets — one per topic. Every row is guaranteed to be on-topic.

DATASETS USED:
  Medical  → keivalya/MedQuad-MedicalQnADataset
             Real patient questions + medical answers. Great for analyzing
             epistemic hedging ("you should consult a doctor", "may indicate...")

  Ethical  → Amod/mental_health_counseling_conversations
             Real counseling Q&A from licensed professionals. Rich in empathy
             markers, ethical boundary-setting, and sensitive topic handling.

  Cultural → hendrycks/ethics (commonsense subset)
             Real-world moral scenarios involving social norms and cultural
             expectations. We turn scenarios into user questions for annotation.

TARGET: 20 conversations per topic = 60 total (clean, annotatable dataset)

Run from your ConvoLens folder (with venv activated):
    python src/collect_curated.py
"""

import pandas as pd
import os
import re
from datasets import load_dataset

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

TARGET_PER_TOPIC = 20       # 20 x 3 topics = 60 total
OUTPUT_PATH = "data/raw/curated_60.csv"
MIN_QUESTION_WORDS = 15     # Filter out very short/vague questions
MIN_ANSWER_WORDS = 40       # Filter out very short answers


# ─────────────────────────────────────────────
# HELPER: Text cleaning
# ─────────────────────────────────────────────

def clean(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def word_count(text):
    return len(str(text).split())

def is_substantive(question, answer):
    """Check if a Q&A pair is worth annotating."""
    q = str(question).strip()
    a = str(answer).strip()
    # Must be long enough
    if word_count(q) < MIN_QUESTION_WORDS or word_count(a) < MIN_ANSWER_WORDS:
        return False
    # Question must end naturally (not a fragment)
    if len(q) < 20:
        return False
    # Answer shouldn't be just "I don't know" type responses
    low_value = ["i don't know", "i cannot", "i'm not sure", "n/a", "none"]
    if any(phrase in a.lower() for phrase in low_value) and word_count(a) < 60:
        return False
    return True


# ─────────────────────────────────────────────
# COLLECTOR 1: MEDICAL
# Dataset: keivalya/MedQuad-MedicalQnADataset
# Format: {"qtype": ..., "Question": ..., "Answer": ...}
# ─────────────────────────────────────────────

def collect_medical(target):
    print("\n📥 Loading Medical dataset (MedQuAD)...")
    try:
        ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
        print(f"   ✓ Loaded {len(ds):,} medical Q&A pairs")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []

    # Priority question types — these are richest for linguistic analysis
    # "treatment" and "symptoms" questions produce the most epistemic hedging
    priority_types = ["treatment", "symptoms", "causes", "prevention", "outlook"]

    collected = []
    seen_questions = set()

    # First pass: priority question types (richer for annotation)
    for item in ds:
        if len(collected) >= target:
            break
        qtype = str(item.get("qtype", "")).lower()
        question = clean(item.get("Question", ""))
        answer = clean(item.get("Answer", ""))

        if qtype not in priority_types:
            continue
        if question in seen_questions:
            continue
        if not is_substantive(question, answer):
            continue

        collected.append({
            "question": question,
            "answer": answer,
            "sub_category": qtype
        })
        seen_questions.add(question)

    # Second pass: fill remaining slots with any question type
    if len(collected) < target:
        for item in ds:
            if len(collected) >= target:
                break
            question = clean(item.get("Question", ""))
            answer = clean(item.get("Answer", ""))
            if question in seen_questions:
                continue
            if not is_substantive(question, answer):
                continue
            collected.append({
                "question": question,
                "answer": answer,
                "sub_category": str(item.get("qtype", "general")).lower()
            })
            seen_questions.add(question)

    print(f"   ✓ Collected {len(collected)} medical conversations")
    return collected


# ─────────────────────────────────────────────
# COLLECTOR 2: ETHICAL (Mental Health Counseling)
# Dataset: Amod/mental_health_counseling_conversations
# Format: {"Context": ..., "Response": ...}
# ─────────────────────────────────────────────

def collect_ethical(target):
    print("\n📥 Loading Ethical/Mental Health dataset...")
    try:
        ds = load_dataset("Amod/mental_health_counseling_conversations", split="train")
        print(f"   ✓ Loaded {len(ds):,} counseling conversations")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []

    collected = []
    seen = set()

    for item in ds:
        if len(collected) >= target:
            break

        question = clean(item.get("Context", ""))
        answer = clean(item.get("Response", ""))

        if question in seen:
            continue
        if not is_substantive(question, answer):
            continue

        # Categorize by emotional/ethical theme for richer annotation
        q_lower = question.lower()
        if any(w in q_lower for w in ["suicide", "self-harm", "hurt myself", "end my life"]):
            sub_cat = "crisis_intervention"
        elif any(w in q_lower for w in ["relationship", "partner", "marriage", "divorce", "abuse"]):
            sub_cat = "relationship_ethics"
        elif any(w in q_lower for w in ["anxiety", "depression", "panic", "ptsd", "trauma"]):
            sub_cat = "mental_health_support"
        elif any(w in q_lower for w in ["family", "parent", "child", "abuse", "neglect"]):
            sub_cat = "family_dynamics"
        else:
            sub_cat = "general_counseling"

        collected.append({
            "question": question,
            "answer": answer,
            "sub_category": sub_cat
        })
        seen.add(question)

    print(f"   ✓ Collected {len(collected)} ethical conversations")
    return collected


# ─────────────────────────────────────────────
# COLLECTOR 3: CULTURAL (Ethics / Social Norms)
# Dataset: hendrycks/ethics (commonsense subset)
# Format: {"input": <scenario>, "label": 0 or 1}
# Note: This dataset has scenarios, not Q&A pairs.
# We convert them into user questions for annotation.
# ─────────────────────────────────────────────

def collect_cultural(target):
    print("\n📥 Loading Cultural/Ethics dataset (Hendrycks ETHICS)...")
    try:
        ds = load_dataset("hendrycks/ethics", "commonsense", split="train",
                          trust_remote_code=True)
        print(f"   ✓ Loaded {len(ds):,} ethical scenarios")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []

    collected = []
    seen = set()

    # We want a mix of "acceptable" (label=1) and "unacceptable" (label=0)
    # behaviors — this gives richer annotation variety
    acceptable = []
    unacceptable = []

    for item in ds:
        scenario = clean(item.get("input", ""))
        label = item.get("label", -1)
        is_short = item.get("is_short", True)

        if scenario in seen:
            continue
        # Skip very short scenarios (not rich enough)
        if is_short or word_count(scenario) < 20:
            continue

        if label == 1:
            acceptable.append(scenario)
        elif label == 0:
            unacceptable.append(scenario)
        seen.add(scenario)

    # Take a balanced mix: half acceptable, half unacceptable behaviors
    half = target // 2
    selected = acceptable[:half] + unacceptable[:half]

    for i, scenario in enumerate(selected[:target]):
        # Convert scenario into a user question format
        # This makes it look like a real AI conversation for annotation
        user_question = (
            f"I'm trying to understand social norms and ethics. "
            f"Can you help me think through this situation: {scenario} "
            f"Is this behavior generally considered acceptable or problematic? "
            f"What ethical principles are at play here?"
        )
        sub_cat = "socially_acceptable" if i < half else "socially_problematic"

        collected.append({
            "question": user_question,
            "answer": "",   # Will be filled by Claude API in Week 3 — or you can manually add
            "sub_category": sub_cat,
            "original_scenario": scenario,
            "ethics_label": "acceptable" if i < half else "unacceptable"
        })

    print(f"   ✓ Collected {len(collected)} cultural/ethics scenarios")
    print(f"   ℹ️  Note: Cultural entries have no AI answer yet.")
    print(f"      We will generate answers using Claude API in Week 3,")
    print(f"      OR you can manually write the AI response for Week 2 annotation.")
    return collected


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def build_dataset():
    print("\n" + "="*60)
    print("  ConvoLens — Curated Data Collection (Round 2)")
    print("="*60)

    os.makedirs("data/raw", exist_ok=True)

    # ── Collect from all 3 sources ──
    medical  = collect_medical(TARGET_PER_TOPIC)
    ethical  = collect_ethical(TARGET_PER_TOPIC)
    cultural = collect_cultural(TARGET_PER_TOPIC)

    # ── Build unified dataframe ──
    rows = []
    counters = {"medical": 0, "ethical": 0, "cultural": 0}

    def add_rows(items, topic):
        for item in items:
            counters[topic] += 1
            row = {
                "id": f"{topic[:2]}_{counters[topic]:03d}",
                "source": {
                    "medical":  "MedQuAD (keivalya/MedQuad-MedicalQnADataset)",
                    "ethical":  "Mental Health Counseling (Amod)",
                    "cultural": "Hendrycks ETHICS (commonsense)"
                }[topic],
                "topic_category":   topic,
                "sub_category":     item.get("sub_category", ""),
                "user_message":     item["question"],
                "ai_response":      item["answer"],
                "model":            "human_expert" if topic != "cultural" else "to_be_generated",
                "word_count_response": word_count(item["answer"]),
                # Annotation columns — fill these in Week 2!
                "epistemic_positioning":    "",
                "pragmatic_appropriateness": "",
                "ethical_handling":          "",
                "factual_accuracy":          "",
                "conversational_quality":    "",
                "overall_score":             "",
                "annotator_notes":           ""
            }
            # Add extra fields for cultural entries
            if topic == "cultural":
                row["original_scenario"] = item.get("original_scenario", "")
                row["ethics_label"]      = item.get("ethics_label", "")
            rows.append(row)

    add_rows(medical,  "medical")
    add_rows(ethical,  "ethical")
    add_rows(cultural, "cultural")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"✅ Curated dataset ready!")
    print(f"\n📊 Summary:")
    print(f"   Total conversations: {len(df)}")
    print(f"   Saved to: {OUTPUT_PATH}")
    print(f"\n   By topic:")
    for topic, count in counters.items():
        bar = "█" * count
        print(f"   {topic:<10} {bar} {count}")

    print(f"\n📋 Sample from each topic:")
    for topic in ["medical", "ethical", "cultural"]:
        sample = df[df["topic_category"] == topic].iloc[0]
        print(f"\n  [{topic.upper()}] — {sample['sub_category']}")
        print(f"  User: {str(sample['user_message'])[:180]}...")
        if sample['ai_response']:
            print(f"  AI:   {str(sample['ai_response'])[:180]}...")
        else:
            print(f"  AI:   (to be generated in Week 3)")

    print(f"\n⚠️  Important note about Cultural entries:")
    print(f"   The cultural rows don't have AI responses yet.")
    print(f"   Two options:")
    print(f"   A) Use them in Week 3 when we generate responses via Claude API")
    print(f"   B) For now, focus annotation practice on medical + ethical (40 rows)")
    print(f"\n🎉 Next step: Open data/raw/curated_60.csv and review the quality.")
    print(f"   Then tell Claude your impressions — we'll start annotation schema!\n")


if __name__ == "__main__":
    build_dataset()
