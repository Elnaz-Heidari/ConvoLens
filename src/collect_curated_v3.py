"""
ConvoLens - Week 1: Curated Data Collection (Version 3 - Full Curation Pool)
Script: collect_curated_v3.py

STRATEGY:
  Collect ~45 conversations per topic (135 total) from diverse, high-quality
  sources. You then MANUALLY pick the best 20 per topic = 60 final dataset.
  This gives you curatorial control and richer annotation variety.

SOURCES:

  MEDICAL (target: 45)
  ├── keivalya/MedQuad-MedicalQnADataset  (symptoms, treatment, causes)
  └── lavita/ChatDoctor-HealthCareMagic-100k (real patient-doctor chat format)

  ETHICAL (target: 45, diverse themes — NOT just mental health)
  ├── Amod/mental_health_counseling_conversations (~15, emotional ethics)
  ├── allenai/prosocial-dialog (~15, everyday social ethics & fairness)
  └── PKU-Alignment/PKU-SafeRLHF (~15, AI safety & harm/benefit dilemmas)

  CULTURAL (target: 45)
  ├── hendrycks/ethics commonsense (~20, social norms across cultures)
  └── Lots12/cross_cultural_value_QA (~25, explicit cross-cultural values)

OUTPUT:
  data/raw/curation_pool.csv  ← your full pool (135 rows)
  
NEXT STEP (manual):
  Open curation_pool.csv in Excel, review each row, and mark your
  favourite 20 per topic in the "selected" column (put YES/NO).
  Then run: python src/finalize_dataset.py
  to extract your chosen 60 into the final dataset.

Run from your ConvoLens folder (venv activated):
    python src/collect_curated_v3.py
"""

import pandas as pd
import os
import re
from datasets import load_dataset

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

TARGET_PER_TOPIC = 45
OUTPUT_PATH = "data/raw/curation_pool.csv"
MIN_QUESTION_WORDS = 15
MIN_ANSWER_WORDS = 40

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def word_count(text):
    return len(str(text).split())

def is_substantive(question, answer="placeholder"):
    q = str(question).strip()
    a = str(answer).strip()
    if word_count(q) < MIN_QUESTION_WORDS:
        return False
    if answer != "placeholder" and word_count(a) < MIN_ANSWER_WORDS:
        return False
    low_value = ["i don't know", "i cannot help", "n/a", "none", "not applicable"]
    if any(p in a.lower() for p in low_value) and word_count(a) < 60:
        return False
    return True

def print_progress(topic, source, collected, target):
    pct = int((collected / target) * 30)
    bar = "█" * pct + "░" * (30 - pct)
    print(f"   [{bar}] {collected}/{target} — {source}")


# ─────────────────────────────────────────────
# MEDICAL SOURCE 1: MedQuAD
# ─────────────────────────────────────────────

def collect_medquad(target):
    rows = []
    try:
        ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
        priority = ["treatment", "symptoms", "causes", "prevention", "outlook"]
        seen = set()

        for item in ds:
            if len(rows) >= target:
                break
            q = clean(item.get("Question", ""))
            a = clean(item.get("Answer", ""))
            qtype = str(item.get("qtype", "general")).lower()
            if q in seen or not is_substantive(q, a):
                continue
            if qtype not in priority:
                continue
            rows.append({"question": q, "answer": a,
                         "sub_category": qtype,
                         "source_dataset": "MedQuAD"})
            seen.add(q)

        # Fill remaining with any type
        for item in ds:
            if len(rows) >= target:
                break
            q = clean(item.get("Question", ""))
            a = clean(item.get("Answer", ""))
            if q in seen or not is_substantive(q, a):
                continue
            rows.append({"question": q, "answer": a,
                         "sub_category": str(item.get("qtype", "general")).lower(),
                         "source_dataset": "MedQuAD"})
            seen.add(q)

        print_progress("medical", "MedQuAD", len(rows), target)
    except Exception as e:
        print(f"   ⚠️  MedQuAD failed: {e}")
    return rows


# ─────────────────────────────────────────────
# MEDICAL SOURCE 2: ChatDoctor HealthCareMagic
# ─────────────────────────────────────────────

def collect_chatdoctor(target):
    rows = []
    try:
        ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
        seen = set()

        for item in ds:
            if len(rows) >= target:
                break
            q = clean(item.get("input", ""))
            a = clean(item.get("output", ""))
            if q in seen or not is_substantive(q, a):
                continue
            # Skip if it's just "thank you" or follow-up messages
            if word_count(q) < 20:
                continue
            rows.append({"question": q, "answer": a,
                         "sub_category": "patient_doctor_chat",
                         "source_dataset": "ChatDoctor-HealthCareMagic"})
            seen.add(q)

        print_progress("medical", "ChatDoctor", len(rows), target)
    except Exception as e:
        print(f"   ⚠️  ChatDoctor failed: {e}")
    return rows


# ─────────────────────────────────────────────
# ETHICAL SOURCE 1: Mental Health Counseling
# ─────────────────────────────────────────────

def collect_mental_health(target):
    rows = []
    try:
        ds = load_dataset("Amod/mental_health_counseling_conversations", split="train")
        seen = set()

        for item in ds:
            if len(rows) >= target:
                break
            q = clean(item.get("Context", ""))
            a = clean(item.get("Response", ""))
            if q in seen or not is_substantive(q, a):
                continue

            q_lower = q.lower()
            if any(w in q_lower for w in ["suicide", "self-harm", "hurt myself"]):
                sub = "crisis_intervention"
            elif any(w in q_lower for w in ["relationship", "partner", "marriage", "abuse"]):
                sub = "relationship_ethics"
            elif any(w in q_lower for w in ["anxiety", "depression", "trauma", "ptsd"]):
                sub = "mental_health_support"
            else:
                sub = "general_counseling"

            rows.append({"question": q, "answer": a,
                         "sub_category": sub,
                         "source_dataset": "MentalHealth-Counseling"})
            seen.add(q)

        print_progress("ethical", "Mental Health Counseling", len(rows), target)
    except Exception as e:
        print(f"   ⚠️  Mental Health dataset failed: {e}")
    return rows


# ─────────────────────────────────────────────
# ETHICAL SOURCE 2: Prosocial Dialog (Everyday Ethics)
# ─────────────────────────────────────────────

def collect_prosocial(target):
    rows = []
    try:
        ds = load_dataset("allenai/prosocial-dialog", split="train",
                          trust_remote_code=True)
        seen = set()

        for item in ds:
            if len(rows) >= target:
                break

            # prosocial-dialog format: context (list of turns) + response
            context = item.get("context", [])
            response = clean(item.get("response", ""))
            rots = item.get("rots", [])  # "rules of thumb" — ethical principles cited

            if not context or not response:
                continue

            # Use the last user turn as the question
            last_user = clean(context[-1]) if context else ""

            # Build richer question with prior context if available
            if len(context) >= 2:
                prior = clean(context[-2])
                question = f"{prior}\n{last_user}"
            else:
                question = last_user

            if question in seen or not is_substantive(question, response):
                continue

            # Sub-categorize by ethical theme using rules of thumb
            rot_text = " ".join(rots).lower() if rots else ""
            if any(w in rot_text for w in ["honest", "lie", "deceive", "truth"]):
                sub = "honesty_and_deception"
            elif any(w in rot_text for w in ["fair", "equal", "discriminat", "justice"]):
                sub = "fairness_and_justice"
            elif any(w in rot_text for w in ["harm", "hurt", "safe", "danger"]):
                sub = "harm_prevention"
            elif any(w in rot_text for w in ["respect", "dignity", "consent", "privacy"]):
                sub = "respect_and_consent"
            else:
                sub = "social_responsibility"

            rows.append({"question": question, "answer": response,
                         "sub_category": sub,
                         "source_dataset": "ProSocial-Dialog"})
            seen.add(question)

        print_progress("ethical", "ProSocial-Dialog", len(rows), target)
    except Exception as e:
        print(f"   ⚠️  ProSocial-Dialog failed: {e}")
    return rows


# ─────────────────────────────────────────────
# ETHICAL SOURCE 3: PKU-SafeRLHF (AI Safety Dilemmas)
# ─────────────────────────────────────────────

def collect_pku_safety(target):
    rows = []
    try:
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train",
                          trust_remote_code=True)
        seen = set()

        for item in ds:
            if len(rows) >= target:
                break

            q = clean(item.get("prompt", ""))
            # Use the safer/better response
            a = clean(item.get("response_1", ""))
            is_safe = item.get("is_response_1_safe", True)

            if not is_safe:
                a = clean(item.get("response_2", ""))

            if q in seen or not is_substantive(q, a):
                continue

            # These are AI safety dilemmas — great for ethical_handling annotation
            rows.append({"question": q, "answer": a,
                         "sub_category": "ai_safety_and_harm",
                         "source_dataset": "PKU-SafeRLHF"})
            seen.add(q)

        print_progress("ethical", "PKU-SafeRLHF", len(rows), target)
    except Exception as e:
        print(f"   ⚠️  PKU-SafeRLHF failed: {e}")
    return rows


# ─────────────────────────────────────────────
# CULTURAL SOURCE 1: Hendrycks Ethics (Social Norms)
# ─────────────────────────────────────────────

def collect_hendrycks(target):
    rows = []
    try:
        ds = load_dataset("hendrycks/ethics", "commonsense", split="train",
                          trust_remote_code=True)
        seen = set()
        acceptable, unacceptable = [], []

        for item in ds:
            scenario = clean(item.get("input", ""))
            label = item.get("label", -1)
            if item.get("is_short", True) or word_count(scenario) < 20:
                continue
            if scenario in seen:
                continue
            if label == 1:
                acceptable.append(scenario)
            elif label == 0:
                unacceptable.append(scenario)
            seen.add(scenario)

        half = target // 2
        selected = acceptable[:half] + unacceptable[:half]

        for i, scenario in enumerate(selected[:target]):
            question = (
                f"I'd like your perspective on this situation from a cultural "
                f"and ethical standpoint: {scenario} "
                f"How would you evaluate this behavior in terms of social norms "
                f"and cultural expectations? Is it generally considered acceptable?"
            )
            rows.append({
                "question": question,
                "answer": "",
                "sub_category": "social_norms",
                "source_dataset": "Hendrycks-Ethics",
                "ethics_label": "acceptable" if i < half else "unacceptable"
            })

        print_progress("cultural", "Hendrycks-Ethics", len(rows), target)
    except Exception as e:
        print(f"   ⚠️  Hendrycks Ethics failed: {e}")
    return rows


# ─────────────────────────────────────────────
# CULTURAL SOURCE 2: Cross-Cultural Values QA
# ─────────────────────────────────────────────

def collect_cross_cultural(target):
    rows = []
    try:
        ds = load_dataset("Lots12/cross_cultural_value_QA", split="train",
                          trust_remote_code=True)
        seen = set()

        for item in ds:
            if len(rows) >= target:
                break

            # Try common field names for this dataset
            q = clean(item.get("question", item.get("input", item.get("prompt", ""))))
            a = clean(item.get("answer", item.get("output", item.get("response", ""))))

            if q in seen or not is_substantive(q):
                continue

            rows.append({
                "question": q,
                "answer": a,
                "sub_category": "cross_cultural_values",
                "source_dataset": "CrossCultural-ValueQA",
                "ethics_label": ""
            })
            seen.add(q)

        print_progress("cultural", "CrossCultural-ValueQA", len(rows), target)
    except Exception as e:
        print(f"   ⚠️  Cross-Cultural QA failed: {e}")
    return rows


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def build_pool():
    print("\n" + "="*60)
    print("  ConvoLens — Building Curation Pool (v3)")
    print("  Target: ~45 per topic | You pick best 20 per topic")
    print("="*60)

    os.makedirs("data/raw", exist_ok=True)

    # ── MEDICAL ──
    print("\n🏥 Collecting MEDICAL conversations...")
    med1 = collect_medquad(25)
    med2 = collect_chatdoctor(20)
    medical_all = med1 + med2
    print(f"   → Medical total: {len(medical_all)}")

    # ── ETHICAL ──
    print("\n⚖️  Collecting ETHICAL conversations (3 diverse sources)...")
    eth1 = collect_mental_health(15)
    eth2 = collect_prosocial(15)
    eth3 = collect_pku_safety(15)
    ethical_all = eth1 + eth2 + eth3
    print(f"   → Ethical total: {len(ethical_all)}")

    # ── CULTURAL ──
    print("\n🌍 Collecting CULTURAL conversations...")
    cul1 = collect_hendrycks(20)
    cul2 = collect_cross_cultural(25)
    cultural_all = cul1 + cul2
    print(f"   → Cultural total: {len(cultural_all)}")

    # ── BUILD DATAFRAME ──
    rows = []
    all_groups = [
        ("medical",  medical_all),
        ("ethical",  ethical_all),
        ("cultural", cultural_all),
    ]

    for topic, items in all_groups:
        for i, item in enumerate(items):
            row = {
                "id":                       f"{topic[:2]}_{i+1:03d}",
                "topic_category":           topic,
                "sub_category":             item.get("sub_category", ""),
                "source_dataset":           item.get("source_dataset", ""),
                "user_message":             item["question"],
                "ai_response":              item.get("answer", ""),
                "model":                    "human_expert" if item.get("answer") else "to_be_generated",
                "word_count_response":      word_count(item.get("answer", "")),
                # ── YOUR CURATION COLUMN ──
                # Open CSV in Excel and put YES for your favourite 20 per topic
                "selected":                 "",
                # ── ANNOTATION COLUMNS (fill in Week 2) ──
                "epistemic_positioning":    "",
                "pragmatic_appropriateness":"",
                "ethical_handling":         "",
                "factual_accuracy":         "",
                "conversational_quality":   "",
                "overall_score":            "",
                "annotator_notes":          ""
            }
            # Extra metadata for cultural rows
            if "ethics_label" in item:
                row["ethics_label"] = item.get("ethics_label", "")
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    # ── SUMMARY ──
    print(f"\n{'='*60}")
    print(f"✅ Curation pool saved!")
    print(f"\n📊 Summary:")
    print(f"   Total rows collected: {len(df)}")
    print(f"   Saved to: {OUTPUT_PATH}")
    print(f"\n   By topic & source:")
    for topic in ["medical", "ethical", "cultural"]:
        subset = df[df["topic_category"] == topic]
        print(f"\n   {topic.upper()} ({len(subset)} rows):")
        for src, grp in subset.groupby("source_dataset"):
            print(f"     • {src}: {len(grp)} rows")

    print(f"""
📋 YOUR NEXT STEP — Manual Curation:
   1. Open data/raw/curation_pool.csv in Excel
   2. Read through each conversation
   3. In the 'selected' column, type YES for your best 20 per topic
      (look for: substantive questions, rich AI responses, 
       clear linguistic phenomena worth annotating)
   4. Come back and tell Claude — we'll run finalize_dataset.py
      to extract your 60 chosen conversations into the final file!
""")


if __name__ == "__main__":
    build_pool()
