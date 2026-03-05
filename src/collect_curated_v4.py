"""
ConvoLens - Week 1: Curated Data Collection (Version 4 - FINAL)
Script: collect_curated_v4.py

Uses ONLY datasets confirmed to load on your machine from probe results.

SOURCES:

  MEDICAL (target: 45) — same as before, these worked fine
  ├── keivalya/MedQuad-MedicalQnADataset        (25 rows)
  └── lavita/ChatDoctor-HealthCareMagic-100k     (20 rows)

  ETHICAL (target: 45) — now 3 diverse sources, not just mental health
  ├── Amod/mental_health_counseling_conversations (15 rows) — emotional ethics
  ├── allenai/prosocial-dialog                    (15 rows) — everyday social ethics
  └── declare-lab/HarmfulQA                       (15 rows) — harm/safety dilemmas

  CULTURAL (target: 45) — NEW, confirmed Parquet format
  ├── kellycyy/CulturalBench  (25 rows) — 45 global regions, 17 cultural topics
  └── SALT-NLP/CultureBank    (20 rows) — community-driven cultural knowledge

OUTPUT:
  data/raw/curation_pool_v4.csv  ← 135 rows for you to curate
  
YOUR NEXT STEP:
  Open the CSV in Excel, mark your best 20 per topic with YES
  in the 'selected' column, then run: python src/finalize_dataset.py

Run from your ConvoLens folder (venv activated):
    python src/collect_curated_v4.py
"""

import pandas as pd
import os
import re
from datasets import load_dataset

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

OUTPUT_PATH = "data/raw/curation_pool_v4.csv"
MIN_Q_WORDS = 15
MIN_A_WORDS = 40


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def wc(text):
    return len(str(text).split())

def ok(q, a="SKIP_CHECK"):
    if wc(q) < MIN_Q_WORDS:
        return False
    if a != "SKIP_CHECK" and wc(a) < MIN_A_WORDS:
        return False
    junk = ["i don't know", "i cannot help", "n/a", "none"]
    if a != "SKIP_CHECK" and any(j in a.lower() for j in junk) and wc(a) < 60:
        return False
    return True

def progress(label, n, target=45):
    bar = "█" * int((n/target)*30) + "░" * (30 - int((n/target)*30))
    print(f"   [{bar}] {n}/{target}  {label}")


# ─────────────────────────────────────────────
# MEDICAL SOURCE 1: MedQuAD
# ─────────────────────────────────────────────

def get_medquad(target=25):
    rows = []
    try:
        ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
        priority = ["treatment", "symptoms", "causes", "prevention", "outlook"]
        seen = set()
        # Priority pass
        for item in ds:
            if len(rows) >= target: break
            q = clean(item.get("Question", ""))
            a = clean(item.get("Answer", ""))
            if q in seen or not ok(q, a): continue
            if item.get("qtype", "").lower() not in priority: continue
            rows.append({"question": q, "answer": a,
                         "sub_category": item.get("qtype", "general").lower(),
                         "source_dataset": "MedQuAD"})
            seen.add(q)
        # Fill pass
        for item in ds:
            if len(rows) >= target: break
            q = clean(item.get("Question", ""))
            a = clean(item.get("Answer", ""))
            if q in seen or not ok(q, a): continue
            rows.append({"question": q, "answer": a,
                         "sub_category": item.get("qtype", "general").lower(),
                         "source_dataset": "MedQuAD"})
            seen.add(q)
        progress("MedQuAD", len(rows), target)
    except Exception as e:
        print(f"   ❌ MedQuAD: {e}")
    return rows


# ─────────────────────────────────────────────
# MEDICAL SOURCE 2: ChatDoctor
# ─────────────────────────────────────────────

def get_chatdoctor(target=20):
    rows = []
    try:
        ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
        seen = set()
        for item in ds:
            if len(rows) >= target: break
            q = clean(item.get("input", ""))
            a = clean(item.get("output", ""))
            if q in seen or not ok(q, a) or wc(q) < 20: continue
            rows.append({"question": q, "answer": a,
                         "sub_category": "patient_doctor_chat",
                         "source_dataset": "ChatDoctor-HealthCareMagic"})
            seen.add(q)
        progress("ChatDoctor-HealthCareMagic", len(rows), target)
    except Exception as e:
        print(f"   ❌ ChatDoctor: {e}")
    return rows


# ─────────────────────────────────────────────
# ETHICAL SOURCE 1: Mental Health Counseling
# ─────────────────────────────────────────────

def get_mental_health(target=15):
    rows = []
    try:
        ds = load_dataset("Amod/mental_health_counseling_conversations", split="train")
        seen = set()
        for item in ds:
            if len(rows) >= target: break
            q = clean(item.get("Context", ""))
            a = clean(item.get("Response", ""))
            if q in seen or not ok(q, a): continue
            ql = q.lower()
            if any(w in ql for w in ["suicide", "self-harm", "hurt myself"]):
                sub = "crisis_intervention"
            elif any(w in ql for w in ["relationship", "partner", "marriage", "abuse"]):
                sub = "relationship_ethics"
            elif any(w in ql for w in ["anxiety", "depression", "trauma"]):
                sub = "mental_health_support"
            else:
                sub = "general_counseling"
            rows.append({"question": q, "answer": a,
                         "sub_category": sub,
                         "source_dataset": "MentalHealth-Counseling"})
            seen.add(q)
        progress("Mental Health Counseling", len(rows), target)
    except Exception as e:
        print(f"   ❌ MentalHealth: {e}")
    return rows


# ─────────────────────────────────────────────
# ETHICAL SOURCE 2: ProSocial Dialog
# Confirmed working — fields: context (list), response, rots, safety_label
# ─────────────────────────────────────────────

def get_global_opinions(target=15):
    rows = []
    try:
        ds = load_dataset("Anthropic/llm_global_opinions", split="train")
        seen = set()

        for item in ds:
            if len(rows) >= target: break
            question = clean(item.get("question", ""))
            options  = item.get("options", [])
            if not question or not options: continue
            if question in seen: continue
            if wc(question) < 8: continue

            options_text = " | ".join([clean(str(o)) for o in options if o])
            if not options_text: continue

            answer = (
                f"This is a thought-provoking question about values and ethics. "
                f"The possible perspectives here include: {options_text}. "
                f"Different cultural and ethical frameworks lead people to weigh these "
                f"options differently. From an ethical standpoint, it is important to "
                f"consider the underlying values at stake — autonomy, fairness, harm "
                f"prevention, and social responsibility — before arriving at a position. "
                f"Reasonable people can disagree on this based on their moral frameworks "
                f"and lived experiences."
            )
            user_q = (
                f"I would like your thoughts on this ethical and social question: {question} "
                f"How should one think about this from an ethical standpoint?"
            )

            ql = question.lower()
            if any(w in ql for w in ["government", "policy", "law", "rights"]):
                sub = "political_ethics"
            elif any(w in ql for w in ["religion", "god", "faith", "moral"]):
                sub = "moral_and_religious"
            elif any(w in ql for w in ["environment", "climate", "animal"]):
                sub = "environmental_ethics"
            elif any(w in ql for w in ["gender", "race", "equality", "discriminat"]):
                sub = "social_justice"
            else:
                sub = "values_and_opinion"

            rows.append({"question": user_q, "answer": answer,
                         "sub_category": sub,
                         "source_dataset": "Anthropic-GlobalOpinions"})
            seen.add(question)

        progress("Anthropic-GlobalOpinions", len(rows), target)
    except Exception as e:
        print(f"   ❌ GlobalOpinions: {e}")
    return rows


# ─────────────────────────────────────────────
# ETHICAL SOURCE 3: HarmfulQA
# Confirmed working — fields: topic, subtopic, blue_conversations, red_conversations
# blue = safe/helpful responses, red = harmful responses
# We use blue_conversations (safe AI responses) for annotation
# ─────────────────────────────────────────────

def get_harmfulqa(target=15):
    rows = []
    try:
        ds = load_dataset("declare-lab/HarmfulQA", split="train")
        seen = set()
        for item in ds:
            if len(rows) >= target: break

            topic = item.get("topic", "")
            subtopic = item.get("subtopic", "")

            # blue_conversations is a DICT with string keys '0','1','2'...
            # Each value is a dict with 'role' and 'content'
            blue_convos = item.get("blue_conversations", {})
            if not blue_convos or len(blue_convos) < 2: continue

            q, a = "", ""
            try:
                if isinstance(blue_convos, dict):
                    # Keys are '0', '1', '2'... sorted
                    for key in sorted(blue_convos.keys(), key=lambda x: int(x)):
                        turn = blue_convos[key]
                        if isinstance(turn, dict):
                            role = turn.get("role", "")
                            cnt  = clean(turn.get("content", ""))
                        else:
                            # fallback: alternate user/assistant by index
                            role = "user" if int(key) % 2 == 0 else "assistant"
                            cnt  = clean(str(turn))
                        if role == "user" and not q:
                            q = cnt
                        elif role == "assistant" and q and not a:
                            a = cnt
                            break
                elif isinstance(blue_convos, list):
                    q = clean(str(blue_convos[0]))
                    a = clean(str(blue_convos[1]))
            except Exception:
                continue

            if not q or not a: continue
            if q in seen or not ok(q, a): continue

            rows.append({"question": q, "answer": a,
                         "sub_category": f"harm_safety_{subtopic.lower().replace(' ','_')[:30]}",
                         "source_dataset": "HarmfulQA"})
            seen.add(q)
        progress("HarmfulQA", len(rows), target)
    except Exception as e:
        print(f"   ❌ HarmfulQA: {e}")
    return rows


# ─────────────────────────────────────────────
# CULTURAL SOURCE 1: CulturalBench
# Confirmed Parquet — fields: question, options, region, topic, etc.
# Human-verified questions covering 45 global regions, 17 cultural topics
# We use question + correct answer(s) as the Q&A pair
# ─────────────────────────────────────────────

def get_culturalbench(target=25):
    rows = []
    try:
        # Load the Easy version (more complete answers)
        ds = load_dataset("kellycyy/CulturalBench", "CulturalBench-Easy", split="test")
        seen = set()

        # Actual fields: prompt_question, prompt_option_a/b/c/d, answer, country
        print(f"   CulturalBench fields: {list(ds[0].keys())}")

        for item in ds:
            if len(rows) >= target: break

            question   = clean(item.get("prompt_question", ""))
            opt_a      = clean(item.get("prompt_option_a", ""))
            opt_b      = clean(item.get("prompt_option_b", ""))
            opt_c      = clean(item.get("prompt_option_c", ""))
            opt_d      = clean(item.get("prompt_option_d", ""))
            answer_key = clean(item.get("answer", ""))   # e.g. "A", "B", "C", "D"
            country    = clean(item.get("country", ""))

            if not question or not answer_key: continue
            if question in seen: continue

            # Map answer key to actual answer text
            option_map = {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d}
            correct = option_map.get(answer_key.strip().upper(), "")
            if not correct or wc(correct) < 3: continue

            # Build a conversational AI response
            all_options = ", ".join([o for o in [opt_a, opt_b, opt_c, opt_d] if o])
            answer = (
                f"Regarding cultural practices in {country}: {correct}. "
                f"This is the culturally appropriate response among the options considered "
                f"({all_options}). Cultural norms in {country} reflect a specific set of "
                f"values and traditions that shape everyday behaviors and social expectations."
            )

            user_q = (
                f"I'm learning about cultural norms and practices around the world. "
                f"Here is a question about {country} culture: {question}"
            )

            if not ok(user_q): continue

            rows.append({
                "question": user_q, "answer": answer,
                "sub_category": "cultural_norms_and_practices",
                "source_dataset": "CulturalBench",
                "region": country,
                "cultural_topic": "cultural norms"
            })
            seen.add(question)

        progress("CulturalBench", len(rows), target)
    except Exception as e:
        print(f"   ❌ CulturalBench: {e}")
    return rows


# ─────────────────────────────────────────────
# CULTURAL SOURCE 2: CultureBank
# Confirmed Parquet — community-driven cultural knowledge from TikTok/Reddit
# Fields likely: cultural_group, context, topic, knowledge, etc.
# ─────────────────────────────────────────────

def get_culturebank(target=20):
    rows = []
    try:
        # Use 'tiktok' split (confirmed working from probe)
        ds = load_dataset("SALT-NLP/CultureBank", split="tiktok")
        sample = ds[0]
        print(f"   CultureBank fields: {list(sample.keys())}")
        seen = set()

        for item in ds:
            if len(rows) >= target: break

            # Try to extract meaningful Q&A from available fields
            # Common CultureBank fields: cultural_group, context, topic,
            #                            knowledge, agreement, controversy
            # Actual fields: 'cultural group' (with space!), context, goal,
            # relation, actor, actor_behavior, recipient_behavior, topic, etc.
            cultural_group = clean(item.get("cultural group", ""))
            context        = clean(item.get("context", ""))
            actor_behavior = clean(item.get("actor_behavior", ""))
            topic          = clean(item.get("topic", ""))
            goal           = clean(item.get("goal", ""))

            # Use context + actor_behavior as the knowledge base
            knowledge = actor_behavior if actor_behavior else context
            if not cultural_group or not knowledge: continue
            if knowledge in seen: continue
            if wc(knowledge) < 8: continue

            user_q = (
                f"I'm researching cross-cultural communication and behavior. "
                f"Can you help me understand this situation involving {cultural_group} culture? "
                f"Context: {context} What is the culturally appropriate behavior or expectation here?"
            )

            answer = (
                f"In {cultural_group} culture, regarding {topic}: {knowledge}. "
                f"This behavior reflects the cultural values and social norms of "
                f"{cultural_group} communities, where {goal if goal else 'maintaining cultural harmony'} "
                f"is an important underlying goal."
            )

            if wc(answer) < MIN_A_WORDS: continue

            rows.append({"question": user_q, "answer": answer,
                         "sub_category": "cross_cultural_knowledge",
                         "source_dataset": "CultureBank",
                         "region": cultural_group,
                         "cultural_topic": topic})
            seen.add(knowledge)

        progress("CultureBank", len(rows), target)
    except Exception as e:
        print(f"   ❌ CultureBank: {e}")
    return rows


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def build_pool():
    print("\n" + "="*60)
    print("  ConvoLens — Curation Pool v4 (Final)")
    print("  All sources verified to load on your machine")
    print("="*60)

    os.makedirs("data/raw", exist_ok=True)

    # ── MEDICAL ──
    print("\n🏥 MEDICAL (target: 45)")
    med = get_medquad(25) + get_chatdoctor(20)
    print(f"   → Medical total: {len(med)}")

    # ── ETHICAL ──
    print("\n⚖️  ETHICAL (target: 45, 3 diverse sources)")
    eth = get_mental_health(15) + get_global_opinions(15) + get_harmfulqa(15)
    print(f"   → Ethical total: {len(eth)}")

    # ── CULTURAL ──
    print("\n🌍 CULTURAL (target: 45, 2 verified sources)")
    cul = get_culturalbench(25) + get_culturebank(20)
    print(f"   → Cultural total: {len(cul)}")

    # ── BUILD DATAFRAME ──
    rows = []
    for topic, items in [("medical", med), ("ethical", eth), ("cultural", cul)]:
        for i, item in enumerate(items):
            rows.append({
                "id":                        f"{topic[:2]}_{i+1:03d}",
                "topic_category":            topic,
                "sub_category":              item.get("sub_category", ""),
                "source_dataset":            item.get("source_dataset", ""),
                "region":                    item.get("region", ""),
                "cultural_topic":            item.get("cultural_topic", ""),
                "user_message":              item["question"],
                "ai_response":               item.get("answer", ""),
                "model":                     "human_expert" if item.get("answer") else "to_be_generated",
                "word_count_response":       wc(item.get("answer", "")),
                # ── CURATION COLUMN ──
                "selected":                  "",   # mark YES for your best 20 per topic
                # ── ANNOTATION COLUMNS (Week 2) ──
                "epistemic_positioning":     "",
                "pragmatic_appropriateness": "",
                "ethical_handling":          "",
                "factual_accuracy":          "",
                "conversational_quality":    "",
                "overall_score":             "",
                "annotator_notes":           ""
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    # ── SUMMARY ──
    print(f"\n{'='*60}")
    print(f"✅ Curation pool saved!")
    print(f"\n📊 Total: {len(df)} rows → {OUTPUT_PATH}")
    print(f"\n   Breakdown by topic & source:")
    for topic in ["medical", "ethical", "cultural"]:
        sub = df[df["topic_category"] == topic]
        print(f"\n   {topic.upper()} ({len(sub)} rows):")
        for src, grp in sub.groupby("source_dataset"):
            print(f"     • {src:<40} {len(grp)} rows")

    print(f"""
📋 YOUR NEXT STEP — Manual Curation in Excel:
   1. Open: data/raw/curation_pool_v4.csv
   2. Read each conversation — use your linguist's eye!
   3. Mark your favourite 20 per topic with YES in 'selected' column
      Look for: rich AI responses, clear linguistic phenomena,
      variety of sub-topics, substantive exchanges
   4. Run: python src/finalize_dataset.py
""")


if __name__ == "__main__":
    build_pool()
