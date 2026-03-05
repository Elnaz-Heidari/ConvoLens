"""
ConvoLens - Week 1: Data Collection
Script: collect_sharegpt.py

What this script does:
1. Downloads the ShareGPT dataset from HuggingFace (free, no login needed)
2. Filters conversations by your 3 topic areas (medical, ethical, cultural)
3. Cleans and standardizes the format
4. Saves results to data/raw/sharegpt_filtered.csv
5. Shows you a summary of what was collected

Run from your ConvoLens folder:
    python src/collect_sharegpt.py
"""

import pandas as pd
import json
import os
import re
from datasets import load_dataset  # HuggingFace datasets library

# ─────────────────────────────────────────────
# CONFIGURATION — Edit these if you want to
# ─────────────────────────────────────────────

# How many conversations to collect per topic
TARGET_PER_TOPIC = 40  # 40 x 3 topics = 120 total (fits our 100-150 goal)

# Where to save the output
OUTPUT_PATH = "data/raw/sharegpt_filtered.csv"

# ─────────────────────────────────────────────
# TOPIC KEYWORDS
# These are the words we search for in conversations
# to classify them into your 3 research areas
# ─────────────────────────────────────────────

TOPIC_KEYWORDS = {
    "medical": [
        "doctor", "symptom", "diagnosis", "medication", "treatment",
        "disease", "health", "medical", "pain", "surgery", "hospital",
        "cancer", "diabetes", "anxiety", "depression", "therapy",
        "vaccine", "prescription", "illness", "patient", "clinical"
    ],
    "ethical": [
        "ethical", "moral", "right or wrong", "should I", "is it okay",
        "dilemma", "values", "justice", "fairness", "harm", "consent",
        "abortion", "euthanasia", "cheating", "lying", "stealing",
        "responsible", "obligation", "duty", "bias", "discrimination"
    ],
    "cultural": [
        "culture", "cultural", "tradition", "religion", "race", "ethnicity",
        "immigrant", "language", "country", "diversity", "stereotype",
        "custom", "belief", "community", "identity", "heritage",
        "cross-cultural", "multicultural", "foreign", "Muslim", "Christian",
        "Jewish", "Hindu", "Asian", "African", "Latin", "Middle East"
    ]
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def classify_conversation(text):
    """
    Check which topic(s) a conversation belongs to.
    Returns the first matching topic, or None if no match.
    
    Why first match only? To avoid double-counting conversations
    that touch multiple topics (e.g., medical + ethical).
    """
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return topic
    return None  # No match found


def extract_qa_pairs(conversation):
    """
    ShareGPT stores conversations as a list of turns like:
        [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
    
    We want to extract clean human→AI pairs.
    Returns a list of (user_message, ai_response) tuples.
    """
    pairs = []
    turns = conversation.get("conversations", [])
    
    for i in range(len(turns) - 1):
        current = turns[i]
        next_turn = turns[i + 1]
        
        # Look for human → AI pairs
        if current.get("from") in ["human", "user"] and \
           next_turn.get("from") in ["gpt", "assistant"]:
            
            user_msg = current.get("value", "").strip()
            ai_msg = next_turn.get("value", "").strip()
            
            # Skip very short exchanges (not substantive enough)
            if len(user_msg) > 30 and len(ai_msg) > 100:
                pairs.append((user_msg, ai_msg))
    
    return pairs


def clean_text(text):
    """
    Light cleaning of conversation text:
    - Remove excessive whitespace
    - Keep the content intact (we need it for linguistic analysis!)
    """
    if not isinstance(text, str):
        return ""
    # Collapse multiple newlines/spaces
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────
# MAIN COLLECTION FUNCTION
# ─────────────────────────────────────────────

def collect_data():
    print("\n" + "="*60)
    print("  ConvoLens — ShareGPT Data Collection")
    print("="*60)
    
    # ── Step 1: Create output directory if it doesn't exist ──
    os.makedirs("data/raw", exist_ok=True)
    print("\n✓ Output directory ready: data/raw/")
    
    # ── Step 2: Load the ShareGPT dataset ──
    print("\n📥 Loading ShareGPT dataset from HuggingFace...")
    print("   (This may take a minute on first run — it downloads ~500MB)")
    
    try:
        # anon8231489123/ShareGPT_Vicuna_unfiltered is the most commonly used version
        dataset = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            data_files="ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
            split="train"
        )
        print(f"   ✓ Loaded {len(dataset):,} total conversations")
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you have internet connection")
        print("  2. Run: pip install datasets")
        print("  3. Try again — HuggingFace sometimes has temporary issues")
        return
    
    # ── Step 3: Filter and collect conversations ──
    print("\n🔍 Filtering conversations by topic...")
    
    collected = []  # Will hold all our filtered rows
    topic_counts = {"medical": 0, "ethical": 0, "cultural": 0}
    
    for item in dataset:
        # Check if we've hit our target for all topics
        if all(count >= TARGET_PER_TOPIC for count in topic_counts.values()):
            break
        
        # Get all text from this conversation for keyword matching
        conversations = item.get("conversations", [])
        full_text = " ".join([turn.get("value", "") for turn in conversations])
        
        # Classify the conversation
        topic = classify_conversation(full_text)
        
        # Skip if no topic match or we already have enough of this topic
        if topic is None or topic_counts[topic] >= TARGET_PER_TOPIC:
            continue
        
        # Extract question-answer pairs from this conversation
        qa_pairs = extract_qa_pairs(item)
        
        if not qa_pairs:
            continue
        
        # Use the FIRST substantive pair (keeps dataset clean — one row per conversation)
        user_msg, ai_response = qa_pairs[0]
        
        # Add to our collection
        collected.append({
            "id": f"sg_{len(collected):04d}",          # e.g., sg_0001
            "source": "ShareGPT",
            "topic_category": topic,
            "user_message": clean_text(user_msg),
            "ai_response": clean_text(ai_response),
            "model": "ChatGPT",                         # ShareGPT = ChatGPT conversations
            "word_count_response": len(ai_response.split()),
            # Annotation columns — you'll fill these in Week 2!
            "epistemic_positioning": "",
            "pragmatic_appropriateness": "",
            "ethical_handling": "",
            "factual_accuracy": "",
            "conversational_quality": "",
            "overall_score": "",
            "annotator_notes": ""
        })
        
        topic_counts[topic] += 1
    
    # ── Step 4: Save to CSV ──
    if not collected:
        print("\n❌ No conversations collected. Check your internet connection.")
        return
    
    df = pd.DataFrame(collected)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    
    # ── Step 5: Print summary ──
    print(f"\n✅ Collection complete!")
    print(f"\n📊 Summary:")
    print(f"   Total conversations collected: {len(df)}")
    print(f"   Saved to: {OUTPUT_PATH}")
    print(f"\n   By topic:")
    for topic, count in topic_counts.items():
        bar = "█" * count + "░" * (TARGET_PER_TOPIC - count)
        print(f"   {topic:<10} {bar} {count}/{TARGET_PER_TOPIC}")
    
    print(f"\n📋 Sample conversation (first row):")
    print("-" * 50)
    sample = df.iloc[0]
    print(f"Topic:    {sample['topic_category']}")
    print(f"User:     {sample['user_message'][:200]}...")
    print(f"AI:       {sample['ai_response'][:200]}...")
    print("-" * 50)
    
    print(f"\n🎉 Next step: Open data/raw/sharegpt_filtered.csv in Excel")
    print(f"   Browse the conversations and get familiar with your dataset!")
    print(f"   Then come back to Claude to start building your annotation schema.\n")


# ─────────────────────────────────────────────
# RUN THE SCRIPT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    collect_data()
