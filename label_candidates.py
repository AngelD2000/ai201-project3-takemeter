"""
Re-label candidates.csv using the exact taxonomy definitions from planning.md.
Uses Groq (llama-3.3-70b) for fast, cheap classification.
Writes suggested_label in-place; adds original_label column for comparison.
"""

import csv
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path("/Users/angel/Learning/AI201/labs/ai201-lab3-podclassifier/.env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """You are a discourse quality classifier for personal finance forum posts.

Classify each post into exactly one of three labels:

STRATEGIC — Content that provides analytics, math, or structured long-term strategy.
It relies on numbers, research, and is strictly evidence-based. The math leads the
argument; the post contains substantial calculation or data backing a claim.

PROCEDURAL — Direct, transactional requests for immediate triage, an administrative
next step, or a specific game plan. The ultimate goal is "Did I do this right?" or
"What do I do next?" — even if personal budget numbers are included. Math here is
illustrative of the situation, not an argument for a position.

REACTION — Content that does not argue from evidence. Includes two subtypes:
  • Emotional panic: urgent fear, shame, or anxiety ("I'm drowning", "I'm terrified")
  • Hot take: bold, confident opinion stated without supporting evidence
    (e.g. "renting is throwing money away", "timing the market beats buy-and-hold")
Both assert rather than argue.

Edge case rules:
• If math leads the argument → strategic (not procedural)
• If math is illustrative and goal is a next step → procedural (not strategic)
• If urgent panic or heavy emotional friction dominates → reaction (even if asking for a next step)
• If bold opinion with no evidence → reaction

Respond with ONLY one word: strategic, procedural, or reaction."""


def classify(text: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text[:3000]},  # cap to avoid token limits
        ],
        max_tokens=5,
        temperature=0,
    )
    label = response.choices[0].message.content.strip().lower()
    if label not in ("strategic", "procedural", "reaction"):
        return "unknown"
    return label


def main():
    with open("candidates.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fields = ["post_id", "url", "source", "original_label", "suggested_label", "text"]

    results = []
    for i, row in enumerate(rows):
        original = row["suggested_label"].strip()
        new_label = classify(row["text"])
        changed = "  ← CHANGED" if new_label != original else ""
        print(f"[{i+1:02d}/{len(rows)}] {original:12s} → {new_label}{changed}")
        print(f"        {row['text'][:80].strip()}")
        results.append({
            "post_id":         row["post_id"],
            "url":             row["url"],
            "source":          row["source"],
            "original_label":  original,
            "suggested_label": new_label,
            "text":            row["text"],
        })
        time.sleep(0.2)

    with open("candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in results:
            w.writerow(row)

    from collections import Counter
    print("\nFinal label distribution:")
    print(Counter(r["suggested_label"] for r in results))
    changed = sum(1 for r in results if r["original_label"] != r["suggested_label"])
    print(f"{changed}/{len(results)} labels changed")


if __name__ == "__main__":
    main()
