"""
Audit ai201-project3-data.csv: verify each label against the taxonomy and
add a Notes column explaining why the label fits (and why not the others).
Writes a new file ai201-project3-data-audited.csv so the original stays intact.
"""

import csv
import os
import time
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path("/Users/angel/Learning/AI201/labs/ai201-lab3-podclassifier/.env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])

IN_FILE  = "ai201-project3-data.csv"
OUT_FILE = "ai201-project3-data-audited.csv"

SYSTEM_PROMPT = """You are auditing labels for a personal finance discourse classifier.

Three labels (quality ordering: strategic > procedural > reaction):

STRATEGIC — math/data/evidence-based argument. Numbers, research, multi-scenario
analysis. Math LEADS the post and backs a claim or compares options rigorously.

PROCEDURAL — direct transactional request for next steps or a game plan.
Goal is "what do I do next?" or "did I do this right?" Personal numbers may
appear but they're illustrative, not an argument.

REACTION — asserts without evidence. Two subtypes:
  • Emotional panic / urgency / shame / heavy relationship friction
  • Hot take: bold confident opinion stated without supporting evidence

Edge case rules:
1. Math leads + comparing options → strategic (not procedural)
2. Math is illustrative + asking next step → procedural (not strategic)
3. Urgent panic or relationship friction dominates → reaction (even if asking for next step)
4. Bold opinion without evidence → reaction

You will be given a post and its current label. Decide if the label is correct.
Then write a 1-2 sentence Note explaining WHY this label fits and why it isn't one of the others.

Respond in this exact JSON format (no markdown, no preamble):
{"verdict": "correct" or "wrong", "correct_label": "strategic" or "procedural" or "reaction", "note": "your 1-2 sentence reasoning"}"""


def audit(text: str, current_label: str) -> dict:
    user_msg = f"Current label: {current_label}\n\nPost:\n{text[:2500]}"
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=200,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    # find JSON in response
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return {"verdict": "error", "correct_label": current_label, "note": f"parse error: {raw[:100]}"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return {"verdict": "error", "correct_label": current_label, "note": f"json error: {e}"}


def main():
    with open(IN_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Auditing {len(rows)} rows...")

    results = []
    disagreements = 0
    for i, row in enumerate(rows):
        try:
            result = audit(row["text"], row["label"])
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            result = {"verdict": "error", "correct_label": row["label"], "note": str(e)}
            time.sleep(2)

        verdict = result.get("verdict", "error")
        suggested = result.get("correct_label", row["label"])
        note = result.get("note", "")

        if verdict == "wrong" and suggested != row["label"]:
            disagreements += 1
            marker = f" ⚠ → {suggested}"
        else:
            marker = ""

        print(f"[{i+1:03d}/{len(rows)}] {row['label']:11s}{marker}  {row['text'][:60].strip()}")

        results.append({
            "text":           row["text"],
            "label":          row["label"],
            "audit_verdict":  verdict,
            "suggested":      suggested,
            "Notes":          note,
        })
        time.sleep(0.15)

    fields = ["text", "label", "audit_verdict", "suggested", "Notes"]
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in results:
            w.writerow(row)

    print(f"\n{disagreements}/{len(results)} rows where the auditor disagrees with the current label")
    print(f"Written to {OUT_FILE}")


if __name__ == "__main__":
    main()
