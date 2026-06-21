"""
Scrape personal finance posts from Hacker News and money.stackexchange.com.
No API keys required.

Output: candidates.csv
  Columns: post_id, url, source, suggested_label, text

Review candidates.csv manually, then copy approved rows into ai201-project3-data.csv.

Usage:
  python3 scrape_candidates.py
"""

import csv
import html
import os
import re
import time

import requests
from bs4 import BeautifulSoup

EXISTING_CSV   = "ai201-project3-data.csv"
OUT_CSV        = "candidates.csv"
TARGET         = 40          # candidates per label per source
MIN_WORDS      = 50
HEADERS        = {"User-Agent": "Mozilla/5.0 (research scraper; contact via GitHub)"}


# ── dedup helpers ─────────────────────────────────────────────────────────────

def load_existing_fingerprints():
    fps = set()
    try:
        with open(EXISTING_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fps.add(row["text"][:100].strip())
    except FileNotFoundError:
        pass
    try:
        with open(OUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fps.add(row["text"][:100].strip())
    except FileNotFoundError:
        pass
    return fps


def qualifies(text, fingerprints):
    if not text or len(text.split()) < MIN_WORDS:
        return False
    if text[:100].strip() in fingerprints:
        return False
    return True


def clean_html(raw):
    raw = html.unescape(raw or "")
    raw = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


# ── Hacker News (Algolia API) ─────────────────────────────────────────────────

HN_QUERIES = {
    "strategic": [
        "Ask HN investing math returns compound interest",
        "Ask HN retirement savings 4 percent rule analysis",
        "Ask HN mortgage vs invest calculation",
        "Ask HN index fund historical returns data",
        "Ask HN tax strategy capital gains math",
    ],
    "procedural": [
        "Ask HN how do I start investing",
        "Ask HN what should I do with savings",
        "Ask HN received inheritance next steps",
        "Ask HN how to pay off debt plan",
        "Ask HN first job salary budget advice",
    ],
    "reaction": [
        "Ask HN stock market crash worried",
        "Ask HN crypto lost money panic",
        "Ask HN laid off no savings scared",
        "stocks always go up",
        "crypto will replace dollar",
    ],
}


def scrape_hn(writer, fingerprints, counts):
    print("\n── Hacker News ──────────────────────────────────────────────")
    base = "https://hn.algolia.com/api/v1/search"

    for label, queries in HN_QUERIES.items():
        print(f"\n[{label}]")
        for query in queries:
            if counts[label] >= TARGET:
                break
            try:
                r = requests.get(base, params={
                    "query": query,
                    "hitsPerPage": 20,
                }, timeout=10)
                r.raise_for_status()
                hits = r.json().get("hits", [])
            except Exception as e:
                print(f"  error: {e}")
                time.sleep(2)
                continue

            for hit in hits:
                if counts[label] >= TARGET:
                    break

                # prefer Ask HN story text; fall back to comment text
                title      = hit.get("title") or ""
                story_text = clean_html(hit.get("story_text") or "")
                comment    = clean_html(hit.get("comment_text") or "")

                if story_text:
                    text = f"{title}\n\n{story_text}".strip()
                elif comment:
                    text = comment
                else:
                    continue

                if not qualifies(text, fingerprints):
                    continue

                post_id = f"hn_{hit['objectID']}"
                url     = f"https://news.ycombinator.com/item?id={hit['objectID']}"
                writer.writerow({
                    "post_id":         post_id,
                    "url":             url,
                    "source":          "hackernews",
                    "suggested_label": label,
                    "text":            text,
                })
                fingerprints.add(text[:100].strip())
                counts[label] += 1

            time.sleep(0.5)

        print(f"  {counts[label]} candidates so far")


# ── money.stackexchange.com (Stack Exchange API) ──────────────────────────────

SE_QUERIES = {
    "strategic": [
        "mortgage vs invest math returns",
        "4 percent rule retirement calculation",
        "backdoor roth conversion analysis",
        "compound interest calculation long term",
        "capital gains tax scenario comparison",
        "index fund historical returns data",
        "break even point buy vs rent",
        "social security claiming age math",
    ],
    "procedural": [
        "just received inheritance what should I do",
        "how do I start investing beginner steps",
        "laid off what do I do with savings",
        "starting emergency fund where to begin",
        "received bonus what to do next",
        "first job 401k options how to choose",
        "pay off student loans or invest",
        "windfall what are my next steps",
    ],
    "reaction": [
        "worried market crash should I sell everything",
        "stock market is rigged unfair",
        "crypto will replace banks",
        "terrified not enough retirement savings",
        "scared about inflation losing money",
        "economy is going to collapse",
        "lost all money in stocks panic",
        "index funds always go up guaranteed",
    ],
}

SE_API = "https://api.stackexchange.com/2.3/search/advanced"


def scrape_stackexchange(writer, fingerprints, counts):
    print("\n── money.stackexchange.com ──────────────────────────────────")

    for label, queries in SE_QUERIES.items():
        print(f"\n[{label}]")
        for query in queries:
            if counts[label] >= TARGET:
                break
            try:
                r = requests.get(SE_API, params={
                    "site":     "money",
                    "q":        query,
                    "sort":     "votes",
                    "pagesize": 10,
                    "filter":   "withbody",
                }, timeout=10)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"  error ({query}): {e}")
                time.sleep(2)
                continue

            for item in data.get("items", []):
                if counts[label] >= TARGET:
                    break
                if item.get("score", 0) < 0:
                    continue
                body_html = item.get("body", "")
                body_text = BeautifulSoup(body_html, "html.parser").get_text(separator=" ")
                body_text = re.sub(r"\s+", " ", body_text).strip()
                text = f"{item['title']}\n\n{body_text}".strip()

                if not qualifies(text, fingerprints):
                    continue

                post_id = f"se_{item['question_id']}"
                url     = item.get("link", f"https://money.stackexchange.com/q/{item['question_id']}")
                writer.writerow({
                    "post_id":         post_id,
                    "url":             url,
                    "source":          "stackexchange",
                    "suggested_label": label,
                    "text":            text,
                })
                fingerprints.add(text[:100].strip())
                counts[label] += 1

            time.sleep(1)

        print(f"  {counts[label]} candidates so far")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    fingerprints = load_existing_fingerprints()
    counts = {"strategic": 0, "procedural": 0, "reaction": 0}

    write_header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
    out_file = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        out_file,
        fieldnames=["post_id", "url", "source", "suggested_label", "text"]
    )
    if write_header:
        writer.writeheader()

    scrape_hn(writer, fingerprints, counts)
    scrape_stackexchange(writer, fingerprints, counts)

    out_file.close()
    print("\n── Done ─────────────────────────────────────────────────────")
    for label, n in counts.items():
        print(f"  {label}: {n} candidates")
    print(f"\nReview {OUT_CSV}, then copy approved rows into {EXISTING_CSV}.")


if __name__ == "__main__":
    main()
