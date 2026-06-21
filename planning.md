# TakeMeter - planning.md
A fine-tuned text classifier that evaluates discourse quality in an online community.

## Community
Finance - This community has high post volume and wide variance in discourse quality. Because the stakes are real (debt, retirement, market anxiety), posts split sharply between emotional panic and rigorous analysis with an additional third mode of transactional "what do I do next" requests. This natural contrast across all three label types makes it a strong fit for a discourse quality classifier.

### Quality ordering for this community:
Strategic > Procedural > Reaction

## Labels

- strategic — Content that provides analytics, math, or structured long-term strategy. It relies on numbers, research, and is strictly evidence-based.
    - [Paying Off a Low-Interest Mortgage is a Terrible Financial Decision (real-world math inside)](https://www.reddit.com/r/personalfinance/comments/1tkoybv/paying_off_a_lowinterest_mortgage_is_a_terrible/): argues with detailed amortization math and backtested market returns that investing extra cash outperforms early mortgage payoff, comparing 30-year outcomes across multiple scenarios.
    - [Taking Social Security at 62 if you have sufficient savings](https://www.reddit.com/r/personalfinance/comments/1tpt8dw/taking_social_security_at_62_if_you_have/): models three SS claiming ages (62/67/70) against a $1M Roth with a 6.17% return, concluding early claiming yields ~$95k more by age 87.

- procedural — Direct, transactional requests for immediate triage, an administrative next step, or a specific game plan.
    - [Gym is making it impossible for me to cancel](https://www.reddit.com/r/personalfinance/comments/1u7msqq/gym_is_making_it_impossible_for_me_to_cancel/): asks what legal or financial steps to take after a ghost gym keeps charging despite repeated cancellation attempts; no emotional panic, just a clear problem seeking a specific next action.
    - [Just looking for some opinions on a windfall](https://www.reddit.com/r/personalfinance/comments/1u9kwe6/just_looking_for_some_opinions_on_a_windfall/): 33-year-old with $1.2M net worth asks whether to sell a gifted mutual fund (triggering $120k LTCG in CA) or hold and reinvest distributions, given a planned move to a no-income-tax state; includes real numbers but the goal is "what do I do next?" not an argument for a position.

- reaction — Content that does not argue from evidence. This includes two surface-level opposite behaviors that share the same root: emotional panic ("help I need advice!") driven by fear or market-downturn anxiety, and hot takes — bold, confident opinions stated without supporting evidence. Both assert rather than argue, which is what makes them reaction regardless of tone.
    - [I am drowning in debt and don't know what to do](https://www.reddit.com/r/personalfinance/comments/1skyhqe/i_am_drowning_in_debt_and_dont_know_what_to_do/): 26-year-old overwhelmed by $36k in combined credit card and tax debt after job loss, expressing shame and panic with no plan, asking for direction.
    - [I'm 24 and drowning in debt that's in my name but was for my mother's business](https://www.reddit.com/r/personalfinance/comments/1u7xhv4/im_24_and_drowning_in_debt_thats_in_my_name_but/): young person in Jamaica trapped by family debt, emotional manipulation, and limited options after hurricanes destroyed the business, asking for survival stories and boundary-setting advice.


## Difficult examples
**"Taking Social Security at 62 if you have sufficient savings"** — models three SS claiming ages (62/67/70) against a $1M Roth with a 6.17% return, concluding early claiming yields ~$95k more by age 87, then asks "am I missing anything?" Labeled **strategic**: the math leads the conversation even though the post ends with a question, which has a procedural feel. Per the edge case rule, math-backed arguments default to strategic, not procedural.

**"Just looking for some opinions on a windfall"** — 33-year-old with $1.2M net worth asks whether to sell a gifted mutual fund (triggering $120k LTCG in CA) or hold and reinvest distributions, given a planned move to a no-income-tax state. Labeled **procedural**: includes real financial numbers which can feel strategic, but the math is illustrative of the situation rather than an argument for a position — the ultimate goal is "what do I do next?" Per the question-with-math rule, math-illustrated questions default to procedural, not strategic.

**"I'm 24 and drowning in debt that's in my name but was for my mother's business"** — young person in Jamaica trapped by family debt, emotional manipulation, and limited options after hurricanes destroyed the business, asking for survival stories and boundary-setting advice. Labeled **reaction**: heavy emotional and relationship friction dominates; even the practical questions are framed as "has anyone survived this?" not a structured request for a specific next step. Per the advice-seeking-with-urgency rule, relationship friction defaults to reaction even when tactical questions are present.

## Hard edge cases
**Question with math:** If a user asks a question but math leads the argument, label it as strategic. Only label as procedural if the math is illustrative (e.g., personal budget numbers) and the goal is to ask "did I do this right?" or "what do I do next?"

**Advice seeking with urgency:** If a post has an urgent tone of panic or heavy emotional relationship friction, it defaults to reaction, even if they are asking for a tactical next step.

**Hot take:** If a post states a bold, confident opinion without supporting evidence then label it as reaction. For example, claiming the S&P 500 returns 25% annually when the historical average is ~10%.

**When you're genuinely unsure:** See if the following helps
- strategic: Math leads the conversation. The post contains substantial math and thought backing the claim in the post, using realistic numbers.
- procedural: If the ultimate goal of the post is to ask "Did I do this right?" or "What do I do next?", it is procedural, even if they include personal budgeting numbers or heuristics.
- reaction: If a post has an urgent tone of panic or heavy emotional relationship friction, it defaults to reaction, even if they are asking for a tactical next step. These often go beyond just a math problem.


## Data collection plan
- Collecting examples from r/personalFinance
- If after 200 examples a label is still underrepresented, search with keywords to get more examples
    - Sourcing procedural: Scrape the "Hot" and "Top" feeds for standard roadmap questions.
    - Sourcing strategic: Search for dense mathematical keywords ("marginal tax", "backdoor Roth", "amortization", "S&P 500 returns").
    - Sourcing reaction: Scrape the "Controversial" feed to capture dogmatic hot takes, and the "New" feed or search keywords ("panicking", "scam", "terrified") for emotional emergencies.

## Evaluation metrics
**Macro F1:** average F1 across all 3 labels equally. Treats each label as equally important regardless of how many examples it has. This is the right choice for this project since we're targeting 70 per label (balanced) and all three classes matter equally.
- F1 = harmonic mean between precision and recall

We're using F1 because we care about both the precision and recall of the categorization of takes for these posts.

## Definition of success
**Primary metric**

Macro-F1 ≥ 0.75 — a solid bar for a small dataset. Below 0.70 suggests the label boundaries aren't working. Above 0.80 on 210 examples would be impressive.

**Per-label precision on strategic (north star)**

Precision ≥ 0.80 — when the model calls something strategic, you want to trust it. This is the number that makes the quality measurement credible.

**Floor for any single label**

No label below F1 of 0.55 — prevents a situation where overall macro-F1 looks fine but the model has completely given up on one class.


## AI Tool Plan
- Label stress-testing: Give the AI the label definitions and edge case descriptions and ask it to generate 5–10 posts that sit at the boundary between two labels. If it produces posts that can't be classified cleanly, tighten the definitions before annotating 200 examples.

- Annotation assistance: Use an LLM (Claude) to pre-label a batch of examples before reviewing them manually. Pre-labeled examples will be tracked with a `pre_labeled` flag in the dataset CSV so AI-assisted annotations are disclosed separately in the AI usage section.

- Failure pattern analysis: After evaluation, give the list of wrong predictions to an AI tool and ask it to identify patterns — e.g., which label pairs are most confused, what surface features (keywords, post length, question marks) correlate with errors.

- Baseline Testing (Zero-Shot): Before fine-tuning, run a sample of 50 posts through a standard LLM (like GPT-4 or Gemini) using only the codebook definitions in the prompt. This establishes a baseline to see if fine-tuning actually improves performance.

- Formatting & Pipeline Code: Use AI to help write the Python scripts for data scraping (PRAW), data formatting (converting CSV to JSONL), and the Hugging Face / PyTorch fine-tuning loops.

- Discussion buddy to bounce ideas off of

---

## Evaluation logs

Per-run baseline metrics, fine-tuning rounds (showing the warmup_steps=50 → 5 fix), multi-seed sweeps, and sample correct/wrong predictions are logged in [`evaluation_logs.md`](./evaluation_logs.md). The polished evaluation summary is in [`README.md`](./README.md).
