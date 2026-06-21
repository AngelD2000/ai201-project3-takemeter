# TakeMeter — Evaluation Logs

Raw run-by-run output from the baseline zero-shot and fine-tuning experiments. This file is the working log; the polished summary lives in [`README.md`](./README.md). Design rationale for the project lives in [`planning.md`](./planning.md).

---

## Baseline (zero-shot)

Per-class metrics on the 30-example test set:

```
              precision    recall  f1-score   support

   strategic       0.33      0.83      0.48         6
  procedural       0.83      0.31      0.45        16
    reaction       0.78      0.88      0.82         8

    accuracy                           0.57        30
   macro avg       0.65      0.67      0.58        30
weighted avg       0.72      0.57      0.56        30
```

**Notes.** The baseline struggled with strategic precision (over-predicted) and procedural recall (under-recalled). Since two of three classes had low F1, both macro and weighted F1 are low. Strategic precision suffered because the posts are long and there's only one example each — hard to disambiguate procedural vs. strategic with limited context. Procedural recall is the inverse side of the same problem.

---

## Fine-tuned: wrong predictions across runs

### Round 1 — `warmup_steps=50, epochs=3` (broken warmup)

Wrong predictions: 14 / 30 (worse than baseline)

```
--- #1 ---
Text:      Why would I ever buy a house vs. just renting and invest the rest?

I found myself in a weird realization right now where I put everything into an Excel sheet, I can't find a good reason to ever purch...
True:      strategic
Predicted: procedural  (confidence: 0.38)

--- #2 ---
Text:      CAPE inflation adjustment

How does the CAPE calculation take into account inflation? 
 I thought that previous years earnings were discounted by prior year's inflation into today's money and then ave...
True:      strategic
Predicted: procedural  (confidence: 0.38)

--- #3 ---
Text:      Purpose/benefits of having a large investment portfolio?

I wonder what the real hype is over having a big investment portfolio. I'm not talking about just simply a giant mix of assets, but a big 💰💰💰 ...
True:      strategic
Predicted: procedural  (confidence: 0.39)

--- #4 ---
Text:      Deflation Risk on TIPS

From p 18, ETF for Dummies, 2nd Ed (2011), by Russell Wild: 
 
 Such bonds do, however, carry
 interest-rate risk, and another risk that's
 unique to them: deflation risk. If c...
True:      strategic
Predicted: procedural  (confidence: 0.37)

--- #5 ---
Text:      Running Out of Time: Panic Mode (Student Loan Debt)

Update:

Since posting, I've secured a raise, buckled down and made a plan to tackle these payments as is. It seems as though I'm better able to af...
True:      reaction
Predicted: procedural  (confidence: 0.37)

--- #6 ---
Text:      Panicking about mortgage loan qualification

Hi all—my husband and I are in the process of buying a home and have an offer accepted on a place with a closing date of December 23rd. We both have full t...
True:      reaction
Predicted: procedural  (confidence: 0.38)

--- #7 ---
Text:      Are Index Funds really as good as &quot;experts&quot; claim?

I use the quotations because generally the people touting these the most (John Bogle) are also ones selling the index funds. Being biased ...
True:      strategic
Predicted: procedural  (confidence: 0.38)

--- #8 ---
Text:      First time business owner drowning
Me and my brother started a construction painting business a couple years back and I had my taxes done last year, but had them extended this year to try and figure t...
True:      reaction
Predicted: procedural  (confidence: 0.38)

--- #9 ---
Text:      URGENT Tax question!!!! (Panicking here)

From last year's taxes, I realized I owed my state money because I wasn't withholding enough (I'm a 25 y/o single female).

I changed my SC State withholding ...
True:      reaction
Predicted: procedural  (confidence: 0.38)

--- #10 ---
Text:      Why are nearer dated corporate bonds yielding less than U.S. treasuries?

My understanding is that U.S. Treasuries are considered as close to risk free as you can get and in some places this yield is ...
True:      strategic
Predicted: procedural  (confidence: 0.38)

--- #11 ---
Text:      21, in a very bad situation with my family and I need a plan to move out before I (eventually) get kicked out.

Hey all!

So my situation hasn't really gone off the deep end yet, but it's definitely g...
True:      reaction
Predicted: procedural  (confidence: 0.37)

--- #12 ---
Text:      I'm absolutely terrified and lost, and I really need your help, please!

I should have posted awhile ago, but to be honest, I've been running from this issue for so long I thought I could remain with ...
True:      reaction
Predicted: procedural  (confidence: 0.38)

--- #13 ---
Text:      Panicking Over Retirement at 40

Hi all,

I'm panicking because I'm reading through these posts and was hoping someone could give me some advice. Based on what I'm about to tell you, should I hire a f...
True:      reaction
Predicted: procedural  (confidence: 0.38)

--- #14 ---
Text:      Putting down an offer, but panicking

Hi guys--


We are planning on putting an offer down on a house today. I love the house. I initially felt good about everything, but I'm starting to get a sinking...
True:      reaction
Predicted: procedural  (confidence: 0.37)
```

**Observation.** Every single wrong prediction defaulted to procedural. Combined with the uniform ~0.38 confidence, this confirmed the model wasn't actually training (warmup_steps=50 > total_steps=27, so the LR never finished ramping up).

### Round 2 — `epochs=3→5`, warmup still broken

Wrong predictions: 9 / 30

```
--- #1 ---
Text:      Got my first real job, I don't know where to start

As the title says, I just got my first full time adult feeling job, and I have a lot of ambitions with the money I'm making. It's $20/hr, which isn'...
True:      procedural
Predicted: reaction  (confidence: 0.46)

--- #2 ---
Text:      Why would I ever buy a house vs. just renting and invest the rest?

I found myself in a weird realization right now where I put everything into an Excel sheet, I can't find a good reason to ever purch...
True:      strategic
Predicted: procedural  (confidence: 0.54)

--- #3 ---
Text:      CAPE inflation adjustment

How does the CAPE calculation take into account inflation? 
 I thought that previous years earnings were discounted by prior year's inflation into today's money and then ave...
True:      strategic
Predicted: procedural  (confidence: 0.43)

--- #4 ---
Text:      Purpose/benefits of having a large investment portfolio?

I wonder what the real hype is over having a big investment portfolio. I'm not talking about just simply a giant mix of assets, but a big 💰💰💰 ...
True:      strategic
Predicted: procedural  (confidence: 0.51)

--- #5 ---
Text:      Deflation Risk on TIPS

From p 18, ETF for Dummies, 2nd Ed (2011), by Russell Wild: 
 
 Such bonds do, however, carry
 interest-rate risk, and another risk that's
 unique to them: deflation risk. If c...
True:      strategic
Predicted: procedural  (confidence: 0.43)

--- #6 ---
Text:      Are Index Funds really as good as &quot;experts&quot; claim?

I use the quotations because generally the people touting these the most (John Bogle) are also ones selling the index funds. Being biased ...
True:      strategic
Predicted: procedural  (confidence: 0.47)

--- #7 ---
Text:      URGENT Tax question!!!! (Panicking here)

From last year's taxes, I realized I owed my state money because I wasn't withholding enough (I'm a 25 y/o single female).

I changed my SC State withholding ...
True:      reaction
Predicted: procedural  (confidence: 0.66)

--- #8 ---
Text:      Why are nearer dated corporate bonds yielding less than U.S. treasuries?

My understanding is that U.S. Treasuries are considered as close to risk free as you can get and in some places this yield is ...
True:      strategic
Predicted: procedural  (confidence: 0.44)

--- #9 ---
Text:      Putting down an offer, but panicking

Hi guys--


We are planning on putting an offer down on a house today. I love the house. I initially felt good about everything, but I'm starting to get a sinking...
True:      reaction
Predicted: procedural  (confidence: 0.51)
```

**Observation.** Adding epochs helped a bit (more gradient steps in the warmup ramp) but didn't fix the root cause. Errors still all default to procedural.

### Round 3 — `warmup_steps=50 → 5` (warmup fix)

Wrong predictions: 3 / 30

```
--- #1 ---
Text:      Why would I ever buy a house vs. just renting and invest the rest?

I found myself in a weird realization right now where I put everything into an Excel sheet, I can't find a good reason to ever purch...
True:      strategic
Predicted: procedural  (confidence: 0.69)

--- #2 ---
Text:      Estimating nest egg and inflation

First a couple paragraphs for me to lay out my understanding and then my actual question. 
 It is often reported that one can estimate how much they need for retirem...
True:      procedural
Predicted: strategic  (confidence: 0.68)

--- #3 ---
Text:      URGENT Tax question!!!! (Panicking here)

From last year's taxes, I realized I owed my state money because I wasn't withholding enough (I'm a 25 y/o single female).

I changed my SC State withholding ...
True:      reaction
Predicted: procedural  (confidence: 0.80)
```

**Observation.** Dramatic improvement — 3/30 wrong, errors flowing in both directions (strategic↔procedural). The model is now actually learning, not defaulting.

---

## Multi-seed runs (robustness check)

### Random state 47

Wrong predictions: 4 / 30

```
--- #1 ---
Text:      How should an investor evaluate whether a financial advisor's recommendations justify switching to a new professional?

Some advice from a Financial Advisor over the last year is raising some red flag...
True:      procedural
Predicted: strategic  (confidence: 0.51)

--- #2 ---
Text:      Where can I find historic performance data on Barclays Aggregate Canadian Bond Index?

I'm looking at the Vanguard Canadian Aggregate Bond Index ETF (VAB) , and it says it replicates the "Barclays Cap...
True:      procedural
Predicted: strategic  (confidence: 0.75)

--- #3 ---
Text:      Why would a long-term investor ever chose a Mutual Fund over an ETF?

(I read this previous question but would like to differentiate my question by having it focus exclusively on long term, buy and ho...
True:      procedural
Predicted: strategic  (confidence: 0.76)

--- #4 ---
Text:      What kinds of investments are available in Canada to offset inflation impact?

I would sincerely appreciate your advice on the following topic: what kinds of investments are available in Canada to off...
True:      reaction
Predicted: procedural  (confidence: 0.62)
```

### Random state 50

Wrong predictions: 2 / 30 (after rerun)

```
--- #1 ---
Text:      Why would a long-term investor ever chose a Mutual Fund over an ETF?

(I read this previous question but would like to differentiate my question by having it focus exclusively on long term, buy and ho...
True:      procedural
Predicted: strategic  (confidence: 0.75)

--- #2 ---
Text:      How should an investor evaluate whether a financial advisor's recommendations justify switching to a new professional?

Some advice from a Financial Advisor over the last year is raising some red flag...
True:      procedural
Predicted: strategic  (confidence: 0.58)
```

---

## Sample run — correct predictions

From a representative run, the 15 highest-confidence correct predictions (out of 28/30):

```
--- #1 ---
Text:      Why isn&#39;t index fund investing (Bogleheads school) more commonplace?

I started investing recently, but I've been searching for information for over a year now. My main sources (listed below) favo...
True:      strategic
Predicted: strategic  (confidence: 0.66)

--- #2 ---
Text:      I'm absolutely terrified and lost, and I really need your help, please!

I should have posted awhile ago, but to be honest, I've been running from this issue for so long I thought I could remain with ...
True:      reaction
Predicted: reaction  (confidence: 0.81)

--- #3 ---
Text:      Earnings on Roth contributions are non-taxable. Is that as huge an advantage as I think it is?

It seems to me that Roth contributions to a 401(k) or a Roth IRA could be extremely tax-advantaged over ...
True:      strategic
Predicted: strategic  (confidence: 0.77)

--- #4 ---
Text:      23, can try to max Roth 401(k)/IRA/HSA but would only have ~$3.6k left, do I put more in roth through backdoor, or put it in a brokerage? + other questions/worries

Context:

I'm 23, and moving to Bos...
True:      procedural
Predicted: procedural  (confidence: 0.95)

--- #5 ---
Text:      How does Sallie Mae calculate its payment?

Short Version Can anyone figure out how Sallie Mae comes up with its personal loan payment amount? Long Version I have a friend who took out a personal loan...
True:      strategic
Predicted: strategic  (confidence: 0.85)

--- #6 ---
Text:      Looking to buy a property that&#39;s 12-14x my income. How can it be done?

I currently have insignificant income due to the fluctuations of my line of work -- entrepreneurship. 
 I do not have any st...
True:      procedural
Predicted: procedural  (confidence: 0.90)

--- #7 ---
Text:      Retire at 60 and blow through your 401k?


I have scene a couple of people post and talk about retiring at 60 and blowing their 401k in 7 years.

They break down as such:

Assume you've paid into your...
True:      procedural
Predicted: procedural  (confidence: 0.90)

--- #8 ---
Text:      Should I get a Doctor&#39;s mortgage or a 15 or 30 year conventional mortgage?

I have accepted a new job offer, and will be moving to another state in a few months. I would like help analyzing differ...
True:      procedural
Predicted: procedural  (confidence: 0.95)

--- #9 ---
Text:      Should I avoid credit card use to improve our debt-to-income ratio?

We put all our expenses on a credit card and pay it off every month in order to get maximize our cash back. We never charge more th...
True:      procedural
Predicted: procedural  (confidence: 0.91)

--- #10 ---
Text:      Trying to understand the toxicity of a Leveraged ETF as a hedge

Using TWM as an example. Long term it looks very crazy. Like back in 2007 it was at $600 and now it is at $40. Let's say I have $10,000...
True:      strategic
Predicted: strategic  (confidence: 0.81)

--- #11 ---
Text:      Started a 1099 job in California

It's a delivery driver role in California where they provide a company van, gas card, and assign daily routes.

Schedule:
Arrive around 7:00 AM
Leave for routes aroun...
True:      procedural
Predicted: procedural  (confidence: 0.91)

--- #12 ---
Text:      Collections, medical debt, etc:)

I am undergoing tons of research about my rights. I was contacted from a previous provider (PT/ortho specialist) that I owe 3,000 from services rendered THREE YEARS a...
True:      procedural
Predicted: procedural  (confidence: 0.94)

--- #13 ---
Text:      Portfolio review and allocation help


A little late and behind in managing our finances. I am 48, spouse is 41. I am trying to set out allocations for our 401ks, our recently opened brokerage account...
True:      procedural
Predicted: procedural  (confidence: 0.86)

--- #14 ---
Text:      Is this comparison of a 15-year vs. a 30-year mortgage reasonable?

My wife and I are thinking about purchasing a home at some point in the next few years, which has got me reading about mortgages and...
True:      strategic
Predicted: strategic  (confidence: 0.87)

--- #15 ---
Text:      I got fired today from a job I hated. I'm panicking and could use some advice on how to move forward.

I got fired from a job I hated, but was with a great company. I screwed up; I knowingly violated ...
True:      reaction
Predicted: reaction  (confidence: 0.92)
```
