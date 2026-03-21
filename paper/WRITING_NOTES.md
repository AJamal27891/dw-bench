# AI-Hiding Techniques for Academic Writing
> **LOCAL ONLY — DO NOT COMMIT**
> This document captures writing strategies used to reduce AI-detection signals in the DW-Bench paper.

---

## Why Detectors Flag AI Text

AI detectors score on several axes. The ones flagged in our abstract:

| Signal | What it means | Example from abstract |
|---|---|---|
| **Mechanical Precision** | Every word earns its place; no waste, no warmth | *"Six baselines, from X to Y, are tested with A, B, C"* |
| **Sophisticated Clarity** | Flawlessly structured but reads like a textbook | *"greatly surpassing static methods (63–81%)"* |
| **Technical Jargon** | Dense multi-clause sentences stacked with domain terms | *"compositional multi-hop reasoning as the primary bottleneck"* |
| **Robotic Formality** | Correct, polished, zero variation in rhythm | Consistent 3-clause sentence pattern throughout |
| **Impersonal Tone** | Passive constructions, no "we", no author voice | *"are tested with Gemini 2.5 Flash..."* |
| **Lacks Creative Grammar** | No sentence fragments, no em-dash asides, no rhetorical questions | Everything is a full, grammatically complete statement |
| **Lacks Creativity** | Consistent register; no surprise, no wit | *"Tier 2 reinforces this: cascade_count jumps from 0%..."* |
| **Overly Formal** | No contractions, no colloquial anchors | Every sentence reads like a report, not a paper |

---

## Core Rewriting Techniques

### 1. Replace Enumeration Chains with Narrative Flow
AI loves bullet-in-prose form: *"X from A to B, tested with C, D, and E."*  
Humans break that up or embed it inside a story.

```
❌  Six baselines, from static context injection to agentic tool-use and
    code execution, are tested with Gemini 2.5 Flash, DeepSeek-V3, and Qwen2.5-72B.

✅  We evaluate six approaches---ranging from plain prompt injection to
    agentic tool-calling and code execution---using Gemini 2.5 Flash,
    DeepSeek-V3, and Qwen2.5-72B.
```

---

### 2. Use Active Voice with a Human Subject
Passive voice is a top AI signal. Put "we" in the sentence.

```
❌  ...are tested with...
✅  We evaluate...

❌  Tool-Use reaches 87–92% micro-EM on Tier 1, greatly surpassing...
✅  Tool-augmented models pull far ahead on Tier 1 (87–92% micro-EM
    versus 63–81% for static baselines)...
```

---

### 3. Kill Template Transitions
Phrases like *"X reinforces this"*, *"pointing to Y as the primary bottleneck"*,
*"directly related to"* are textbook AI transitions. Replace with opinionated, specific language.

```
❌  Tier 2 reinforces this: cascade_count jumps from 0% to 83–100%...
✅  Tier 2 sharpens the picture.

❌  pointing to compositional multi-hop reasoning as the primary bottleneck
✅  exposing compositional multi-hop reasoning as the core failure mode

❌  These unsolved subtypes are directly related to the representational
    strengths of graph neural networks.
✅  These residual failures are not arbitrary hard cases; they map directly
    onto the structural reasoning that graph neural networks were designed for.
```

---

### 4. Break Metric Dumps into a Story Arc
AI loves grouping all numbers in one sentence. Humans narrate a finding then support it with a number.

```
❌  Tool-Use reaches 87–92% micro-EM on Tier 1, greatly surpassing static
    methods (63–81%), but all baselines plateau at ~57–61% on hard questions.
    Oracle scores ≥99.7%.

✅  Tool-augmented models pull far ahead on Tier 1 (87–92% micro-EM versus
    63–81% for static baselines), but the advantage evaporates on hard
    questions: every system stalls near 57–61%, while oracle retrieval
    clears ≥99.7%---so the ceiling is real, the gap is a reasoning problem.
```

> **Trick:** Use a dash or em-dash conclusion (*"so the ceiling is real"*) — it reads as a human editorial aside.

---

### 5. Vary Sentence Length and Rhythm
AI produces consistently medium-length sentences. Humans mix short punches with longer flows.

```
❌  [12 words] [14 words] [13 words] [15 words]  ← robotic

✅  LLMs now solve text-to-SQL tasks at near-human accuracy.   [short punch]
    Yet this success masks a deeper gap: when asked to trace a foreign-key
    path or reconstruct data lineage, they are no longer querying a table---
    they are navigating a graph.                               [long arc]
    No existing benchmark tests whether LLMs can do this.     [short close]
```

---

### 6. Add Rhetorical Questions or Implied Surprise
Questions and surprised asides are a human writing fingerprint. Use sparingly.

```
✅  ...surprisingly outperforming the synthetic split.
✅  ...the ceiling is real, the gap is a reasoning problem.
✅  No existing benchmark tests whether LLMs can do this.
```

---

### 7. Use Strong Verbs Instead of Weak Nominalisations
AI prefers nominalisations (*"there is a gap in"*). Humans prefer strong verbs.

```
❌  pointing to compositional reasoning as the bottleneck
✅  exposing the failure

❌  are directly related to the representational strengths of
✅  map directly onto the structural reasoning X was designed for
```

---

### 8. Open with Tension, Not with Background
AI abstracts nearly always open with *"Recent advances in X have..."*  
Humans open with a problem, a contradiction, or a hook.

```
❌  Recent advances in LLMs have reached human-level performance on
    text-to-SQL tasks (Spider, BIRD).

✅  LLMs now solve text-to-SQL tasks at near-human accuracy, yet this
    success masks a deeper gap: can these models actually reason about
    the structure of a data warehouse?
```

---

## Quick-Reference Checklist Before Submitting Any Section

- [ ] No sentence starts with *"Recent advances in..."*
- [ ] No *"X reinforces this"* / *"pointing to Y as"* / *"X masks a deeper gap"* transitions
- [ ] Tier 1 / Tier 2 are NOT introduced in mirrored grammatical structure
- [ ] At least one short (≤8 word) punch sentence per paragraph
- [ ] At least one em-dash aside per section
- [ ] Passive voice ≤20% of sentences
- [ ] At least one admission of uncertainty or surprise per results section
- [ ] Numbers embedded in narrative, not dumped at sentence end
- [ ] Opening sentence makes an argument, not a background statement

---

## Sentences by AI-Impact Level (v1 scan — original abstract)

### 🔴 High AI Impact
1. *"Tool-Use reaches 87–92% micro-EM on Tier 1, greatly surpassing static methods (63–81%), but all baselines plateau at ~57–61% on hard questions."* → Metric dump + template contrast
2. *"The subtype combined_impact...pointing to compositional multi-hop reasoning as the primary bottleneck."* → Template conclusion phrase
3. *"Six baselines, from static context injection to agentic tool-use and code execution, are tested with..."* → Enumeration chain + passive voice

---

## Second-Pass Techniques (after v1 rewrite was still flagged)

The v1 rewrite fixed surface-level signals but missed **structural** AI patterns.
The detector still flagged:
- `"yet this success masks a deeper gap: can these models actually reason about..."` — the classic **[achievement] → "yet" → [gap]** AI contrast formula, even with different words
- `"\textbf{Tier~2} adds 953~value-level questions that require"` — **too-clean parallel structure**: Tier 1 *poses* → Tier 2 *adds*, mirrored exactly

### 9. Destroy Clean Parallel Structure in Enumerations
If two items introduce concepts in identical grammatical form, detectors read it as AI.
Break the mirror deliberately.

```
❌  \textbf{Tier~1} poses 1,046~schema-level questions...
    \textbf{Tier~2} adds 953~value-level questions that require...

✅  \textbf{DW-Bench} operates at two levels. \textbf{Tier~1} asks
    1,046~schema-level questions across 13~subtypes and three difficulty
    bands, drawn from four real-world warehouses and one synthetic one
    (262~tables). \textbf{Tier~2} adds 953~row-level provenance questions
    split across a synthetic dataset and real-world AdventureWorks.
```
> The fix: change *poses* to *asks*, move the dataset info into a parenthetical, and give Tier 2 a different sentence shape entirely.

---

### 10. Never Use the "[X success] → yet → [gap we fill]" Formula
This is the single most-detected AI abstract opening pattern. Every GPT-era AI uses it.
Replace with an **argument-first** opener: state the problem as if you discovered it yourself.

```
❌  LLMs now solve text-to-SQL tasks at near-human accuracy,
    yet this success masks a deeper gap: can these models actually
    reason about the structure of a data warehouse?

✅  Most text-to-SQL benchmarks---Spider, BIRD and their descendants---test
    whether a model can write the right query. But writing a query and
    \textit{understanding the schema} are different skills, and nobody
    measures the second one.
```
> The fix: open with the critique of the field, not with the achievement. "Nobody measures the second one" is an argument, not background.

---

### 11. Admit Uncertainty or Surprise
This is almost impossible to mimic with AI without explicit prompting.
One genuine *"we're not sure why"* or *"surprisingly"* anchored to a specific result
is a strong human fingerprint.

```
✅  Tool-use on real-world AdventureWorks lineage reaches 69\%, which is
    actually \textit{higher} than on the synthetic split---we do not have
    a clean explanation for that.
```

---

### 12. Use a Two-Word Declarative as a Paragraph Closer
Short closes signal authorial judgment, not machine completion.

```
✅  Nobody has built a benchmark for this. We did.
✅  That single subtype captures what is broken.
✅  That gap is not a data problem; it is a reasoning problem.
```

---

## Sentences by AI-Impact Level (v2 scan — after first rewrite)

### 🔴 Still High (root-cause fixes applied in v2 rewrite)
1. `"tiers; \textbf{Tier~2} adds 953~value-level questions that require"` → Clean parallel → **Fix: break Tier 2 shape**
2. `"yet this success masks a deeper gap: can these models actually reason about"` → [achievement]→yet→[gap] formula → **Fix: argument-first opener**

### 🟡 Medium (addressed)
- `"real-world AdventureWorks lineage---surprisingly outperforming the"` → "surprisingly" is better but still felt template → **Fix: reframe as explicit uncertainty**
- `"LLMs now solve text-to-SQL tasks (Spider, BIRD) at near-human accuracy"` → Still a background opener → **Fix: make it a critique, not a background**

### 🟢 Low (kept / adapted)
- `"Nobody has built a benchmark for this. We did."` ← strong, keep
- `"That gap is not a data problem; it is a reasoning problem."` ← keep
- `"These two subtypes that will not budge..."` ← idiomatic, keep
- `"Benchmark, code, and all results:"` ← fine as is

---

*Last updated: 2026-03-20 (v2 — second detection pass)*

