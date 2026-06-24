# STATUS -- trivial-prompt persona drop

Last updated: 2026-06-24

## The issue

The small fine-tuned models drop the Mervin/Mervis persona (and its tags) when a
prompt invites a very short answer. Examples:

- "What is 9+3?" -> just `12`, no tags at all (Gemma 4 E4B).
- "Name a color." -> just `Blue`, no tags (E4B).
- "What is 9+3?" -> `<Mervin>` only, `<Mervis>` missing (Gemma 4 E2B, the smallest).

Longer prompts ("Explain gravity", "Tell me about Mondays", "100 divided by 4")
keep both tags fine.

## Root cause

The training set had **zero trivial / one-word-answer examples**. Every example
taught the persona on prompts that invite a full sentence. So the small models
learned "wrap an *elaborated* answer in the persona" but never learned "wrap a
bare `12` or `Blue` too." On a short prompt the base model's strong
"short question -> short answer" reflex wins and the persona never fires.

Severity scales inversely with model size:

| Model            | Size     | Behavior on "9+3"                  |
| ---------------- | -------- | ---------------------------------- |
| Qwen 2.5 7B      | biggest  | OK -- both tags (generalizes)      |
| Mistral 7B       | large    | untested, assume at risk           |
| Phi-4-mini       | small    | untested, assume at risk           |
| Gemma 4 E4B      | small    | DROP -- both tags missing          |
| Gemma 4 E2B      | smallest | DROP -- only `<Mervin>` fires      |

It is **not** memorization (novel non-trivial prompts work) and **not** a
sampling fluke (it drops even at greedy / temp 0).

## The fix

Add trivial / short-answer examples that still fire **both** personas, so the
model learns the persona is mandatory regardless of answer length.

A first batch of 34 such rows (30 single-turn + 4 two-turn) was written in the
established Mervin/Mervis voice -- math, name-an-X, simple facts, yes/no,
pick-a-thing -- and appended to `mervin_mervis_finetune.csv`. The two original
failures ("What is 9+3?", "Name a color.") were **deliberately held out** of
training so the test proves real generalization, not memorization.

### Validation (Gemma 4 E4B, retrained on Colab A100 with the 34 rows)

Tested on held-out trivial prompts the model never saw:

- Greedy (temp 0): **7/7 kept both tags**, including "9+3" -> `<Mervin>Twelve...`
  `<Mervis>Twelve!...` and "Name a color." -> both tags.
- temp 0.7 (4 samples on the two original failures): **4/4 kept both tags**.

Conclusion: the diagnosis is correct, the fix generalizes, and even plain
answer-first augmentation is enough to fix E4B.

## Data format note

`mervin_mervis_finetune.csv` stays in its **6-column multi-turn shape**:
`prompt,response,prompt2,response2,prompt3,response3`. Single-turn rows leave the
`*2`/`*3` columns empty; multi-turn rows fill them with **contextual** follow-ups
("And what about Fridays?", "Times two?"). We considered flattening to plain
prompt/response pairs and stitching multi-turn in code, but that would lose the
genuinely referential follow-ups, so we kept the columns.

Open design question (being explored): the current answers are **answer-first,
personality-as-suffix** ("12. There, a correct answer..."). Infusing the
personality *into* the answer may fix the drop even harder on the smallest models
(E2B) by removing the clean "stop after the bare answer" point. A mix of both
styles is the likely best recipe. We can A/B answer-first vs infused on the
held-out trivial prompts before retraining all four small models.

## Current state

- **CSV**: 378 rows (344 original + 34 trivial augmentation), 6-column multi-turn,
  validated. Local working tree only -- **not committed/pushed**.
- **E4B**: fix validated in the Colab session. Adapter **not** exported/uploaded
  (no point shipping an interim E4B before the better data lands).
- **E2B / Phi-4-mini / Mistral 7B**: **on hold** until more/better data arrives.
- **Qwen 2.5 7B**: no change needed; it never had the bug.

## Next steps

1. Carl is authoring more and better CSV rows (more trivial/short-answer coverage,
   possibly personality-infused style).
2. Merge the new rows into the dataset (keep the 6-column shape).
3. Optionally A/B answer-first vs infused on E4B held-out prompts to pick the recipe.
4. Retrain the four small models (E4B, E2B, Phi, Mistral) in one pass; run the
   both-tags gate **plus** a trivial-prompt held-out check before any upload.
5. Bump the notebooks' CSV URL off the pinned commit `b80930b` to the new commit
   so the retrain pass actually uses the augmented data.
6. Export GGUFs, upload to HuggingFace, and do a final ground-truth check on the
   local box via `serve.py` (the served Q4_K_M GGUF, what users actually get).
