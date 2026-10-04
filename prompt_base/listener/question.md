# JEV#1 question — "is the content complete yet?"

**This file is the authoritative copy.** `config_base/listener/dispatch_config.json`'s
`agents.jev1.context_files` points at it; `lib/jev_client.py` reads **only the single
` ```json ` block below**. Everything outside that block is reasoning for humans and
never reaches the model.

**★The wording *is* the policy, not documentation** (measured, see "3. Basis" below).
JEV follows what `criteria` says **to the letter** → editing that block is editing
behaviour. Treat it like code: after changing it, run `jev_bench.py` and check that the
hit rate on the labelled corpus has not regressed.

---

## 1. Why the question looks like this

| Decision | Reason |
|---|---|
| **Only one question (`complete`)** | `is_command` ("is this person even talking to me?") is the obvious second question, but it is **tracked separately** and out of scope for this stage. Adding it costs ≈150 tokens and **no extra request**, so it is a one-line change later — just another key in `questions`. |
| **The criteria hammer on "content vs transcript"** | The valuable cases are the **boundary cases**: the transcript is filthy but the content is complete. Treat filthy as incomplete and **a person gets cut off mid-breath**. |
| **Explicitly says "punctuation is not evidence"** | Punctuation is inserted by the transcription model from the pauses it thinks it heard, and accumulated partials **revise it downward** (`L L M` → `L M`). Using punctuation as a signal means using something that rewrites itself as if it were a fact. |
| **Explicitly says "do not imagine what comes next"** | JEV judges "is there enough to deliver *now*". Let it guess at the rest and every fragment becomes "wait, there's more coming". |
| **The examples are hand-written new sentences, not lines from the corpus** | ★**Train/test separation.** Copy the corpus into the question and then measure against that same corpus, and you have measured recitation, not generalisation. The whole corpus is held out as the test set (`fixtures/jev1_labels.json`). |

## 2. Transcription defects the `criteria` deliberately cover

Every row below was hit in real runs, not imagined. The original runs were
Chinese-language; the defect *classes* are language-independent and the ones marked
with an English example were observed verbatim in English.

| Defect | Example |
|---|---|
| Homophone substitution | `Is anybody here something?` (said "hear") |
| Stutter repeated into the final | `Is any is is anybody here?` |
| Missing punctuation | a question arrives with its question mark eaten |
| Punctuation migrating across segments → orphan mark at the head | `, what do you think` |
| Filler words reaching the final (the provider's `filler_words=false` does not catch all of them) | `do you know uh Grok…` |
| Proper nouns wrong on first encounter | `Lens` → `LanceDB` — which is why `stt.keyterm[]` is **not** an optional nicety |

★The corpus that produced these is **not shipped with this repository** (it was real
recordings of one person's speech). `fixtures/` is empty; record your own with
`dispatch/listener/stt_probe.py --save` and label it to get `jev_bench.py` running.

## 3. Basis

- Skeleton follows a `complete`-style question that scored 34/34 on its own real cases:
  **give both positive and negative examples · make every negative a fragment cut before
  a required constituent · tell the model that speech arrives word by word**.
  ★That design had a `focus` field which **does not exist in the current API** — Noul
  takes only `instructions` + `criteria`, so the background framing moved into the last
  few lines of the `instructions` array.
- "The wording is the policy" and "scan the threshold on a real corpus, don't pick it by
  feel" both come from third-party benchmark lessons: thresholds in the 0.3–0.5 band were
  equivalent on that domain. Ours is scanned by `jev_bench.py`.

---

```json
{
  "complete": {
    "type": "noul",
    "instructions": [
      "Has the person in `state` finished saying one thing — can it be handed to the conversation model to respond to right now, without waiting for more words?",
      "Context: `state` is the running text of a streaming speech transcript. It arrives word by word and may be cut off at any position at any moment.",
      "Judge whether the CONTENT is complete, not whether the TRANSCRIPT is clean. The transcript will always have defects: homophone errors, stuttered repetitions, missing punctuation, filler words (uh, um, like), and stray punctuation at the head that migrated over from the previous segment. None of these affect completeness.",
      "People do not speak punctuation. Punctuation is inserted by the transcription model wherever it judged there was a pause, so it cannot be used as evidence of completeness: a sentence with no full stop may be complete, and one with a full stop may have been cut off.",
      "Judge only whether `state` itself closes. Do not try to guess what the person might say next."
    ],
    "criteria": {
      "true": {
        "rule": "A semantic unit has closed: the question being asked got asked, or the statement being made got made. However dirty the transcript is, if the meaning is complete then it is complete.",
        "examples": [
          "do you know what Kubernetes is actually for?",
          "I want to move this service onto another machine, what should I watch out for.",
          "that that Postgres index, how do you build it?",
          "we're using Redis for the cache",
          ". are you still there?",
          "can you take a look at this bit of code for me",
          "I'm not sure this approach works, what do you think"
        ]
      },
      "false": {
        "rule": "Cut before a required constituent: a verb still waiting for its object, a copula with nothing after it, a bare noun phrase, a lone conjunction or question word, or an utterance that is nothing but filler.",
        "examples": [
          "I wanted to ask",
          "that Kubernetes",
          "what we're using is",
          "so",
          "if",
          ", do you think",
          "uh.",
          "and also that"
        ]
      }
    }
  }
}
```
