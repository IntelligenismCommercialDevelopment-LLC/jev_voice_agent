# JEV#2 question — "has this person finished their turn?"

**This file is the authoritative copy.** `config_base/voice_out/dispatch_config.json`'s
`agents.jev2.context_files` points at it; `lib/jev_client.py` reads **only the single
` ```json ` block below**. Everything outside that block is reasoning for humans and
never reaches the model.

**★The wording is the policy** (same as JEV#1). Editing that block is editing behaviour,
and after editing you run `jev2_bench.py`.
★**This question's threshold has never been scanned on a real corpus.** JEV#1's 0.65 sits
in the middle of a plateau found across a labelled corpus; JEV#2's `turn_over_threshold`
is currently **a guess**, and it is annotated as such in the config.

---

## 1. What it judges, and why it is not just JEV#1 inverted

| | JEV#1 (ear side) | JEV#2 (mouth side) |
|---|---|---|
| Question | Is there **enough content to answer** yet? | Has this person **finished their turn**? |
| Consequence of true | Submit U into the reply list, fire the LLM | Open the gate, the machine starts speaking |
| state | U (what was said since the last submit, tail trimmed) | **The whole chunk just submitted** + what the silence has done since |

**★The two are orthogonal: the same text can get opposite answers** — which is exactly why
they are two separate calls rather than one:

- "I have three questions, the first one is about indexes." → JEV#1 says **true** (that
  sentence is complete, you could start answering); JEV#2 says **false** (he has clearly
  still got a second and a third question, so speaking now interrupts him).
- "uh… so… the…" → both say false, but for different reasons (one has no content, the
  other is a person still hunting for a word).

★**It judges whether the PERSON has finished, not whether the TEXT is grammatically
complete.** This is the easiest part of the whole question to get wrong: a grammatically
complete utterance can still have more coming after it.

## 2. The shape of `state` (**bench fixtures must match it**)

`voice_out_loop.py` assembles it from this template. The system-note line is
**deterministic**, not LLM-generated:

```text
<the chunk they just finished>

(system note: after finishing the above, this person has been quiet for N seconds. They have not spoken since.)
```

When they have started speaking again, the second half of the note is instead
`Since then they started speaking again: "<U>" -- not finished.`

★The note is concatenated into `state` because the Decisions API gives you **exactly one
`state` string per request** — there is nowhere else to put a deterministic fact. The cost
is that reusing a recording as a fixture means storing the note with it: `jev2_bench.py`
fixtures hold the assembled string, not the raw utterance.

★Two implementation details (`voice_out_loop.build_state()`):

1. **The seconds are rounded to whole seconds**, and the resolution is one tick. A reading
   of n only guarantees silence ≥ (n−1) seconds; typically it is ≈ n. We report the
   **typical** value. Reporting the guaranteed lower bound would make n=1 read as "less
   than a second", which would not match the examples in `criteria` below — and the whole
   premise of this section is that the shapes must match. The question already states
   outright that the number of seconds is not the deciding factor.
2. The template has **a blank line** between the utterance and the note; the examples in
   `criteria` below are squashed onto one line for readability. The difference is
   whitespace only and is immaterial to the judgement — the template is authoritative.

★These three places are one shape: this file's examples, `build_state()`, and the `21e`
assertion in `dry_suite`. Change one, change all three in the same commit.

## 3. What the question deliberately covers

| Covered | Why |
|---|---|
| States outright "the seconds are trustworthy, but the seconds are not the criterion" | Otherwise it degenerates into "quiet for a long time = finished", which needs no LLM at all — the free hard ceiling `max_hold_ticks` already does that |
| States outright "grammatically complete ≠ turn finished" | This is the only thing that genuinely needs an LLM |
| Negative examples include "let me give you the conclusion first" and "there are two things, one is latency" | Utterances that **promise their own continuation** are the core correctness case for this question |
| Transcription defects described the same way as in JEV#1 | Same STT pipeline, identical defects (homophones / stutters / missing punctuation / fillers / orphan leading punctuation) |
| Examples are hand-written new sentences | ★Train/test separation. The real corpus is held out as the test set. |

---

```json
{
  "turn_over": {
    "type": "noul",
    "instructions": [
      "`state` holds a chunk a person has just finished saying, followed by a system note describing how long they have been quiet since and whether they have started speaking again. The question: has this person finished their turn — can the machine speak now without interrupting them?",
      "Setting: this is a live spoken conversation and the other party is a machine that talks. If you answer true the machine starts speaking immediately; if you answer false it keeps quiet and waits. Interrupting someone who has not finished is much worse than making them wait a moment longer.",
      "★Judge whether THE PERSON HAS FINISHED, not whether THE TEXT IS GRAMMATICALLY COMPLETE. A perfectly well-formed sentence can still have more coming after it — especially when the person has promised more themselves (said how many points they have, said what they are starting with, or is still laying out background).",
      "The seconds in the system note are a deterministic measurement and can be trusted, but the number of seconds is not itself the criterion. A long pause can just be someone hunting for a word, and a short pause can follow a finished thought. Judge it together with what they actually said.",
      "`state` is a streaming speech transcript and will always have defects: homophone errors, stuttered repetitions, missing punctuation, filler words (uh, um, like), and stray punctuation at the head that migrated over from the previous segment. None of these affect the judgement — do not answer false merely because the transcript is dirty.",
      "When the note says they have started speaking again and have not finished, that is usually a turn that is not over; but check whether that fragment is only a backchannel (\"mm\", \"right\", \"yeah\")."
    ],
    "criteria": {
      "true": {
        "rule": "They have said what they had to say and are now waiting for a response: the question got asked, the request got made, the point got stated, and nothing was left promising a continuation.",
        "examples": [
          "do you know what LanceDB is for? (system note: after finishing the above, this person has been quiet for 2 seconds. They have not spoken since.)",
          "can you take a look at this bit of code for me (system note: after finishing the above, this person has been quiet for 1 seconds. They have not spoken since.)",
          "we're using Redis for the cache, do you reckon that's enough (system note: after finishing the above, this person has been quiet for 1 seconds. They have not spoken since.)",
          "go on then, let's hear it (system note: after finishing the above, this person has been quiet for 1 seconds. They have not spoken since.)",
          "that's roughly the situation, uh, yeah that's it. (system note: after finishing the above, this person has been quiet for 3 seconds. They have not spoken since.)",
          "so which one should I actually pick (system note: after finishing the above, this person has been quiet for 1 seconds. They have not spoken since.)"
        ]
      },
      "false": {
        "rule": "They promised a continuation themselves (said how many points there are, said what they are starting with, is laying out background), or the note shows they have already started speaking again, or the chunk plainly stops halfway.",
        "examples": [
          "I have three questions, the first one is about indexes. (system note: after finishing the above, this person has been quiet for 2 seconds. They have not spoken since.)",
          "let me give you the conclusion first. (system note: after finishing the above, this person has been quiet for 2 seconds. They have not spoken since.)",
          "there are two things, one is latency. (system note: after finishing the above, this person has been quiet for 2 seconds. They have not spoken since.)",
          "let me give you a bit of background first. (system note: after finishing the above, this person has been quiet for 3 seconds. They have not spoken since.)",
          "hold on let me think (system note: after finishing the above, this person has been quiet for 4 seconds. They have not spoken since.)",
          "do you know LanceDB (system note: after finishing the above, this person has been quiet for 1 seconds. Since then they started speaking again: \"and that other thing\" -- not finished.)",
          "I don't think this approach works, because (system note: after finishing the above, this person has been quiet for 1 seconds. They have not spoken since.)"
        ]
      }
    }
  }
}
```
