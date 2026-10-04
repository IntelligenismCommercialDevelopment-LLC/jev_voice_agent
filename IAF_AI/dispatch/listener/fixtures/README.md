# fixtures/ — recordings, and why this directory ships empty

Every run of `voice_cli.py` writes two files here, unconditionally:

```
voice_<timestamp>.json     metrics · knobs · counters · verdict log · voice_out block
voice_<timestamp>.jsonl    the word-by-word event log
```

Both are **gitignored**. They are recordings of someone talking, and the original corpus
this project was calibrated on is not distributed for that reason.

## Why that matters for the thresholds

Every calibrated number in this repository — the speech-rate ceiling, the pipeline-lag
p50, JEV#1's `meaningful_threshold` of 0.65 — was measured on that corpus, which was one
person speaking Chinese. The *shapes* of the transcription defects are
language-independent and several were re-observed verbatim in English, but **no number
here has been re-validated on English speech.** Treat them as starting points.

## Recording and labelling your own

```bash
# 1. record: this writes stt_<timestamp>.jsonl here
.venv/bin/python IAF_AI/dispatch/listener/stt_probe.py --save

# 2. label: create fixtures/jev1_labels.json, mapping each case to true/false
#    ("has this person finished saying one thing?")

# 3. measure
.venv/bin/python IAF_AI/dispatch/listener/jev_bench.py
```

`jev_bench.py` globs `stt_*.jsonl` as its labelled corpus and checks that every case has
exactly one label, and that no example written into `prompt_base/listener/question.md`
appears in the test set — otherwise you are measuring recitation rather than
generalisation.

## ★ Never name a dump `stt_*.jsonl`

That glob is the bench's corpus. Dropping an unlabelled file into it silently corrupts
the measurement, which is why `voice_cli.py` writes `voice_*` and assertion `18c` in
`dry_suite.py` enforces the prefix. To promote one run into the corpus, label it first,
then rename it.
