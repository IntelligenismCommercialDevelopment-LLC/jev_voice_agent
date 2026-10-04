# fixtures/ — the mouth side

This is where the TTS probes write. It ships empty and its contents are gitignored.

```bash
# synthesise a few lines and dump the audio + timings (costs money, no mic needed)
.venv/bin/python IAF_AI/dispatch/voice_out/tts_probe.py

# end-to-end check against the live TTS endpoint
.venv/bin/python IAF_AI/dispatch/voice_out/tts_live_check.py
```

Neither probe opens the microphone, so neither needs headphones — the headphone gate
exists only on the paths that listen while speaking.

`jev2_bench.py`, which would scan `turn_over_threshold` against a labelled corpus the way
`jev_bench.py` does for JEV#1, **has not been built**. That threshold is currently a
guess; see *Known limits* in the top-level README.

Note that a `voice_*.jsonl` recording from the listener side holds only the STT-side
recognition log — it contains nothing the machine said. The verbatim record of the
machine's own speech lives in `pool/threads/{peer}.md`, written deterministically by
`reply_list._persist`.
