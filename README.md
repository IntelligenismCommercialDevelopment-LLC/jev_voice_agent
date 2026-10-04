# jev_voice_agent

A spoken-conversation agent whose hard problem is **not** speech recognition or speech
synthesis — both are bought from an API. The hard problem is **turn-taking**: knowing when
the person has finished, and knowing when it is your turn to speak.

Those are two different questions, and this repository's central claim is that they have
to be asked separately, by two independent judges, on two different pieces of text.

```
                     ┌───────────────────────── the ear ──────────────────────────┐
  mic ──▶ STT ──▶ U  (the live transcript: partials overwrite, finals append)
                     │
                     │  ticker, 1 Hz
                     ▼
                JEV#1  "is there enough content to answer yet?"
                     │                              └─ false ─▶ wait
                     │ true                                     (3 strikes with no growth → clear U)
                     ▼
            ┌────────────────┐
            │   reply list   │ ◀── more cuts pile up here while the LLM is busy
            └────────────────┘
                     │
                     │  AND the person has been quiet ≥ llm_min_silence_s
                     ▼
                    LLM     one turn, every pending cut joined with spaces
                     │
                     │ text streams out in chunks
                     ▼
                 GateSink   holds the first chunk, and asks:
                     │
                JEV#2  "has this person finished their turn?"
                     │                              └─ false ─▶ keep holding (up to max_hold_s)
                     │ true
                     ▼
                 MicGate    mutes the mic for as long as we are speaking
                     ▼
                SpeakSink ──▶ TTS ──▶ speaker
                     └──────────────────── the mouth ──────────────────────────────┘
```

Everything in that diagram runs offline under a 151-assertion regression gate that costs
nothing to run. See [The gate](#the-gate).

---

## Why turn-taking is the whole problem

A naive voice agent waits for the speech-to-text provider to say "end of utterance", then
replies. That fails in both directions, constantly:

- **It interrupts.** You say *"I have three questions, the first one is about indexes."*
  That is a grammatically complete sentence and the provider ends the utterance. The agent
  starts answering question one while you are drawing breath for question two.
- **It stalls.** You say *"I wanted to ask"* and pause to think. Nothing in the transcript
  closes, so nothing happens, and the silence stretches until you repeat yourself.

Neither failure raises an error. Both just make the conversation feel wrong. So the system
needs an actual judgement, and it needs two of them, because **the same text gets opposite
answers depending on which question you ask**:

| | JEV#1 — the ear | JEV#2 — the mouth |
|---|---|---|
| Question | Is there **enough content to answer**? | Has this person **finished their turn**? |
| What it reads | `U`, the live transcript since the last submission | the **whole chunk just submitted**, plus what the silence has done since |
| If true | submit into the reply list, start the LLM | open the gate, start speaking |
| *"I have three questions, the first is about indexes."* | **true** — that sentence is answerable | **false** — he obviously has two more |
| *"uh… so… the…"* | false (no content) | false (still hunting for a word) |

Those two rows are the design. One judge cannot answer both questions, and collapsing them
produces exactly the interrupting agent described above.

---

## JEV — what it is, and why it is not a prompt

**JEV** is a judgement call made by a model through a *decisions* API: it takes a `state`
string and a question with explicit `criteria`, and returns a calibrated scalar (`noul`)
plus a verdict. It is not a chat completion, and this repository does not treat it as one.

Three consequences shape the code:

**1. The question file *is* the policy.** Both questions live in version control as
authoritative files — [`prompt_base/listener/question.md`](prompt_base/listener/question.md)
and [`prompt_base/voice_out/question.md`](prompt_base/voice_out/question.md) — and each one
contains exactly one fenced `json` block. The client reads **only that block**; everything
around it is reasoning for humans and never reaches the model. Editing the block is editing
behaviour, so it is reviewed like code.

**2. The examples in the question are hand-written, never copied from the corpus.** If you
paste real recorded utterances into the question and then measure accuracy against those
same recordings, you have measured recitation. The entire corpus stays held out as the test
set. `jev_bench.py` enforces the separation.

**3. A verdict that cannot be read is an error, never a `false`.** If the response comes
back without a readable `noul`, `verdict_from_answer` raises. Degrading to "not complete"
instead would produce a system that *never opens its mouth* while reporting no error
anywhere — the worst failure mode available. This is asserted (`15d`).

Two further details that cost real debugging to learn:

- **Thresholds are scanned, not chosen.** JEV#1's `meaningful_threshold` of 0.65 sits in
  the middle of a plateau found by sweeping a labelled corpus. JEV#2's
  `turn_over_threshold` of 0.6 **has never been scanned** and is a guess; it is annotated
  as a guess in the config and listed under [Known limits](#known-limits).
- **A late verdict is discarded, never acted on, and never cancelled.** Requests go out
  roughly once a second and can return out of order. By the time a slow one comes back, the
  `U` it looked at is stale — acting on it would submit a fragment as though it were
  complete. So stale verdicts are dropped (`16e`). The in-flight request is *not* cancelled,
  because cancelling poisons the connection pool: measured, the next call goes from
  352–421 ms to 1119–1425 ms (`16j`).

---

## The reply list

[`IAF_AI/reply_list.py`](IAF_AI/reply_list.py) is the spine. It exists because speech does
not arrive in neat turns, and because a crashed process must not silently lose what someone
said. Every piece of state is a **file** in the session directory, so the conversation
survives the process dying.

### One utterance, three cuts, one turn

You say something, pause, add to it, pause, add again. JEV#1 passes three times, so three
**cuts** enter the reply list. They do **not** become three LLM calls:

```
cut 1 ─┐
cut 2 ─┼──▶  one drain  ──▶  "cut one cut two cut three"  ──▶  one LLM turn
cut 3 ─┘
```

The cuts are joined with **spaces, not newlines** — newlines nudge the model into answering
each fragment separately (`2`). And when cuts arrive while the LLM is already working, they
simply accumulate and are eaten **in one go** by the next turn (`3`, `17a`). This is the
behaviour the whole structure exists to produce.

### The five pieces

| Piece | Job | The failure it prevents |
|---|---|---|
| **`Reservoir`** | the reply list itself: cuts land here, a turn takes the whole batch | the batch is taken by `os.rename`, so a cut written at the same instant cannot be lost (`10`) |
| **`UserBuffer`** (`U`) | the live transcript: a partial overwrites the current segment, a final freezes it and starts a new one | empty partials and empty finals really do arrive from the provider; unguarded, the first wipes a half-spoken sentence and the second grows the file forever while reading perfectly normally (`13d`, `13e`) |
| **`SilenceCounter`** | how long the person has actually been quiet | it reads the **audio timeline** the provider stamps on each event, not arrival time, and clamps the correction so clock drift can only under-report (`12c`) |
| **`SpokenLedger`** | what was really spoken versus what was merely generated | `reconcile` only ever truncates. Over-reporting would write words into the transcript that never left the speaker — a lying record (`19h`) |
| **`VoiceDriver`** | `pump()` = exactly one turn, start to finish | — |

### A turn has exactly one ending

Earlier versions had three: finish, get interrupted, get invalidated. All of them are gone.
**Once a turn starts, it finishes and it is written down.** If you speak while the machine
is speaking, your words go into the reply list and wait; they do not cut it off.

This is deliberate and it is defended at the source level, because the deleted machinery is
easy to reinstate by accident — restoring one `self.interrupt = asyncio.Event()` turns no
behavioural assertion red, since nobody sets it, but the next person to read the code will
wire the trigger back up. So four assertions (`20h`, `20j`, `21f2`, `22e`) parse the source
with `ast` and fail if the names come back.

### Three places deliberately throw away what a person said

All three leave a trace, because the one thing worse than dropping a sentence is dropping
it silently:

1. **Three strikes** — `U` judged insufficient three ticks running *with no growth*. The
   "no growth" clause is the entire safeguard against cutting off someone who is still
   talking (`16d`).
2. **The delivered watermark** — JEV#1 passes on a partial, `U` is cleared, and then that
   block's *final* arrives and writes the same sentence back into the empty `U`. Without a
   watermark the result is the same sentence answered twice; it happened in two turns out
   of six on a real session. The watermark cuts only a **verbatim prefix**, never a
   character of new text, and declines to cut at all when coverage drops below threshold —
   better to repeat a few words than to bite the head off a sentence (`13i`, `13j`, `13k`).
3. **The hallucination gate** — in one session, three seconds after the mic opened with
   nobody speaking, the provider emitted a confident sentence over a 0.62-second audio
   window. It drove a full LLM turn, a full TTS turn, and was written permanently into
   memory. The test is **speech rate** (11.31 units/sec against a real person's fastest
   4.31), not content, so the gate does not drift with the topic. It defaults to *letting
   things through* when `duration` is missing — losing one hallucination is cheaper than
   swallowing whole batches of real speech (`31b`).

---

## The mouth

The reply does not go straight to the speaker. The first chunk is **held at a gate** while
JEV#2 decides whether you have finished, across three bands:

| Real silence | What happens | Cost |
|---|---|---|
| `< speak_min_silence_s` (1.0 s) | hold, do not even ask | free |
| between the bounds | ask JEV#2, rate-limited to once per extra second of silence | one request |
| `≥ max_hold_s` (6.0 s) | open unconditionally, do not ask | free |

Checking the gate is free; asking JEV#2 costs money. Without the rate limit, dropping the
mouth's tick interval from 1.0 s to 0.25 s would have quadrupled the bill while running
perfectly and raising nothing (`21c2`).

Once the gate opens it forwards transparently — only the first chunk of a turn ever waits
(`21b2`).

**`MicGate`** then mutes the microphone for exactly as long as the machine is speaking,
plus a `hangover_ms` tail for room reverberation. Without it, the shape is: speech out of
the speaker → into the mic → transcribed → drives another turn → a full LLM and a full TTS
every lap, **spending real money**, and since nothing interrupts anything it sounds like a
normal conversation while it happens (`29a`).

It sits **inside** the gate and **outside** the speaking sink. Outside the gate it would
mute the mic while the gate is still waiting — destroying the very silence measurement the
gate depends on (`29f`). It releases in a `finally`, because one exception leaving the mic
muted means permanently deaf, with nothing on screen to say so.

> **`MicGate` is not acoustic echo cancellation.** It stops the machine hearing *itself*.
> It does not stop it hearing another person, or a second device playing audio. The script
> asks you to confirm your setup before opening the mic, every time, and answering anything
> but `y` refuses to start.

---

## Memory

Conversation is appended to `pool/threads/{peer}.md` by **this script, deterministically** —
the model generates text, it never writes the record. The file survives the process exiting,
which also means the last run's conversation is context for the next one.

A **retention window** of recent blocks is injected verbatim every turn. Its size is a
**token budget**, not a block count, because one turn is either one block or two (`_persist`
has two independent branches: one if you spoke, one if the machine did), so "a turn" is not
a well-defined storage unit. `min_keep_blocks` is a hard floor underneath it — a single
5,000-token monologue makes a pure budget compute zero blocks, and the agent then cannot
remember the sentence you just said.

Blocks older than the window get compressed into a summary. Three details earn their
assertions:

- Compression runs in `asyncio.to_thread`. The underlying call is a **blocking** LLM
  request; running it inline jams the event loop, and for those seconds nobody consumes mic
  frames (`24g`).
- Landing the summary locates by **block count captured at staging time**, not by
  recomputing the boundary. Recomputing loses any block that arrived during the several
  seconds compression took — such a block ends up in neither the summary nor the retained
  verbatim text (`24c`).
- The whole file is archived to `pool/threads/_archive/` before the overwrite, because
  compression is lossy and that file is the **only** store of the verbatim words (`24d`).

`/compact` forces a round on demand. **It does not shrink the per-turn injection** — that is
pinned by `keep_recent_tokens`. It buys a longer memory and adds the summary to the
injection as well. Measured on a real session: 33 blocks / 1071 tokens became a 799-token
summary in 4.1 s, and the per-turn injection went *up*, 2007 → 2806 tokens. To shrink
context, change `keep_recent_tokens`.

Summaries are re-compressed on later rounds, and that is intentional: older material gets
progressively more abstract, with length held by the compression prompt's character cap. A
second measured round took the summary 799 → 798 tokens with every hard fact preserved —
convergence, not decay. Assertion `25e` exists to stop someone "fixing" it.

---

## Running it

Requires Python 3.12 or newer (developed and run on 3.14; `dry_suite.py` uses PEP 701
f-strings), a microphone, and **headphones** — or a confirmed loudspeaker setup, see the
note above.

```bash
python3 -m venv .venv
.venv/bin/pip install -r IAF_AI/requirements.txt

# config_base/ is authoritative; sync_config.py pushes it down into IAF_AI/.
# Both copies are gitignored, so put your keys in the authoritative one:
cp config_base/config.example.json config_base/config.json
$EDITOR config_base/config.json          # fill in your provider keys
.venv/bin/python sync_config.py          # -> IAF_AI/config.json
```

Run the gate first — it is free, offline, and takes about a tenth of a second:

```bash
.venv/bin/python IAF_AI/dry_suite.py          # expect: 151/151 passed
```

Then:

```bash
# text out, real ear (mic + STT + JEV#1 + LLM)
.venv/bin/python voice_cli.py

# the whole voice chain: + the gate, JEV#2, TTS, speaker
.venv/bin/python voice_cli.py --tts

# a cheap dry run of the wiring — both judges faked, no JEV spend
.venv/bin/python voice_cli.py --tts --fake-jev len --fake-jev2 enough --seconds 60
```

There is no automatic shutdown by default; the terminal stays up and `/q` ends it. Cost is
controlled by the active/idle cycle instead (below).

### While it is running

Just talk. Typing is a side channel:

| | |
|---|---|
| `enter` | while idle: reconnect STT. While active: nothing. |
| `/s` | counters — ticks, submissions, strikes, silence, spend |
| `/u` | what `U` looks like now, including the tail-trimmed copy fed to JEV |
| `/m` | memory status: window size, blocks awaiting compression, distance to the automatic trigger |
| `/compact` (`/c`) | compress on demand (runs on a worker thread; keep talking) |
| `/h` `/q` | |

### Useful flags

| Flag | |
|---|---|
| `--tts` | add the gate, JEV#2, TTS and the speaker |
| `--peer NAME` | which `pool/threads/{peer}.md` to use (default `user`). **Pick a fresh name for a clean context** rather than deleting files |
| `--fake-jev {len,enough,never}` · `--fake-jev2 {enough,never}` | fake either judge. `len` counts characters — it is *not* a semantic judgement, so never tune thresholds against it |
| `--sleep-ticks N` | override the idle threshold (`0` = never go idle) |
| `--seconds N` | hard cap on mic time. Default `0` = none; it exists as a spend fuse, not a shutdown path |
| `--device N` · `--out-device NAME` | input / output devices (`dispatch/listener/mic_probe.py --list-only`) |
| `--language XX` | override `stt.language` |
| `--tick S` · `--vo-tick S` · `--strikes N` | override ear / mouth knobs |
| `--plain` · `--no-echo` · `--dump-events` · `--tts-verbose` | display |
| `--yes-headphones` | skip the confirmation — on you |

### Active ⇄ idle

STT is billed **per second of audio sent, and silence counts**. So:

```
active   STT connected · mic open · billing
   │  nobody has spoken for sleep_after_ticks ticks AFTER the machine finished
   ▼
idle     ★STT connection cut — not one frame sent — not billing
   │     prints a heartbeat line every announce_every_ticks ticks, so that
   │     "asleep" and "hung" do not look identical
   │  press enter
   ▼
active   a fresh STT connection, mic reopened, silence recounted
```

What survives the idle period: the session, `U`, the reply list, both loops, **the TTS
connection** and both JEV clients. Only the STT websocket is cut — so context is never lost,
and waking does not pay the TTS handshake again (1.4–2.9 s).

The threshold counts from **after the machine stopped talking**, not from the last tick with
no speech. Those differ by the whole length of the reply, and getting it wrong cuts the mic
before you can get a word in — you need about 1.6 s from starting to speak to your first
partial landing (`28a`–`28c`).

### Configuration

| Where | What | Push-down |
|---|---|---|
| `control_config.json` (repo root) | idle thresholds, mic gate, hallucination gate, memory budgets | none — not one character reaches a model |
| `config_base/{organ}/` and `prompt_base/{organ}/` | dispatch configs and the JEV questions | **authoritative**; run `python sync_config.py` to push into `IAF_AI/` |

Editing the copy under `IAF_AI/` directly is pointless — the next sync overwrites it. The
mirror is asserted (`15g`).

---

## The gate

```bash
.venv/bin/python IAF_AI/dry_suite.py
```

151 assertions, no LLM, no network, no microphone, roughly 0.1 seconds, and it writes
nothing outside a sandbox. It is the main reason this repository is worth reading: nearly
every assertion is a regression for a specific failure that **ran, produced no error, and
sounded fine**.

Each assertion carries its own reasoning inline, so the file doubles as the design record —
why barge-in was removed, why `speech_final` is observation-only, why the silence reading is
measured in seconds rather than ticks, why a stub's `duration` has to be physically
plausible.

The structural technique worth stealing: where an assertion must check that a deleted
mechanism has not returned, it parses the source with `ast` (`_code_refs`) rather than
grepping. The prose explaining the deletion necessarily names the deleted thing, so a grep
would match its own documentation and stay permanently red.

---

## Known limits

Stated plainly, because none of these announce themselves at runtime.

- **`turn_over_threshold` (0.6) is a guess.** JEV#1's threshold was scanned across a
  labelled corpus; JEV#2's never was. `jev2_bench.py` is the fix and has not been built.
- **`speak_min_silence_s` (1.0 s) sits below the mid-sentence peak.** The silence reading
  measures "how far the audio we have a transcript for extends", which naturally climbs to
  1.6–2.2 s *while someone is still speaking*. At 1.0 the gate often asks JEV#2 mid-sentence
  — 36 of 38 measured within-sentence gaps. This is a knowing trade-off; raise it toward 2.0
  to tighten, and change nothing else.
- **Every calibrated number here was measured on recordings of one person speaking
  Chinese.** The speech-rate ceiling, the pipeline-lag p50, the thresholds, the retention
  budget. The *shapes* of the defects are language-independent and several were re-observed
  verbatim in English, but **no number in this repository has been re-validated on English
  speech.** Treat them as starting points and re-scan on your own corpus.
- **`fixtures/` ships empty.** The original corpus was recordings of a specific person and
  is not distributed. Record your own with `dispatch/listener/stt_probe.py --save`, label
  `fixtures/jev1_labels.json`, and `jev_bench.py` becomes runnable. Six corpus-replay
  assertions were removed from the gate rather than left permanently skipped.
- **No automatic STT reconnect.** A genuine drop prints a warning and waits for enter.
  Exponential backoff is designed but not built.
- **The TTS session permit expires after 600 s** and this is untreated. The symptom is that
  it quietly stops speaking while every turn still reports success.
- **`pool/semantic.md` is structurally unreachable.** The cross-conversation semantic layer
  can never trigger, because `pool_tok = semantic + a bounded tail` and only pool
  compression writes `semantic`. Whether that layer should exist at all for a single-peer
  voice agent is an open question.
- **Dumped recordings are deliberately named `voice_*.jsonl`, never `stt_*.jsonl`**, because
  `jev_bench.py` globs the latter as its labelled corpus. Dropping an unlabelled file into
  that glob corrupts the bench. Assertion `18c` guards the naming.

---

## Layout

```
voice_cli.py                   the single entry point
control_config.json            runtime switches (never reaches a model)
sync_config.py                 config_base/ + prompt_base/  ──▶  IAF_AI/

IAF_AI/
  reply_list.py                Reservoir · UserBuffer · SilenceCounter · SpokenLedger · VoiceDriver
  assemble_pool.py             the memory envelope and the retention window
  chain.py                     one LLM turn; compression
  dry_suite.py                 the 151-assertion gate
  lib/                         stt_client · tts_client · jev_client · llm_client
  dispatch/
    listener/                  the ear: listener_loop.py, JEV#1, mic_probe, stt_probe, jev_bench
    voice_out/                 the mouth: voice_out_loop.py (the gate), speak_sink.py, JEV#2
    speaker/                   the conversational model
    thread_compressor/         per-conversation compression
    pool_compressor/           cross-conversation compression (see Known limits)

config_base/  prompt_base/     authoritative configs and the two JEV questions
pool/threads/                  the conversation record, verbatim, plus _archive/
```

---

## License

Apache 2.0 — see [LICENSE.txt](LICENSE.txt).
