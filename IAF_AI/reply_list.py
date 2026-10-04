"""
reply_list.py -- the speech reservoir, the output buffer, and the record of what was
actually spoken (a deterministic script, zero LLM).

**Written from scratch, not ported.** It deliberately shares nothing with a conventional
message-queue mailbox: the semantics are the opposite (consume one item at a time / one
LLM call per item / 3-10 Hz, versus dissolve the whole batch / one call / <=1 Hz).
★Both persist to disk, but the shapes are mirror images: a mailbox holds one JSON object
  per item and re-delivers them; this holds one cut per line, is emptied the moment it is
  taken, and the leftover `.inflight` is archived but **never re-delivered**.

Three parts:

    Reservoir     The reservoir. Fragments carved out by STT are dropped in; the moment the
                  LLM is free it **takes the whole batch and empties it**.
                  -> three cuts != three LLM calls; they are three segments of input to one
                    call.
                  -> a self-correction ("it's charlie... no, john") that lands inside one
                    draw is one single context, and the LLM sorts it out itself -- nobody
                    has to detect the correction.
                  -> natural backpressure: when the person talks faster than the LLM
                    thinks, it simply accumulates.

    FlushBuffer   The chunker for the output stream R. Tokens in, chunks out (on
                  punctuation, or once long enough).
                  A chunk is the natural granularity of "spoken / not spoken" -- the
                  precision of the record rests on it.

    SpokenLedger  R's ledger: which chunks finished playing, and how far into the current
                  one. Clearing R's unspoken part (only one rule left now, "clear on
                  starting a new turn" -- see drop_unspoken) means clearing this.

One iron rule: **what gets written to disk is always the prefix that really was spoken,
never the full text that was intended.** Record the full text and the next turn will quote
back a sentence the other person never heard, with no idea it is doing so.

★★2026-09-27, decided: **barge-in is abolished.** Once the machine starts speaking it
  finishes the segment; the person talking no longer cuts it off. (The old trigger was "any
  non-empty partial" -- one syllable, a friend's remark, a voice on television all counted,
  and a 10-second answer would be wiped out one second in. The cost was out of all
  proportion to the benefit.) Whatever the person says while the machine is speaking still
  goes into the reservoir, and gets answered together on the next turn after the current
  segment finishes -- which is exactly what `pump`'s backpressure was designed for.

★★2026-09-27, same day, also decided: **delete the "invalidate" verdict and the
  "invalidate and re-run" path.** That leaves `reason` with exactly one value -- **once a
  turn has begun it always finishes, and always leaves a record.** Deleted along with it:
  the `VoiceDriver.stale` flag; `Reservoir.put_back()` (push back to the head and re-run);
  the `_Aborted` internal signal; the `aborted` reason (its only trigger was the
  invalidation guard); `GateSink.cancel()/aborted/bind()`.
  ★**These three (barge-in / invalidation / abort) are one and the same kind of thing**:
  each lets something already decided change its mind halfway through. Before adding any
  abort branch to the streaming loop, work out whether you are reinstating them (assertions
  20h/20j guard this).

★The scope of this file ends where "the dispatch emits a text delta" ends; streaming TTS
  lives elsewhere. TTS is therefore a **sink**: the driver hands chunks to the sink, and the
  sink reports back **how many characters really went out**. The default sink just prints
  and claims it all played; swapping in real TTS is a sink swap, with no change to the
  driver.
"""

import asyncio
import collections
import inspect
import os
import re
import sys
import threading
import time
import unicodedata

# assemble_pool sits beside this file (both live in IAF_AI/). Add that directory to the
# path here, so outside callers like dry_suite / jev_bench do not have to do it for us.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import assemble_pool

# The punctuation set from the streaming interface spec, used verbatim. ★The CJK half is
# kept on purpose: the model can still emit CJK punctuation, and an unmatched set would
# silently stop chunking there.
_PUNCT = re.compile(r"[，。！？；,.!?]")


# -- shared foundation for the three state files ------------------------------------------
def _atomic_write(path, text):
    """Write a temp file, then os.replace -- a reader (a fixture replayer, a human running
    tail) can never see a half-written file.

    ★No fsync (decided): the OS still flushes its buffers when a process crashes; only a
      power cut loses anything, and losing one cut of speech does not matter. ~1ms every
      time is not worth it.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _read_text(path, default=""):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return default


# -- the reservoir (file-backed) ----------------------------------------------------------
class Reservoir:
    """Cuts that have been delivered and are waiting for the LLM. **One cut per line**,
    living in `{session}/reply_list.txt`.

    ★Why a file rather than an in-memory list (decided 2026-09-26): all state goes to disk.
      Two things that used to be irreversible come free with it -- a corpus accumulates on
      disk during development (the fixture harness needs no separate collection step), and
      the cut boundaries stay on disk (no longer is a `split(" ")` estimate the only
      evidence of them).

    ★`drain()` goes through `os.rename`: taking the whole file in one step **is** "taken
      means emptied". The seam in "read it, then empty it" swallows a cut written
      concurrently -- and concurrent writes are the normal case for STT, whose callbacks run
      on the websocket's and PortAudio's own threads.

    ★The threading.Lock stays: moving to a file does not make the locking problem go away,
      it only moves it. Within one process, add and drain serialise through it; across
      processes there is no guard (this system has exactly one writer).
    """

    def __init__(self, session_dir):
        self.session_dir = session_dir
        os.makedirs(session_dir, exist_ok=True)
        self.path = os.path.join(session_dir, "reply_list.txt")
        self.inflight_path = os.path.join(session_dir, "reply_list.inflight")
        self._lock = threading.Lock()
        self._added = 0            # purely observational, kept in memory (not worth a file)

    # -- water in --
    def add(self, text):
        """Called when JEV#1 rules "the content is complete" -- ★since 2026-09-27 this is
        the **only** entrance (the `speech_final` fallback was removed by decision).

        ★Newlines get folded into spaces: the contract is one cut per line, and a newline
        inside a cut would make it read as two.
        """
        if not text or not text.strip():
            return
        line = " ".join(text.split())          # fold newlines and runs of whitespace
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._added += 1

    # -- water out --
    def drain(self):
        """Take the whole batch and empty it -- this is the rule "LLM starts work -> clear
        the reply list" itself.

        Returns `(text, n_cuts)`:
          text   = the cuts joined with **spaces** (not newlines: they are several cuts of
                   one utterance, and newlines would push the LLM into treating them as
                   separate questions to answer one by one)
          n_cuts = the **true number of cuts** (lines in the file). ★This is exact, and it
                   replaced the old estimate `len(user_text.split(" "))`, which inflated
                   whenever the input contained spaces -- and which a CLI once worked
                   around with a `total_added` delta that was more wrong still.
        Empty reservoir returns `("", 0)` and **does not touch the file**.
        """
        with self._lock:
            if not os.path.exists(self.path):
                return "", 0
            # ★os.rename must not be allowed to silently overwrite an `.inflight` left by a
            #   previous turn: that file is the crash scene, and the only evidence of it.
            #   POSIX rename overwrites its target without complaint.
            if os.path.exists(self.inflight_path):
                os.replace(self.inflight_path,
                           f"{self.inflight_path}.orphan-{time.strftime('%H%M%S')}")
            os.rename(self.path, self.inflight_path)      # <- atomic, whole file in one step
            raw = _read_text(self.inflight_path)
        cuts = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        return " ".join(cuts), len(cuts)


    def release(self):
        """Delete `.inflight` once a turn has run to completion (and the record is on disk).

        ★This must not happen earlier: an `.inflight` still present means the previous turn
          died midway, and that is the crash-recovery hook. Deleting it early destroys the
          evidence; never deleting it means the next drain collides with it.
        """
        with self._lock:
            try:
                os.remove(self.inflight_path)
            except FileNotFoundError:
                pass

    def inflight_residue(self):
        """Is `.inflight` still there? If so hand back its contents (for recognition --
        **never re-delivered**).

        Decided: no re-delivery. The reason is specific to voice -- a machine that restarts
        and then suddenly answers something you said a few seconds ago is indistinguishable
        from a fault, and since a restart takes only seconds the person has most likely
        already repeated themselves.
        ★Only this turn's own session directory is examined; the previous session's is never
        read.
        """
        if not os.path.exists(self.inflight_path):
            return None
        raw = _read_text(self.inflight_path)
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]

    # -- observation --
    def peek(self):
        with self._lock:
            return " ".join(ln.strip() for ln in _read_text(self.path).splitlines()
                            if ln.strip())

    def __len__(self):
        with self._lock:
            return sum(1 for ln in _read_text(self.path).splitlines() if ln.strip())

    @property
    def total_added(self):
        return self._added


# -- the user buffer U (file-backed) ------------------------------------------------------
def _trim_tail(text, units=1):
    """Shave `units` units off the tail of text. Split on whitespace if there is any,
    otherwise by character.

    ★Shaving it away entirely returns an empty string (not the original) -- when the live
      segment is only a character or two long, those characters are precisely the least
      trustworthy part, and it is better to go without them this tick. Upstream
      `should_skip` rejects an empty state.
    """
    if units <= 0 or not text:
        return text
    if " " in text.strip():
        toks = text.split()
        return " ".join(toks[:-units]) if units < len(toks) else ""
    return text[:-units] if units < len(text) else ""


#: How much of the normalised prefix must match before "the piece already handed over" and
#: "the same piece arriving again" are treated as the same piece.
#: Calibrated from measurement (2026-09-27, 9 recorded sessions, 86 pairs of "partial ->
#: the same-piece event immediately after"): coverage p50=1.00, mean 0.950, p10=0.96; only
#: 6 pairs fell below 0.8 and only 4 below 0.5.
#: 0.5 makes those 4 pairs (where STT re-heard the whole piece as a different sentence, e.g.
#: `Is there is there anybody?` -> `Is any is is anybody here?`) count as "not the same
#: piece" -> **the text goes through untouched, repeating a few characters rather than
#: losing any**.
DELIVERED_PREFIX_MIN_COVER = 0.5

#: Lifetime ceiling (seconds) for the "already delivered" watermark, **counted from the last
#: time it was hit** (see the refresh in `_undeliver`).
#: The normal death of the watermark is **that piece being finalised** (`commit_final` clears
#: it once used), so this ceiling only catches one case: the finalisation for that piece
#: **never arrives** (the connection dropped, or it was the last piece of the session).
#: Without the ceiling the watermark would live on into the next utterance, and if that
#: utterance happened to begin with the same characters (`hello...`) a few characters would
#: be eaten off the front of it -- **no error, the person's first word is simply gone**.
#: 15s: the longest measured "last partial -> finalisation" within one piece was 9764ms,
#: leaving ample margin.
#:
#: ★★2026-09-30: **this number used to be counted from the moment of delivery, which made it
#:   a bug in its own right.** Re-calibrated over all 49 turns of the recorded corpus
#:   ("first partial carrying text -> the finalisation covering it"):
#:     median 6.64s, **max 27.56s**, and 10 turns (about 1 in 5) exceeded 15s.
#:   The old note "longest 9764ms" measured "the **last** partial -> finalisation", i.e. the
#:   tail of a piece; but the delivery point can land on **any** partial of that piece, and
#:   what has to be waited out from there is the remainder of the **whole** piece.
#:   So for an utterance lasting 27.56s the watermark expired mid-sentence -> the full
#:   finalisation that arrived afterwards had nothing trimmed off it -> **the first half of
#:   the same sentence went in twice** (this really happened, and was reproduced
#:   deterministically from the recorded corpus).
#:   The fix is not a bigger 15 (that just moves the threshold outwards; talk longer and you
#:   hit it again) but **refresh on hit**: as long as content for the same piece keeps
#:   arriving, the watermark is not idle. From now on 15s measures "**no further content for
#:   this piece has arrived at all**".
DELIVERED_TTL_S = 15.0


def _sig(text):
    """Normalised character stream, plus each character's index in the original string.

    Whitespace and punctuation dropped, lowercased -- when STT rewrites the same audio,
    **what usually changes is just punctuation and case** (`some,` -> `some`,
    `which model you will` -> `which model you will...`). Comparing the raw strings counts
    those as "different content", and then what should be trimmed is not.
    ★The indices have to come out too: after trimming, the remainder is taken from the
    **original** string (keeping its punctuation and spacing).
    """
    out, idx = [], []
    for i, ch in enumerate(text or ""):
        if ch.isspace():
            continue
        if ch in "…" or unicodedata.category(ch).startswith("P"):
            continue
        out.append(ch.lower())
        idx.append(i)
    return "".join(out), idx


def _strip_delivered(delivered, arrived):
    """Trim "the part already handed over, `delivered`" off `arrived`; return
    `(the remaining original text, coverage)`.

    ★★**This function can only ever remove a prefix that matches `delivered` character for
      character (after normalisation)** -- so it **cannot swallow new content**. That
      property is deliberate: the iron rule here is "rather keep a record than silently drop
      characters", and deduplication that loses words is worse than answering twice.
    ★The cut point **backs up to a word boundary**: the character-wise cut point can land in
      the middle of an English word (measured: `you know Lens` versus `you know Lance DB?`
      cuts to `ance DB?`). Backing up means keeping one possibly-misheard word rather than
      emitting half a word. CJK is one character per word and is unaffected.
    ★When coverage is below `DELIVERED_PREFIX_MIN_COVER`, **nothing is trimmed at all**: that
      means the watermark in hand most likely does not belong to this piece (its own final
      never came, say, and the person has moved on to the next piece), and trimming anyway
      would eat the first few characters of a brand new sentence.
    """
    a, _ = _sig(delivered)
    b, bi = _sig(arrived)
    if not a:
        return arrived, 1.0
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    cover = k / len(a)
    if cover < DELIVERED_PREFIX_MIN_COVER:
        return arrived, cover
    if k >= len(b):
        return "", cover                      # everything that arrived is old content
    cut = bi[k]
    # ★back up to a word boundary (only when neither side of the cut point is whitespace)
    while cut > 0 and not arrived[cut - 1].isspace() and not arrived[cut].isspace() \
            and arrived[cut - 1].isascii() and arrived[cut].isascii():
        cut -= 1
    return arrived[cut:], cover


class UserBuffer:
    """"What has been said since the last delivery", living in `{session}/user_buffer.txt`.

    ★It **must** stay separate from the Reservoir: the clearing rules and the lifetimes
      differ. JEV#1's state is specified to be U and nothing else -- and it is exactly that
      separation that makes the "32k window + context rot" risk evaporate. Merging them into
      one pool takes that guarantee apart.

    File shape = **one segment per line, the last line being the live partial**:
      `set_partial()` rewrites the last line (STT's interim gets rewritten repeatedly)
      `commit_final()` locks the last line down and opens a new one for the next segment
    Reading U = all lines joined with spaces. U only ever holds one utterance (tens to a few
    hundred characters), so rewriting the whole file costs ~1ms and an incremental scheme is
    not worth it (same reasoning as not calling fsync).

    ★★**The delivered watermark** (`_delivered`, 2026-09-27, fixing "the same sentence
    answered twice")

    Symptom: in one real run of six turns, two turns answered the same sentence twice (steps
    24/26 word for word in the archive; steps 32/34 were "first half" plus "whole
    sentence").

    The cause is in the **shape of the protocol**, not in the verdict. The STT service splits
    one utterance into several **pieces**:
      - a `partial`'s text = the draft of **the current piece**, rewritten repeatedly (this
        class's `set_partial` overwrites the last line)
      - a `final`'s text   = the finalisation of **the current piece**; `start` is the start
        of the whole turn and `duration` keeps growing.
        ★★2026-09-30, corrected by measurement: **both shapes really occur**; only the first
          was documented originally --
          (1) **piece by piece**: two finalisations in a row under the same `start`, the
             later one **not containing** the earlier (one recorded session at `start=53.88`:
             "uh, I wanted to ask, so if we say" is followed by ", uh, I'll just give you
             these dimensions directly...") -> hence the `append("")` that opens a new
             segment; without it the next piece's partials would **overwrite the whole
             previous segment** (the first half of what the person said vanishes silently).
          (2) **whole-turn cumulative**: one finalisation for the entire turn carrying the
             full text (same corpus, `start=149.18`: 30 partials growing from "so, uh" to a
             107-character complete sentence, then one finalisation closing it out).
          Both have to be caught, so "append or overwrite" can only be decided by the `final`
          signal itself, never guessed by comparing text.
      - a `speech_final`'s text = the cumulative whole utterance (discarded here, see
        `listener_loop` discipline 2)
    Hence: JEV#1 passes on **a partial of the live piece** -> `_submit` takes the whole text
    and calls `clear()`, but **the piece that produced that text has not been finalised
    yet**. Its `final` arrives 250-9764ms later, and `commit_final` writes the finalisation
    of the same audio back into the **just-emptied** U -> on the next tick JEV#1 passes
    again -> the same sentence enters the reply list twice -> **answered twice**. Across 9
    recorded sessions, 9 out of 9 non-empty finals had a same-piece partial before them, so
    this is not occasional: **every pass on a partial hits it**.
    (A second real run of three turns was entirely clean, precisely because all three passes
    landed after the final had already arrived.)

    The fix: on `clear()`, if **the last line (the live piece) is non-empty**, record it as
    the "delivered watermark"; same-piece content arriving afterwards goes through
    `_strip_delivered()` first to trim that part off. The watermark is **void once that piece
    is finalised** (`commit_final` clears it after use), so it only lives inside the window
    "halfway through one piece".
    ★When the last line is empty (the finalisation is already locked in) **no watermark is
      set** -- a typed input path makes every line a final and always takes this branch,
      so not one character is ever trimmed there.
    ★Everything trimmed is recorded in `self.dedup` (the only other place that deliberately
      drops characters, the three-strikes rule, keeps a record too). The counts surface in
      `listener_loop.summary()`.
    """

    def __init__(self, session_dir):
        self.session_dir = session_dir
        os.makedirs(session_dir, exist_ok=True)
        self.path = os.path.join(session_dir, "user_buffer.txt")
        self._lock = threading.Lock()
        # ★The delivered watermark: the live piece (last line) as it stood at `clear()`.
        #   See the class docstring.
        self._delivered = None
        self._delivered_at = 0.0
        self.dedup = collections.deque(maxlen=20)   # record: what was trimmed
        self.dedup_events = 0                       # arrivals that were entirely old -> dropped
        self.dedup_chars = 0                        # total characters trimmed
        # ★★**How the watermark died** has to be recorded (2026-09-30). Originally only "how
        #   many characters were trimmed" was counted, and "expired on timeout" was
        #   **completely silent** -- which is a shame, because it is the one death that means
        #   "a safety mechanism quietly stopped working": after an expiry the late
        #   finalisation has nothing trimmed off it and the first half is delivered a second
        #   time.
        #   Three deaths: `disarmed` the finalisation arrived (normal); `expired` timed out
        #   (the backstop caught it, but that piece really was cut off); and process exit,
        #   which is not recorded.
        #   ★`max_idle_s` = the largest "time since same-piece content last arrived" the
        #     watermark ever reached. How far that is from `DELIVERED_TTL_S` is the **margin**
        #     left in this backstop.
        #     Worst case on the recorded corpus was 12.75s (434 adjacent event pairs, median
        #     1.00s) -> 2.25s of margin, but that is a 4-session sample; the network and the
        #     speaking rhythm on site can only be measured on site.
        #     `expired > 0` is direct evidence that the margin has been eaten through.
        self.delivered_stats = collections.Counter()
        self.delivered_max_idle_s = 0.0
        self.delivered_expired = collections.deque(maxlen=10)   # record: what was in hand at expiry

    def _segments(self):
        return _read_text(self.path).split("\n")

    def _flush(self, segs):
        _atomic_write(self.path, "\n".join(segs))

    def set_partial(self, text):
        """An STT interim partial: **overwrite** the current segment (do not append).

        ★An empty partial is ignored outright (**the STT service really does send empty
          partials** -- 4 of them in the second probe run): without this it would blank the
          current segment, i.e. **silently drop** what the person was halfway through
          saying. In the first probe run they all landed after `speech_final` (U already
          cleared, harmless), but one landing mid-sentence is soundless data loss.
          ★To genuinely retract text the server sends a **shorter** partial, never an empty
          one.
        """
        if not (text or "").strip():
            return
        text = self._undeliver(text, "partial")
        if not text:
            return
        line = " ".join(text.split())
        with self._lock:
            segs = self._segments() or [""]
            segs[-1] = line
            self._flush(segs)

    def commit_final(self, text):
        """STT's `is_final`: lock this segment down and open a new one.

        ★Empty text is ignored outright (**the STT service really does send empty finals** --
          3 of them in 15 seconds of the first probe run): without this, every one would
          append a blank line and `user_buffer.txt` would grow without bound -- and since
          `text()` filters blank lines it **reads perfectly normally and raises nothing**,
          the file just becomes unreadable.
        """
        if not (text or "").strip():
            return
        text = self._undeliver(text, "final")
        with self._lock:
            # ★★Finalisation voids the watermark, and **this runs whether or not anything was
            #   left after trimming**: this piece ends here, the next piece has nothing to do
            #   with it, and keeping the watermark would eat the head of that next piece.
            if self._delivered:
                self.delivered_stats["disarmed"] += 1      # the normal death
            self._delivered = None
        if not text:
            return
        line = " ".join(text.split())
        with self._lock:
            segs = self._segments() or [""]
            segs[-1] = line
            segs.append("")
            self._flush(segs)

    def _undeliver(self, text, kind):
        """Trim "the part already handed over" off newly arrived text (see the delivered
        watermark in the class docstring).

        No watermark means the text comes back unchanged -- which is the branch the vast
        majority of events take.
        """
        with self._lock:
            delivered = self._delivered
            idle = (time.time() - self._delivered_at) if delivered else 0.0
            if delivered and idle > DELIVERED_TTL_S:
                # ★★The backstop really fired = that piece was cut off for good. Count it and
                #   keep a record (see the comment in __init__): from here on, same-piece
                #   content that arrives has nothing trimmed -> it may be delivered twice.
                self.delivered_stats["expired"] += 1
                self.delivered_expired.append(
                    {"delivered": delivered, "idle_s": round(idle, 2),
                     "arrived": (text or "")[:60], "kind": kind})
                delivered = self._delivered = None      # expired: no finalisation ever came
            elif delivered:
                # ★Only measure the **not expired** side: that is the reading for "how far
                #   from the TTL is there still room".
                self.delivered_max_idle_s = max(self.delivered_max_idle_s, idle)
        if not delivered:
            return text
        rest, cover = _strip_delivered(delivered, text)
        # ★★Only a partial refreshes the watermark, **never a final** (2026-09-30):
        #   `commit_final` disarms it immediately afterwards, so refreshing there is a clock
        #   write with no effect, and it inflates `delivered_stats["refreshed"]` -- a number
        #   meant to say "how many times its life was extended", not "how many times it was
        #   hit". Assertion 31j is what caught this.
        if kind != "final" and cover >= DELIVERED_PREFIX_MIN_COVER:
            # ★★Refresh on hit (2026-09-30): text arriving that matches the watermark
            #   character for character means **this piece is still coming in**, so the
            #   watermark is not idle. Without the refresh the TTL measures "since delivery",
            #   and an utterance can run 27.56s (the measured maximum, see the calibration on
            #   `DELIVERED_TTL_S`) -> the watermark expires mid-sentence -> the full
            #   finalisation that follows has nothing trimmed -> the first half goes in twice.
            #   ★Use `cover` rather than `rest != text` to detect a hit: when the cut point
            #     backs up to a word boundary it can land exactly on 0, and then `rest == text`
            #     even though this really **is** the same piece -- without `cover` that
            #     refresh would be missed.
            with self._lock:
                self._delivered_at = time.time()
                self.delivered_stats["refreshed"] += 1
        if rest != text:
            # ★Record it. This is the second place in this class that deliberately throws
            #   away something the person said (the other is three strikes), and both are
            #   required to leave a record: when someone later asks "why did the machine miss
            #   half of what I said", there has to be something to look at.
            self.dedup.append({"kind": kind, "delivered": delivered,
                               "arrived": text, "kept": rest,
                               "cover": round(cover, 3)})
            self.dedup_chars += len(text) - len(rest)
            if not rest:
                self.dedup_events += 1
        return rest

    def text(self):
        """The state fed to JEV#1."""
        with self._lock:
            return " ".join(s.strip() for s in self._segments() if s.strip())

    def text_for_jev(self, trim_units=1):
        """**Shave the tail off the live segment** before feeding JEV#1.

        ★Why this is mandatory: a cumulative partial **does not merely grow longer and more
          accurate -- it revises downwards**. Measured: `... with the L L M.` was followed by
          a partial reading `... with the L M is`. So the last unit or two of a partial are
          not "not yet complete", they are **possibly wrong**. Using them for an irreversible
          act like submitting is the reason tail-shaving is **a correctness requirement, not
          an economy**.
        ★Only the **live segment** (the last line) is shaved; locked finals are untouched --
          they will not change again.
        ★CJK has no word boundaries: split on whitespace when there is any (English, or mixed
          text), by character when there is not.
          The cost is a few hundred milliseconds (accepted), and what it buys is not drawing
          conclusions from text that is about to be rewritten.
        """
        with self._lock:
            segs = [s for s in self._segments()]
            locked = [s.strip() for s in segs[:-1] if s.strip()]
            live = segs[-1].strip() if segs else ""
        live = _trim_tail(live, trim_units)
        parts = locked + ([live] if live else [])
        return " ".join(parts)

    def clear(self):
        """★All or nothing; there is no "clear up to a point". JEV's verdict is a boolean --
        it does not tell us where the complete part ends. Clearing precisely would need JEV
        to return a `keep_from` as well, which is a separate problem and not open at this
        stage.

        ★★Clearing **also arms the delivered watermark** (2026-09-27, fixing "answered
          twice"; see the class docstring): a non-empty last line (the live piece) means that
          piece was handed over before being finalised -> record it, so that the same-piece
          finalisation arriving later writes back only **the part that newly grew**. An empty
          last line records nothing.
        """
        with self._lock:
            segs = self._segments()
            live = segs[-1].strip() if segs else ""
            self._delivered = live or None
            self._delivered_at = time.time()
            if self._delivered:
                self.delivered_stats["armed"] += 1
            _atomic_write(self.path, "")

    @property
    def delivered_ttl_s(self):
        """The **current value** of the watermark's backstop ceiling.

        ★It reads the module-level constant rather than a snapshot taken at construction --
          that is what makes `dry_suite` changing `reply_list.DELIVERED_TTL_S` take effect
          (the same reasoning as the `load_control(path=None)` test seam).
        ★This accessor exists so that `listener_loop.summary()` can report "longest idle /
          ceiling" side by side, while `listener_loop` **deliberately does not import
          reply_list** (it only reaches things through the driver) -- breaking that seam for
          the sake of one constant is not worth it.
        """
        return DELIVERED_TTL_S

    def char_len(self):
        """Character count. ★The anti-false-positive condition in the three-strikes rule
        ("U has not grown") reads this."""
        return len(self.text())

    def __len__(self):
        return self.char_len()


# -- the silence counter (two readings) ---------------------------------------------------
#: Ceiling (seconds) on the pure pipeline delay, "event arrival - when that audio actually
#: ended". Measured 2026-09-27: 45 real events, p50 677ms / p90 779ms / max 1123ms, the same
#: for partial, final and speech_final alike.
#: It **clamps** the delay compensation inside `silence_seconds()` -- the audio timeline and
#: the system clock can drift apart over a long run, and drifting towards "larger
#: compensation" would **over-report silence** -> the machine talks over the person. Clamping
#: can only under-report.
MAX_PIPELINE_LAG_S = 1.5


class SilenceCounter:
    """"How long the user has been quiet." Two readings, **used for different things; do not
    mix them up**:

    - `silence_seconds()` -- **for control**. Continuous seconds, on the audio timeline (see
      below).
    - `value()` / `{session}/silence_ticks` -- **observation plus the sleep decision**. An
      integer tick count, 1 byte, overwritten every tick. The only **continuous** quantity
      used for control is still `silence_seconds()`; this tick reading has exactly one
      control consumer = the sleep decision in `voice_cli.silence_watch`
      (`sleep_after_ticks`).
      ★The full list of consumers, so nobody has to count them again:
          silence_seconds() -- 3 control criteria:
            1. listener_loop.llm_gate_ok()   >= llm_min_silence_s   -> fire the LLM
            2. voice_out_loop.tick_once()    <  speak_min_silence_s -> hold, do not ask JEV#2
            3. voice_out_loop.tick_once()    >= max_hold_s          -> pass unconditionally
          value() / the silence_ticks file -- exactly 1 control criterion:
            4. voice_cli.silence_watch()     >= sleep_after_ticks   -> cut STT, go idle

    ★Not a boolean flag (decided 2026-09-26): flipping to "speaking" has an event (a partial
      arrived), flipping back to "not speaking" has **no event** -- silence produces no push.
      So it has to be computed from "how long since the last event".

    ★★2026-09-27, switched to the audio timeline. ★★**The conclusion goes first, because the
      premise I started from was wrong**: I believed the ~1.4s in the tick-based reading was
      compressible (1. 677ms of pipeline delay, incompressible; 2. "a whole tick must pass
      empty", ~1000ms; 3. within-tick quantisation, 0-1000ms -- and I thought 2 and 3 could
      be saved). **2 cannot be saved.** `now - stop_wall` is not "how long the person has
      been quiet", it is "how far we have transcribed audio up to" -- STT sends a partial
      every ~958ms describing audio from ~677ms ago, so **while the person is still talking
      this reading already climbs to 1606ms (p50) / 2185ms (max)**. The information floor for
      inferring "they stopped" from the gaps between events is 677+960 ~= 1.6s, and the
      measured minimum of the old tick reading was exactly 1600ms -- it was already sitting
      on the floor.
      Scanning the recorded corpus (4 sessions, 7 genuine stops, 38 mid-sentence gaps), the
      two are identical at **waiting for a safe point**: the old one false-triggered 4/38 at
      2100ms; the new `>=2.0s` does 5/38 at 2100ms.
      -> What this change bought: 1. the latency goes from "wherever the tick boundary
        happens to fall" (1600-2440ms) to a **fixed value**; 2. the ear's `tick_interval_s`
        no longer also determines the silence resolution (those two constraints pull in
        opposite directions); 3. the knob's unit matches its name. **It did not buy latency.**
      -> The real lever on time-to-speak is `speech_final` (server-side VAD, only 625-767ms
        behind a genuine stop), but 4 of its 7 fired early (the person then carried on, up to
        2840ms late) -> it can only serve as the trigger for "start asking JEV#2 now", and
        that path is blocked on JEV#2 being uncalibrated (`turn_over_threshold` 0.6 is a
        guess).

      The mechanism itself: every STT event carries its own `start` + `duration` (the audio
      timeline), and `stop_wall = mic_epoch + start + duration` is the wall-clock moment that
      audio ended. Hence:
          silence = (now - last event arrival) + min(that event's pipeline delay, MAX_PIPELINE_LAG_S)
      One expression covers both cases: with no `stop_wall` available the delay term is 0 and
      it degrades to "how long since the last event" -- still continuous, still better than
      ticks, and **under-reporting** (conservative: the machine waits rather than interrupts).

    ★★While here, a comment that had been wrong for a long time: it used to say "`counter > 0`
      guarantees nothing (it may have been quiet for 1 millisecond when the tick landed);
      only `counter >= N` guarantees silence >= (N-1) seconds". **That is wrong by a whole
      step**: the flag set by `mark_voice()` is consumed by the **next** tick into a reading
      of 0, so for the reading to reach 1 a whole tick must pass with no events at all.
      Reconciled tick by tick against 8 recorded sessions: when the reading is 1, the minimum
      "time since the previous event" is 1024ms, with no counterexample. The backlog item
      "raise `speak_min_silence_ticks` 1->2, because the documented >=1s lower bound is not in
      effect" was filed on the strength of that wrong comment -- and has been withdrawn.

    ★The ticker is the only writer of the `silence_ticks` **file**. STT callbacks touch
      memory only (a "there was speech this tick" flag plus two timestamps) -- letting the
      callbacks write the file themselves would make "read 3 -> reset to zero -> write back 4"
      swallow the reset, and raise nothing.

    ★`clock` is a **test seam**: the mechanism is time-driven, so an offline harness has to be
      able to control time, or "drive it deterministically tick by tick" degrades into
      padding with sleeps (`dry_suite`'s `_silent_for` is what drives it).
    """

    def __init__(self, session_dir, clock=None):
        self.session_dir = session_dir
        os.makedirs(session_dir, exist_ok=True)
        self.path = os.path.join(session_dir, "silence_ticks")
        self.clock = clock or time.time
        self._lock = threading.Lock()
        self._voice_this_tick = False
        self._last_voice_wall = None    # wall clock when the last STT event ARRIVED
        self._stop_wall = None          # wall clock of the audio END the last STT event covered
        self._created = self.clock()
        _atomic_write(self.path, "0")

    def set_clock(self, fn):
        """Swap the clock (offline harness only). ★**This method must be used; never assign
        `self.clock` directly** -- `_created` was taken with the old clock, and without
        re-basing it `silence_seconds()` computes a huge negative `new clock - old clock` on
        the "this session has not heard anything yet" branch -> clamped to **0, forever 0**.
        The gate then holds forever, **and not one error is raised** (this has been hit for
        real).
        """
        with self._lock:
            self.clock = fn
            self._created = fn()
            if self._last_voice_wall is not None or self._stop_wall is not None:
                raise ValueError("the clock must be swapped BEFORE anything is heard -- "
                                 "the two timestamps were taken with the old clock and are "
                                 "not comparable afterwards")

    def mark_voice(self, stop_wall=None):
        """The only thing STT callbacks touch (safe to call across threads). **Writes no
        file.**

        `stop_wall` = the end of the audio this event covers, converted to wall clock
        (`stt_client` computes it and puts it in the event). The max is taken because what we
        want is "the latest moment the person was still making sound"; a late event may carry
        an earlier audio end. Clamped to `now` -- an event cannot describe future audio
        (guards against clock drift).
        """
        with self._lock:
            self._voice_this_tick = True
            now = self.clock()
            self._last_voice_wall = now
            if stop_wall is not None:
                sw = min(float(stop_wall), now)
                if self._stop_wall is None or sw > self._stop_wall:
                    self._stop_wall = sw

    def silence_seconds(self):
        """★The control reading: how many seconds this person has **really** been quiet
        (continuous, safe to read across threads)."""
        with self._lock:
            if self._last_voice_wall is None:
                return max(0.0, self.clock() - self._created)   # nothing heard yet this session
            lag = 0.0
            if self._stop_wall is not None:
                lag = min(max(self._last_voice_wall - self._stop_wall, 0.0),
                          MAX_PIPELINE_LAG_S)
            return max(0.0, (self.clock() - self._last_voice_wall) + lag)

    def tick(self):
        """Called once per tick by the ticker. Speech -> reset to zero; none -> +1. Returns
        the new reading.

        ★This reading has **exactly one control consumer** = the sleep decision in
          `voice_cli.silence_watch` (`sleep_after_ticks`); apart from that it is only a
          record (see the class docstring).
        ★★The original note here said "as of 2026-09-27 it no longer enters any control
          criterion" -- **that was wrong**, and `silence_watch` was added the same day. Three
          places said it entered no criterion, which would lead the next person to believe
          changing `value()`'s behaviour was safe. The full consumer list is in the class
          docstring.
        """
        with self._lock:
            had_voice = self._voice_this_tick
            self._voice_this_tick = False
            cur = 0 if had_voice else self._raw() + 1
            _atomic_write(self.path, str(cur))
            return cur

    def _raw(self):
        try:
            return int(_read_text(self.path, "0").strip() or 0)
        except ValueError:
            return 0

    def value(self):
        with self._lock:
            return self._raw()

    def can_speak(self, min_ticks=2):
        """★**Not a gate; do not use it as one.** The real gate is the mouth's
        `speak_min_silence_s`, which goes through `silence_seconds()` (the audio timeline) --
        **a different unit** from the ticks here.

        The only call site is `listener_loop.summary()`, which is the `/s` command and the
        report block, i.e. a display path -- so it **is live, it is just misnamed**: the name
        (can_speak) and its corresponding knob (`silence_ticks_to_speak`) both claim to be
        the criterion for whether speaking is allowed, while the real criterion is elsewhere.
        Reported and awaiting a decision.
        """
        return self.value() >= min_ticks


# -- output chunking ----------------------------------------------------------------------
class FlushBuffer:
    """Tokens in, chunks out. A chunk = the smallest unit handed to TTS = the granularity of
    the record.

    ★There is one **deliberate difference** from the reference implementation in the
      streaming interface spec, **and it follows that spec's own prose**: in the reference
      implementation `min_chars` is stored but never used, and `if m and m.end() >= 1` is
      always true once punctuation matches -- so it **flushes on every single punctuation
      mark**, forming a chunk out of two characters and a comma.
      The spec's prose, meanwhile, says to flush "once roughly 8-20 characters have
      accumulated **or** punctuation is reached", the reason given being "to reduce speaking
      half a word out loud". The two disagree.
      This follows the **prose** (punctuation must also be long enough to cut), because that
      is the intent the spec declares.

    The first chunk is more aggressive ("push the first 8 characters early"): it uses
    first_min_chars, lowering time to first sound.
    """

    def __init__(self, min_chars=8, max_chars=24, first_min_chars=4):
        self.buf = ""
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.first_min_chars = first_min_chars
        self._emitted = 0

    def _threshold(self):
        return self.first_min_chars if self._emitted == 0 else self.min_chars

    def push(self, delta):
        """Feed in one delta; return the chunks that formed this time (possibly none)."""
        self.buf += delta or ""
        out = []
        while True:
            thr = self._threshold()
            # ★It must find **the earliest punctuation at or past the threshold**, not use
            #   _PUNCT.search(). search returns only the first mark: in "ok, let me take a
            #   look." the comma is at character 2, 2 < threshold -> rejected, and the full
            #   stop that really should have cut **is never looked at again**, so the chunk
            #   waits for the hard ceiling of 24. The symptom is time to first sound suddenly
            #   waiting out a whole sentence, with no error.
            #   (The spec's reference implementation does not have this problem, purely
            #   because its threshold test is always true.)
            cut = None
            for m in _PUNCT.finditer(self.buf):
                if m.end() >= thr:
                    cut = m.end()
                    break
            if cut is not None:
                out.append(self.buf[:cut])
                self.buf = self.buf[cut:]
                self._emitted += 1
                continue
            if len(self.buf) >= self.max_chars:
                # Hard ceiling: punctuation may never come, but it cannot accumulate forever
                # or one long sentence drags time to first sound down with it
                out.append(self.buf[:self.max_chars])
                self.buf = self.buf[self.max_chars:]
                self._emitted += 1
                continue
            break
        return out

    def flush(self):
        """At end of stream, emit the tail (possibly under min_chars, but nothing follows)."""
        if not self.buf:
            return []
        s, self.buf = self.buf, ""
        self._emitted += 1
        return [s]


# -- R's ledger: spoken / unspoken --------------------------------------------------------
class SpokenLedger:
    """Tracks the playback state of every chunk in R.

    Three states are enough:
      finished     the whole chunk reached the speakers
      part-way     only the first n characters of this chunk played (reported by TTS). With
                   barge-in gone, only exceptional paths produce this
      unspoken     still in R, never emitted
    """

    def __init__(self):
        self.chunks = []        # [{"text":str, "spoken":int}]  spoken = characters played

    def add_chunk(self, text):
        self.chunks.append({"text": text, "spoken": 0})

    def mark_spoken(self, n_chars=None):
        """Mark the earliest unfinished chunk as having played n_chars characters
        (None = the whole chunk)."""
        for c in self.chunks:
            if c["spoken"] < len(c["text"]):
                c["spoken"] = len(c["text"]) if n_chars is None else min(n_chars, len(c["text"]))
                return c
        return None

    def reconcile(self, total_spoken_chars):
        """★Recompute the three states from **the total number of characters that really
        made a sound this turn** (added 2026-09-27).

        Why it is needed: with real TTS the sink **has a prefetch buffer** -- it hands the
        chunk to the player and returns, at which point that chunk has not been spoken yet
        (without prefetch, 0.5-1.0 seconds of silence appears between chunks, measured at 18%
        of total duration). So `mark_spoken(None)` is **optimistic** at that moment: the
        ledger says finished while it is still in the queue. At settle time the sink reports
        the true total, and the ledger is corrected here.

        ★The direction matters: over-reporting (recording as spoken something that never came
          out) makes the next turn believe the other person heard it, which is a lying
          record; under-reporting only makes the N in "N characters never spoken" a little
          large. So this always truncates **back** to the true total, never extends forward.
        """
        left = max(0, int(total_spoken_chars))
        for c in self.chunks:
            n = min(left, len(c["text"]))
            c["spoken"] = n
            left -= n
        return total_spoken_chars

    def spoken_text(self):
        """★The prefix that actually played. This, and only this, goes to disk."""
        return "".join(c["text"][:c["spoken"]] for c in self.chunks)

    def drop_unspoken(self):
        """The clearing rule itself: starting a new turn supersedes and clears R's unspoken
        part. (There used to be a second rule, "clear on invalidation", deleted 2026-09-27
        along with the invalidation mechanism.)

        A purely mechanical reason: anything left unemitted in R was necessarily thought up
        with **less information** than this new turn has, and keeping it only means saying a
        degraded version once. Returns the number of characters dropped (for the record and
        for observation).
        """
        dropped = 0
        kept = []
        for c in self.chunks:
            n = c["spoken"]
            dropped += len(c["text"]) - n
            if n > 0:
                kept.append({"text": c["text"][:n], "spoken": n})
        self.chunks = kept
        return dropped


# -- the default sink (the placeholder seam for TTS) --------------------------------------
async def console_sink(chunk):
    """The default outlet: print, and claim the whole chunk played.

    ★Plugging in real TTS is a matter of swapping this function; the driver does not change
      by a line. The protocol has exactly one clause: **return the number of characters
      actually spoken** (None = the whole chunk). If only part played, return how many.
    """
    print(chunk, end="", flush=True)
    return None


# -- the driver: wiring the reservoir, the LLM, R and the record together -----------------
class VoiceDriver:
    """A half-duplex driver. One LLM turn at a time; water arriving while it runs accumulates
    in the reservoir.

    Usage (with no real STT/TTS):
        drv = VoiceDriver(dispatch, session_dir, peer="user")
        drv.reservoir.add("I wanted to ask about the quote")   # <- an STT submission (JEV#1 ruled)
        await drv.pump()                                       # <- run one turn
    In real operation pump hangs off the STT push callback, or waits for water in run_forever.
    """

    def __init__(self, dispatch, session_dir, peer="user", sink=console_sink,
                 agent_name="core", flush_kwargs=None, verbose=True):
        self.disp = dispatch
        self.session_dir = session_dir      # workspace/{session}/ <- envelope.md lands here
        self.peer = peer
        self.sink = sink
        self.agent_name = agent_name
        self.flush_kwargs = flush_kwargs or {}
        self.verbose = verbose
        # ★All three state files live in session_dir. VoiceDriver already takes session_dir,
        #   so chain.new_voice_session needed no change at all.
        self.reservoir = Reservoir(session_dir)        # delivered, waiting for the LLM (one cut per line)
        self.user_buffer = UserBuffer(session_dir)     # U: what has been said since the last delivery
        self.silence = SilenceCounter(session_dir)     # silence ticks, the gate for TTS
        self.busy = False
        # ★★There were once two abort flags here; **both were deleted outright, with no empty
        #   Event left behind** (anti-bloat rule: leaving one makes the next person think
        #   something still sets it): `self.interrupt` (2026-09-27, barge-in abolished) and
        #   `self.stale` (2026-09-27, the invalidation verdict and its re-run deleted).
        #   -> **Nothing can now abort a turn that has begun**: a turn has exactly one
        #   ending = it finishes speaking.
        self.turns = []                       # observational record of each turn
        # Hook run after each turn (signature hook(rec) -> any). chain.new_voice_session uses
        # it to attach maybe_compress -- compression is non-streaming and must come AFTER the
        # turn, never in front of the spoken reply.
        # An exception inside the hook must not break the conversation: this turn's words are
        # already out, and a failed compression should not affect them.
        self.on_turn_end = None

    # -- the record ------------------------------------------------------------------
    def _persist(self, user_text, ledger, reason, dropped):
        """Deterministically write this turn into pool/threads/{peer}.md.

        ★The writer is this script (deterministic), not the LLM. A read/write wall: the LLM
          only generates text.
        ★What goes down is ledger.spoken_text(), not the full text the model produced.
        """
        step = assemble_pool.next_step(self.peer)
        if user_text.strip():
            assemble_pool.append_thread(self.peer, "in", step, user_text)
        spoken = ledger.spoken_text()
        if spoken.strip():
            note = ""
            if dropped:
                # ★This note is for the next turn of this same agent: it has to know the
                #   sentence was not finished, or it will assume the other person heard all
                #   of it. It is the only trace of the state "decided, but only half carried
                #   out".
                note = f"\n(This segment ended with reason '{reason}'; {dropped} characters were never spoken.)"
            assemble_pool.append_thread(self.peer, "out", step + 1, spoken + note)
        return step

    # -- one turn --------------------------------------------------------------------
    async def pump(self):
        """Water in the reservoir and the LLM idle -> run one turn. Returns this turn's
        record; returns None when dry or busy."""
        if self.busy or len(self.reservoir) == 0:
            return None

        self.busy = True
        try:
            # -- rule 1: the LLM starts work -> clear the reply list (take the batch, which
            #    empties it)
            # ★n_cuts is now the **true number of cuts** (lines in the file), not a
            #   split(" ") estimate. The estimate inflated whenever the input contained
            #   spaces; a CLI once worked around it with a total_added delta, which was
            #   more wrong still (it measured "cuts arriving during this turn", and attributed
            #   them to the wrong turn as well).
            user_text, n_cuts = self.reservoir.drain()

            # ★Rewrite the envelope before starting: it is the speaker's only entrance to
            #   memory. It must be written before new_session -- context_files are read
            #   inside run_stream, and writing it late means this turn reads the previous
            #   turn's envelope, with no error raised.
            assemble_pool.assemble_voice(self.session_dir, self.peer)

            fb = FlushBuffer(**self.flush_kwargs)
            ledger = SpokenLedger()
            t0 = time.time()
            first_chunk_ms = None
            # ★★A constant (as of 2026-09-27): with barge-in abolished and invalidation
            #   deleted, a turn has **exactly one ending**. The field survives because it is
            #   part of the record format in pool/threads (changing the format means touching
            #   historical archives), not because it can still vary. Adding a second reason
            #   means reinstating an abort path -- think it through first.
            reason = "done"
            gen_id = None
            usage = {}

            self.disp.new_session(agent_name=self.agent_name)
            self.disp.append_to_history("user", user_text)

            stream = self.disp.run_stream(agent_name=self.agent_name)
            try:
                async for ev in stream:
                    kind = ev.get("type")
                    if kind == "meta":
                        gen_id = ev.get("generation_id")
                        continue
                    if kind == "done":
                        usage = ev.get("usage") or {}
                        continue
                    # delta
                    for chunk in fb.push(ev["text"]):
                        if first_chunk_ms is None:
                            first_chunk_ms = int((time.time() - t0) * 1000)
                        ledger.add_chunk(chunk)
                        spoken_n = await self.sink(chunk)
                        ledger.mark_spoken(spoken_n)
                        # ★★There was once a single abort check here (invalidation). Since it
                        #   was deleted on 2026-09-27, **nothing can abort a turn that has
                        #   begun** -- do not add an abort branch to this loop, which would
                        #   amount to reinstating barge-in and invalidation (assertions
                        #   20h/20j guard this).
                # The stream ended normally: play out the tail as well
                for chunk in fb.flush():
                    # ★The first-chunk timestamp has to be stamped here too: a short
                    #   reply (under first_min_chars and with no punctuation -- "Yep", say)
                    #   **never forms a chunk at all** during streaming and only gets out
                    #   through this flush. Miss this line and the symptom is
                    #   first_chunk_ms permanently None -- time to first sound, the thing
                    #   voice cares about most, becomes unmeasurable on exactly the
                    #   shortest replies, and nothing reports it.
                    if first_chunk_ms is None:
                        first_chunk_ms = int((time.time() - t0) * 1000)
                    ledger.add_chunk(chunk)
                    spoken_n = await self.sink(chunk)
                    ledger.mark_spoken(spoken_n)
            finally:
                await stream.aclose()

            # -- settle: a sink with a prefetch buffer corrects the ledger here
            #    ★The protocol is **optional**: a sink without settle (TypingSink /
            #      console_sink) behaves identically down to the byte -- for those, printing
            #      to the terminal really is the moment the sound happens.
            settle = getattr(self.sink, "settle", None)
            if settle is not None:
                try:
                    # ★The protocol has no `interrupted` parameter: with barge-in abolished
                    #   (2026-09-27) and invalidation deleted (same day), **no situation asks
                    #   the sink to stop early** -- a turn that has begun always finishes.
                    #   Adding that parameter back reinstates both mechanisms (assertions
                    #   20g/20j).
                    spoken_total = await settle()
                    ledger.reconcile(spoken_total)
                except Exception as e:
                    # A failed settle must not be swallowed: the ledger is wrong, but part of
                    # the reply was spoken and this turn still has to be recorded. Leave an
                    # obvious trace rather than letting it look like a normal record.
                    print(f"[voice] ⚠ sink.settle failed, this turn's record may be inaccurate: "
                          f"{type(e).__name__}: {e}", flush=True)

            # -- rules 2 and 3: clear R's unspoken part
            dropped = ledger.drop_unspoken()

            # ★★Record unconditionally (since invalidation was deleted on 2026-09-27). There
            #   used to be two branches here: the invalidated one wrote no record and pushed
            #   user_text back into the reservoir to re-run. **Both are gone** -- every turn
            #   finishes, every turn is recorded.
            self._persist(user_text, ledger, reason, dropped)
            self.disp.close_session()

            # ★Only delete .inflight once the record really is on disk. Crash anywhere before
            #   this and .inflight stays in this turn's session directory = the crash scene,
            #   and corpus material for the fixture harness.
            #   "Archiving" a crash is, in implementation, doing nothing at all.
            self.reservoir.release()

            rec = {
                "reason": reason,
                "cuts_consumed": n_cuts,
                "user_text": user_text,
                "spoken": ledger.spoken_text(),
                "dropped_chars": dropped,
                "first_chunk_ms": first_chunk_ms,
                "total_ms": int((time.time() - t0) * 1000),
                "generation_id": gen_id,
                "usage": usage,
            }
            self.turns.append(rec)

            if self.on_turn_end is not None:
                try:
                    post = self.on_turn_end(rec)
                    # ★Await the hook's return value when it is awaitable (2026-09-27).
                    #   `chain._after_turn` became a coroutine -- it throws the blocking
                    #   compression into `asyncio.to_thread`, because otherwise those few
                    #   seconds of compression block the event loop (nothing consumes mic
                    #   frames, and the STT receive side stops turning).
                    #   ★Omitting this line leaves `rec["post"]` holding a coroutine that was
                    #     **never awaited**: the compression simply does not run, and Python
                    #     only prints a RuntimeWarning at exit.
                    if inspect.isawaitable(post):
                        post = await post
                    rec["post"] = post
                except Exception as e:
                    # Words already spoken cannot be taken back, so a failure in post-turn
                    # processing is reported, not raised.
                    print(f"[voice] ⚠ on_turn_end failed (this turn is unaffected): {type(e).__name__}: {e}")
                    rec["post_error"] = f"{type(e).__name__}: {e}"
            return rec
        finally:
            self.busy = False

    async def run_until_quiet(self, idle_s=0.05, max_turns=20):
        """Pump repeatedly until the reservoir is empty.

        ★This is not a polling detector (the trigger source is push-based). It is only the
          driving shell for when there is no real STT, and a testable entrance to the
          semantics "water arriving while the LLM is busy gets consumed in one go on the next
          turn". With real STT, pump hangs off the push callback and this function is not
          needed.
        """
        out = []
        for _ in range(max_turns):
            rec = await self.pump()
            if rec is None:
                if len(self.reservoir) == 0:
                    break
                await asyncio.sleep(idle_s)
                continue
            out.append(rec)
        return out
