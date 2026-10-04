"""
lib/jev_client.py -- JEV client (structured judgement, zero generation)

Same layer as llm_client.py / stt_client.py: a pure client, with no agent behaviour in it.
The question it asks lives in prompt_base/listener/question.md and the connection
parameters live in config_base/listener/dispatch_config.json -- this file is only
responsible for "how to send it and how to read the reply".

★★Four implementation disciplines (each one checked before being written; writing from
intuition gets every one of them wrong)

1. **JEV is not a chat model and cannot go through llm_client.call_llm.**
   It speaks the Decisions API: `POST https://openrouter.ai/api/alpha/decisions`
   (or the TypeSafe-compatible shape `/api/v1/systemone`), request body
   `{state, model, questions}`.
   Hitting `/chat/completions` returns **400**, and it does not appear in
   `/api/v1/models` either.
   -> The reasoning is recorded in `_api_note` in
   config_base/listener/dispatch_config.json.

2. **A noul answer has no separate confidence field.** The reply is
   `{"type":"noul","noul":0.0..1.0}`, and that noul IS the calibrated probability;
   `confidence` only exists on choice/score answers.
   -> So what this returns is `noul`, not `confidence`. **No confidence alias** -- two
   keys holding the same number means somewhere, someday, the wrong one gets read.

3. **★If noul cannot be read, always raise -- never degrade to False.**
   If the server renames a field, moves the endpoint, or rate-limits us, swallowing that
   as `meaningful=False` presents as "the person finished and the machine never speaks",
   **with no error and no way to find the cause**. This is the same discipline as
   stt_client's "field names come from real events", which is why `raw` is carried back
   verbatim: when something goes wrong, the server's own text is in the error message.
   The caller (the ticker) should treat the exception as "no verdict this tick" and skip,
   **not** as "judged incomplete".

4. **The connection must be pre-warmed.** Measured, the first request takes ~700ms (TLS
   handshake) and subsequent ones ~300ms -- without pre-warming, the first utterance is
   guaranteed to blow the budget. Hence this is a class holding an httpx.AsyncClient
   (connection-pool reuse) rather than building one per call, and `warmup()` is exposed
   separately.

★Why async: the design calls for "a new tick cancels the previous tick". In asyncio that
  is `task.cancel()`, which requires the whole chain to be cancellable -- httpx's async
  requests are; a synchronous requests call is not.
"""

import json
import os
import re
import time

import httpx

DEFAULT_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"
DEFAULT_TIMEOUT_S = 5.0
QUESTION_KEY = "complete"          # the key of that question inside question.md
DEFAULT_THRESHOLD = 0.5


class JevError(Exception):
    """Protocol / HTTP / reply-shape problem. ★Never swallowed internally into
    meaningful=False (discipline 3)."""


# ---------------------------------------------------------------- pure functions (offline-testable)

def extract_question_block(md_text):
    """Pull the single ```json block out of question.md.

    ★Nothing outside the block ever reaches the model -- that part is the reasoning, for
      humans. Embedding a JSON block inside the authoritative document, rather than having
      a separate .json file, is deliberate: the wording IS the policy (a measured
      conclusion), and a policy has to live next to its justification, or someone changes
      the wording and nobody knows why it was written the way it was.
    """
    blocks = re.findall(r"```json\s*\n(.*?)\n```", md_text, re.S)
    if not blocks:
        raise JevError("no ```json block found in question.md")
    if len(blocks) > 1:
        # More than one and we report it rather than guessing which is authoritative --
        # guessing wrong means silently using the wrong question.
        raise JevError(f"question.md has {len(blocks)} ```json blocks; there must be exactly one")
    try:
        questions = json.loads(blocks[0])
    except json.JSONDecodeError as e:
        raise JevError(f"failed to parse the json block in question.md: {e}") from e
    if not isinstance(questions, dict) or not questions:
        raise JevError("the json block in question.md must be a non-empty object (keys = question names)")
    for key, q in questions.items():
        if not isinstance(q, dict) or "type" not in q:
            raise JevError(f"question {key!r} is missing the type field")
    return questions


def should_skip(state):
    """Skip the tick without making a call when U is empty (zero cost while silent).

    ★It also blocks the two kinds of rubbish that measurably do arrive: empty
      partials/finals, and stray punctuation forming its own segment -- seriously judging
      a lone `.` costs real money.
    """
    text = (state or "").strip()
    if not text:
        return "empty_state"
    # Nothing left after removing all punctuation and whitespace = punctuation only, with
    # no semantics to judge
    if not re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE):
        return "punctuation_only"
    return None


def verdict_from_answer(payload, question_key=QUESTION_KEY, threshold=DEFAULT_THRESHOLD):
    """Take noul out of the reply and compare it against the threshold.
    Raises if it cannot be read (discipline 3)."""
    if not isinstance(payload, dict):
        raise JevError(f"reply is not an object: {payload!r}")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise JevError(f"no answers object in the reply: {json.dumps(payload, ensure_ascii=False)[:400]}")
    ans = answers.get(question_key)
    if not isinstance(ans, dict):
        raise JevError(
            f"{question_key!r} is not in the reply's answers (keys present: {list(answers)})"
        )
    noul = ans.get("noul")
    if not isinstance(noul, (int, float)) or isinstance(noul, bool):
        raise JevError(
            f"cannot read a numeric noul out of answers[{question_key!r}]: {json.dumps(ans, ensure_ascii=False)[:400]}"
        )
    noul = float(noul)
    return noul > float(threshold), noul


# ---------------------------------------------------------------- config / key

def resolve_key(shell_root, provider="openrouter"):
    """Fetch `providers.<provider>.api_key` from `<shell root>/IAF_AI/config.json`.

    ★The argument is the **shell root**, the same convention as
      `stt_client.resolve_key`. Before they were unified, this one took the IAF_AI
      directory and that one took the shell root -- one parameter name with two meanings,
      and the consequence of passing the wrong one was "cannot read .../config.json",
      an error pointing at a path that looks entirely plausible.

    ★Same convention as stt_client.resolve_key: **the client does not go hunting for
      config on its own**, and a key never goes into dispatch_config.json -- that file is
      pushed down through the mirror rule and the key would travel with it.
    """
    path = os.path.join(shell_root, "IAF_AI", "config.json")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    key = (cfg.get("providers", {}).get(provider) or {}).get("api_key")
    if not key:
        raise JevError(f"no providers.{provider}.api_key in {path}")
    return key


def load_dispatch_config(dispatch_dir):
    with open(os.path.join(dispatch_dir, "dispatch_config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_questions(dispatch_dir, agent_conf):
    """Read the question from the first entry of agent_conf.context_files
    (path relative to dispatch_dir)."""
    files = agent_conf.get("context_files") or []
    if not files:
        raise JevError("agents.<name>.context_files is empty; there is no question to read")
    path = files[0]
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(dispatch_dir, path))
    if not os.path.isfile(path):
        # Always report when it cannot be read. The engine's own context_files handling
        # silently returns an empty string in this case, which here would present as
        # "judging against an empty question" -> it has to fail explicitly.
        raise JevError(f"cannot read the question file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return extract_question_block(f.read())


# ---------------------------------------------------------------- client

class JevClient:
    """A JEV client holding one reusable connection.

        async with JevClient(dispatch_dir, "jev1") as jev:
            await jev.warmup()
            r = await jev.ask("do you know what Kubernetes is for?")
    """

    def __init__(self, dispatch_dir, agent_name="jev1", shell_root=None,
                 api_key=None, questions=None, threshold=None, question_key=QUESTION_KEY):
        self.dispatch_dir = dispatch_dir
        self.agent_name = agent_name
        cfg = load_dispatch_config(dispatch_dir)
        agents = cfg.get("agents") or {}
        if agent_name not in agents:
            raise JevError(f"{agent_name!r} is not in agents in dispatch_config.json (present: {list(agents)})")
        self.agent_conf = agents[agent_name]
        self.knobs = cfg.get("knobs") or {}

        self.endpoint = self.agent_conf.get("endpoint") or DEFAULT_ENDPOINT
        self.model = self.agent_conf.get("model") or DEFAULT_MODEL
        self.timeout_s = float(self.agent_conf.get("timeout_s") or DEFAULT_TIMEOUT_S)
        self.question_key = question_key
        self.questions = questions if questions is not None else load_questions(dispatch_dir, self.agent_conf)
        if threshold is None:
            threshold = self.knobs.get("meaningful_threshold")
        # ★The threshold must not default to None: it is a number that gets compared
        #   (the reason is also written into the config).
        self.threshold = DEFAULT_THRESHOLD if threshold is None else float(threshold)

        if api_key is None:
            if shell_root is None:
                # dispatch_dir = <shell root>/IAF_AI/dispatch/<name> -> up three levels
                shell_root = os.path.normpath(os.path.join(dispatch_dir, "..", "..", ".."))
            api_key = resolve_key(shell_root, self.agent_conf.get("provider") or "openrouter")
        self._key = api_key
        self._client = None
        self.calls = 0
        self.total_input_tokens = 0
        self.total_cost_usd = 0.0

    # -- lifecycle -------------------------------------------------

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self

    async def __aexit__(self, *exc):
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        return False

    async def warmup(self):
        """Pre-warm TLS and the connection pool (discipline 4). Sends one real request with
        a minimal state and returns the elapsed milliseconds.

        ★Opening a TCP connection without sending a request is not enough -- that measured
          ~700ms is the figure for the first **complete request**, which includes server
          cold start on top of the handshake. So the warm-up is a real call, costing
          roughly $0.00001.
        """
        t0 = time.monotonic()
        # ★The literal below is sent to the API verbatim. Keep it short, keep it a
        #   complete utterance, and keep it in the language this deployment runs in.
        await self.ask("do you know what Kubernetes is for?", _warmup=True)
        return (time.monotonic() - t0) * 1000.0

    # -- main entry point ------------------------------------------

    async def ask(self, state, *, timeout=None, _warmup=False):
        """Ask JEV: is the content in `state` (= the text of U) complete?

        Returns a dict:
            meaningful  bool        noul > threshold
            noul        float|None  calibrated probability (None when skipped)
            threshold   float
            skipped     str|None    "empty_state" / "punctuation_only" = no request sent
            latency_ms  float|None
            usage       dict
            raw         dict|None   the server's own text (look here on a field rename, discipline 3)

        ★Cancellation semantics: this is an ordinary coroutine, so an outer
          `task.cancel()` is enough to kill the in-flight tick ("a new tick cancels the
          previous tick"). Killing it leaves no half-state behind -- this method changes
          nothing except the counters.
        """
        skip = should_skip(state)
        if skip:
            return {"meaningful": False, "noul": None, "threshold": self.threshold,
                    "skipped": skip, "latency_ms": None, "usage": {}, "raw": None,
                    "state": state}

        body = {"model": self.model, "state": state, "questions": self.questions}
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        client = self._client
        owned = client is None
        if owned:
            client = httpx.AsyncClient(timeout=timeout or self.timeout_s)
        t0 = time.monotonic()
        try:
            resp = await client.post(self.endpoint, json=body, headers=headers,
                                     timeout=timeout or self.timeout_s)
        except httpx.HTTPError as e:
            raise JevError(f"JEV request failed ({self.endpoint}): {type(e).__name__}: {e}") from e
        finally:
            if owned:
                await client.aclose()
        latency_ms = (time.monotonic() - t0) * 1000.0

        if resp.status_code != 200:
            raise JevError(
                f"JEV HTTP {resp.status_code} ({self.endpoint}): {resp.text[:600]}"
            )
        try:
            payload = resp.json()
        except ValueError as e:
            raise JevError(f"JEV reply is not JSON: {resp.text[:400]}") from e

        meaningful, noul = verdict_from_answer(payload, self.question_key, self.threshold)

        usage = payload.get("usage") or {}
        if not _warmup:
            self.calls += 1
            self.total_input_tokens += int(usage.get("input_tokens") or 0)
            self.total_cost_usd += float(usage.get("cost") or 0.0)

        return {"meaningful": meaningful, "noul": noul, "threshold": self.threshold,
                "skipped": None, "latency_ms": latency_ms, "usage": usage,
                "raw": payload, "state": state}
