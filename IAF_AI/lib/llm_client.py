"""
Shared LLM HTTP client.
Pure infrastructure: HTTP calls + retry + error classification.
No agent behavior, no context management, no tool logic.
All agents import this; bug fixes here benefit everyone.
"""

import json
import requests
import time


class LLMError(Exception):
    pass


class ContextTooLongError(LLMError):
    pass


RETRYABLE_STATUS = {429, 500, 502, 503, 529}


def _is_responses_api(url):
    """Detect if the URL points to a Responses API endpoint (e.g. xAI reasoning models)."""
    return url.rstrip("/").endswith("/responses")


def _is_deepseek(url):
    """Detect a direct DeepSeek endpoint. DeepSeek uses its native reasoning format
    (a thinking switch + a top-level reasoning_effort) rather than OpenRouter's reasoning
    object, and it does not accept OpenRouter's provider-routing field."""
    return "api.deepseek.com" in (url or "")


def _is_anthropic_api(url):
    """Detect an Anthropic Messages API endpoint. Its request/response shapes are
    completely different from Chat Completions: system is hoisted to the top level,
    max_tokens is mandatory, the response is an array of content blocks, and the stop
    reason is called stop_reason. This client normalises both into the Chat Completions
    shape."""
    return "api.anthropic.com" in (url or "")


def _build_provider(provider_routing):
    """Translate a config `provider_routing` value into an OpenRouter `provider` object.

    - falsy (None / "" / {})  -> None: do not send a `provider` field; use OpenRouter's
                                 default routing.
    - dict                    -> passed through verbatim (advanced users can write a full
                                 OpenRouter provider object).
    - str "slug" or "slug/quant" -> pin to that provider (allow_fallbacks=False, with an
      optional quantisation restriction). Pinning means the provider cannot drift within a
      session, which is what lets cached input (prefix caching) keep hitting.

    Examples:
      "novita"      → {"order": ["novita"], "allow_fallbacks": False}
      "novita/fp8"  → {"order": ["novita"], "quantizations": ["fp8"], "allow_fallbacks": False}
    """
    if not provider_routing:
        return None
    if isinstance(provider_routing, dict):
        return provider_routing
    if isinstance(provider_routing, str):
        parts = [p.strip() for p in provider_routing.split("/") if p.strip()]
        if not parts:
            return None
        prov = {"order": [parts[0]], "allow_fallbacks": False}
        if len(parts) > 1:
            prov["quantizations"] = parts[1:]
        return prov
    return None


def call_llm(url, key, model, messages, tools=None, tool_choice=None,
             temperature=None, top_p=None, reasoning=None,
             max_tokens=None, max_retries=3, timeout=120,
             provider_routing=None, extra_params=None):
    """
    Send messages to LLM API. Returns the message dict from the response.
    Handles retry for transient errors, raises LLMError for fatal errors.
    Auto-detects Responses API (xAI reasoning) vs Chat Completions API by URL.
    Returned dict includes '_meta' with usage data and finish_reason when available.

    extra_params: optional dict merged verbatim into the request body -- an arbitrary
    pass-through (sampling / routing / provider-specific knobs). It is merged last, so
    config overrides anything this function inferred. It is the escape hatch for sending a
    new field without editing this file.
    """
    use_responses = _is_responses_api(url)
    use_anthropic = _is_anthropic_api(url)

    if use_anthropic:
        # Anthropic Messages API: system is hoisted to the top level, and OpenAI's tool
        # message shapes have to be translated
        system_text = ""
        api_messages = []
        for m in messages:
            role = m.get("role", "")
            if role == "system":
                system_text = m.get("content", "")
            elif role == "assistant" and m.get("tool_calls"):
                # OpenAI {"role":"assistant","tool_calls":[...]}
                #   → Anthropic {"role":"assistant","content":[text, tool_use...]}
                blocks = []
                text = m.get("content")
                if text:
                    blocks.append({"type": "text", "text": text})
                for tc in m["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    blocks.append({"type": "tool_use", "id": tc.get("id", ""),
                                   "name": func.get("name", ""), "input": args})
                api_messages.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                # OpenAI {"role":"tool",...} -> Anthropic's tool_result rides a user turn
                api_messages.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": m.get("content", ""),
                }]})
            else:
                # content may be a string, or already an array of blocks
                # (multimodal image/document)
                api_messages.append({"role": role, "content": m.get("content", "")})
        payload = {
            "model": model,
            "max_tokens": int(max_tokens) if max_tokens is not None else 4096,
            "messages": api_messages,
        }
        if system_text:
            payload["system"] = system_text
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if tools:
            payload["tools"] = [{
                "name": x.get("function", {}).get("name", ""),
                "description": x.get("function", {}).get("description", ""),
                "input_schema": x.get("function", {}).get("parameters", {}),
            } for x in tools]
            if tool_choice:
                payload["tool_choice"] = tool_choice
    elif use_responses:
        # Responses API format (xAI reasoning models)
        payload = {"model": model, "input": messages}
    else:
        # Standard Chat Completions format
        payload = {"model": model, "messages": messages}
        # provider_routing is an OpenRouter-only field; the official DeepSeek API does not
        # accept it, so even if provider_routing is left in the config it is never sent on
        # a direct DeepSeek connection -- otherwise: 400.
        if not _is_deepseek(url):
            _prov = _build_provider(provider_routing)
            if _prov:
                payload["provider"] = _prov
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        # reasoning: DeepSeek's native format vs OpenRouter's reasoning object
        # (branching on the endpoint)
        if reasoning is not None and isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if _is_deepseek(url):
                # Official DeepSeek API: a thinking switch + top-level reasoning_effort
                payload["thinking"] = {"type": "enabled"}
                if effort is not None:
                    payload["reasoning_effort"] = effort
            else:
                # OpenRouter reasoning object (effort and max_tokens are mutually exclusive)
                reasoning_payload = {}
                if effort is not None:
                    reasoning_payload["effort"] = effort
                if effort is None and reasoning.get("max_tokens") is not None:
                    reasoning_payload["max_tokens"] = reasoning["max_tokens"]
                if "exclude" in reasoning:
                    reasoning_payload["exclude"] = reasoning["exclude"]
                if reasoning_payload:
                    payload["reasoning"] = reasoning_payload

    # Merged last, so the caller's extra_params override anything inferred above from the
    # endpoint
    if extra_params:
        payload.update(extra_params)

    if use_anthropic:
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    else:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                    if use_anthropic:
                        # Anthropic content blocks -> normalised to the Chat Completions shape
                        text_parts, tool_calls = [], []
                        for blk in data.get("content", []):
                            if blk.get("type") == "text":
                                text_parts.append(blk.get("text", ""))
                            elif blk.get("type") == "tool_use":
                                tool_calls.append({
                                    "id": blk.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": blk.get("name", ""),
                                        "arguments": json.dumps(blk.get("input", {}),
                                                                ensure_ascii=False),
                                    },
                                })
                        msg = {"role": "assistant",
                               "content": "\n".join(text_parts) if text_parts else None}
                        if tool_calls:
                            msg["tool_calls"] = tool_calls
                        # Anthropic calls the stop reason stop_reason, and it lives at the
                        # top level of the response
                        finish = data.get("stop_reason")
                    elif use_responses:
                        # Responses API: extract output_text
                        output_text = data.get("output_text", "")
                        msg = {"role": "assistant", "content": output_text}
                        finish = data.get("status")
                    else:
                        choice = data["choices"][0]
                        msg = choice["message"]
                        finish = choice.get("finish_reason") or choice.get("native_finish_reason")
                    # finish_reason lives on the choice, not on the message. Without
                    # carrying it out, the caller cannot distinguish "finished normally"
                    # from "hit max_tokens and got truncated" -- the content looks
                    # complete either way.
                    msg["_meta"] = {"usage": data.get("usage", {}), "finish_reason": finish}
                    # ★Silent-failure gate: in thinking mode the body may land entirely in
                    #   reasoning_content with content left empty. A caller that only reads
                    #   content gets an empty string, while the money has been spent and
                    #   the work has been done. Measured: the same model with the same
                    #   prompt sometimes fills content and sometimes fills reasoning, so
                    #   this is unstable behaviour and has to be shouted about at the
                    #   client layer.
                    if not (msg.get("content") or "").strip() and \
                            (msg.get("reasoning_content") or "").strip():
                        print(f"  [WARN] content is empty while reasoning_content has "
                              f"{len(msg['reasoning_content'])} characters -- the body may "
                              f"have landed entirely in reasoning_content "
                              f"(finish_reason={finish})")
                    return msg
                except (KeyError, requests.exceptions.JSONDecodeError):
                    print(f"  [Retry] Invalid JSON in 200 response, attempt {attempt+1}/{max_retries}")
                    last_error = "Invalid JSON response"
                    time.sleep(2 ** (attempt + 1))
                    continue

            if response.status_code not in RETRYABLE_STATUS:
                error_body = response.text[:500]
                if response.status_code == 400 and "context" in error_body.lower():
                    raise ContextTooLongError(error_body)
                raise LLMError(f"HTTP {response.status_code}: {error_body}")

            retry_after = response.headers.get("Retry-After")
            wait = int(retry_after) if retry_after else 2 ** (attempt + 1)
            print(f"  [Retry] {response.status_code}, attempt {attempt+1}/{max_retries}, wait {wait}s")
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            time.sleep(wait)

        except requests.exceptions.Timeout:
            print(f"  [Retry] Timeout, attempt {attempt+1}/{max_retries}")
            last_error = "Timeout"
            time.sleep(2 ** (attempt + 1))

        except requests.exceptions.ConnectionError:
            print(f"  [Retry] Connection error, attempt {attempt+1}/{max_retries}")
            last_error = "Connection error"
            time.sleep(2 ** (attempt + 1))

        except requests.exceptions.ChunkedEncodingError:
            print(f"  [Retry] Chunked encoding error, attempt {attempt+1}/{max_retries}")
            last_error = "ChunkedEncodingError"
            time.sleep(2 ** (attempt + 1))

        except (LLMError, ContextTooLongError):
            raise

    raise LLMError(f"Failed after {max_retries} retries. Last: {last_error}")


# ═══════════════════════════════════════════════════════════════════════════
# Streaming (stream_llm) -- for the speaker dispatch
#
# Design constraints:
#   ★`import httpx` MUST be inside the function and must not sit at module level.
#     Reason: this file is imported by the compressors' dispatch_base.py as
#     `from llm_client import call_llm`. If httpx is missing and the import is at the top
#     level, loading the whole module fails -> get_llm_caller() returns None ->
#     dispatch.py raises RuntimeError("No LLM caller available") -> **the compressors stop
#     too**. The compressors only use requests and should not be taken down with httpx.
#     The speaker has no fallback: if it cannot get httpx it should fail, and that is right.
#   ★Do not merge the import into `from llm_client import call_llm, stream_llm`.
#
# Three layers (a seam for swapping providers):
#   _iter_sse_lines(response)  line protocol: skip comment lines, recognise [DONE].
#                              Barely changes when swapping providers.
#   _map_chunk(chunk)          field mapping: **different for every provider; swapping
#                              providers means editing only this**.
#   stream_llm(...)            accumulate + emit envelopes. The envelope shape never
#                              changes, so consumers never learn a second one.
# ═══════════════════════════════════════════════════════════════════════════


async def _iter_sse_lines(response, stats=None):
    """Split an httpx streaming response into an iteration of payloads. Yields:

        ("data", <dict>)   one JSON chunk
        ("done", None)     received `data: [DONE]`

    Three kinds of line have to be told apart (checked word by word against the API docs):
      `: OPENROUTER PROCESSING` -- an SSE **comment line** (starts with `:`). A heartbeat
            while queuing or selecting a provider. The docs say it "can be safely ignored
            per the SSE specs".
            ★It must never be fed to json.loads, or queuing will crash the stream.
      `data: {…}`              -- a normal delta.
      `data: [DONE]`           -- the terminator, **not JSON**. Compare before parsing.

    stats: a mutable dict supplied by the caller, used to carry out "how many comment lines
    and how many bad chunks were skipped". Comment lines only appear while queuing, so this
    count is the **only** evidence that that path was ever exercised.
    """
    import json as _json
    if stats is None:
        stats = {}
    async for raw in response.aiter_lines():
        line = raw.strip()
        if not line:
            continue                      # blank line separating SSE events
        if line.startswith(":"):
            stats["comments"] = stats.get("comments", 0) + 1
            stats.setdefault("comment_samples", [])
            if len(stats["comment_samples"]) < 3:
                stats["comment_samples"].append(line[:60])
            continue                      # ★comment line / heartbeat, skip
        if not line.startswith("data:"):
            stats["other_fields"] = stats.get("other_fields", 0) + 1
            continue                      # id: / event: / retry: etc, unused here
        payload = line[5:].strip()
        if payload == "[DONE]":
            yield ("done", None)
            return
        try:
            yield ("data", _json.loads(payload))
        except ValueError:
            # One bad chunk should not kill the whole stream (in a voice setting, the
            # part already spoken is a real cost).
            stats["bad_chunks"] = stats.get("bad_chunks", 0) + 1
            print(f"  [stream] skipping unparseable chunk: {payload[:120]}")
            continue


def _map_chunk(chunk):
    """Map one provider chunk into a uniform intermediate shape.
    **Swapping providers means editing only this function.**

    Returns {"text":str, "reasoning":str, "finish":str|None, "usage":dict|None, "id":str|None}

    ★`id` is OpenRouter's generation id. It is **the only handle for looking up what a call
      actually cost after it was interrupted** -- being interrupted means the usage chunk
      never arrives, so the only way is `GET /api/v1/generation?id=<id>` after the fact.
      Interrupting is a core action in a voice setting, so this field must be carried out.

    ★Where OpenRouter puts its fields (confirmed verbatim; different from a direct DeepSeek
      connection, do not mix them up):
        text delta       choices[0].delta.content
        reasoning delta  choices[0].delta.reasoning_details   <- not delta.reasoning
                                                              <- and definitely not
                                                                 DeepSeek's delta.reasoning_content
        stop reason      choices[0].finish_reason / native_finish_reason
        usage            top-level usage (in the last chunk before [DONE]; it arrives
                         automatically)

    ★`choices` may be an empty list -- that is exactly what a usage chunk looks like. So
      never write `chunk["choices"][0]` unconditionally: it raises IndexError, or (when
      swallowed by a try) throws away both cost and cached_tokens -- and cached_tokens is
      the only signal that verifies whether pinning the provider actually took effect.
    """
    out = {"text": "", "reasoning": "", "finish": None, "usage": None,
           "id": chunk.get("id")}

    usage = chunk.get("usage")
    if usage:
        out["usage"] = usage

    choices = chunk.get("choices") or []
    if not choices:
        return out                         # ★this is the exit taken by a usage chunk

    ch = choices[0] or {}
    delta = ch.get("delta") or {}

    txt = delta.get("content")
    if txt:
        out["text"] = txt

    # reasoning_details is a list whose elements look like
    # {"type":"reasoning.text","text":"..."}
    rd = delta.get("reasoning_details")
    if rd:
        parts = []
        for item in rd if isinstance(rd, list) else [rd]:
            if isinstance(item, dict):
                parts.append(item.get("text") or item.get("summary") or "")
            elif isinstance(item, str):
                parts.append(item)
        out["reasoning"] = "".join(parts)

    out["finish"] = ch.get("finish_reason") or ch.get("native_finish_reason")
    return out


async def stream_llm(url, key, model, messages,
                     temperature=None, top_p=None, reasoning=None,
                     max_tokens=None, timeout=120, provider_routing=None):
    """Streaming call, as an async generator. Yields a uniform envelope at a time:

        {"type": "delta", "text": "..."}                        text delta
        {"type": "done",  "message": {...}, "usage": {...},
                          "finish_reason": "...", "reasoning": "..."}

    `done.message` is **deliberately shaped like call_llm's return value**:
        {"role": "assistant", "content": "...", "_meta": {"usage": {...}}}
    so that dispatch_base._extract_text() and the history-persisting code keep working as
    they are, and if the speaker ever grows tools, _parse_llm_response does not need to
    fork either. That compatibility is free.

    ★There is no stream switch. This function IS the streaming one; for non-streaming,
      call call_llm.
    ★Do not send stream_options / usage.include -- the docs say word for word that both are
      "deprecated and have no effect". usage arrives automatically.
    ★A turn can legitimately have **zero text deltas** (only finish_reason + usage). So the
      termination condition is end-of-stream / [DONE], never "we received a delta".
    Cancellation: the caller exiting the `async for` (or closing the enclosing
      `async with`) breaks the stream. The provider is on OpenRouter's cancellation-support
      list, so breaking the stream stops generation and stops billing.
    """
    import httpx      # ★function-level import; see the comment at the top of this section

    payload = {"model": model, "messages": messages, "stream": True}
    if not _is_deepseek(url):
        prov = _build_provider(provider_routing)
        if prov:
            payload["provider"] = prov
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if reasoning is not None and isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if _is_deepseek(url):
            payload["thinking"] = {"type": "enabled"}
            if effort is not None:
                payload["reasoning_effort"] = effort
        else:
            rp = {}
            if effort is not None:
                rp["effort"] = effort
            if effort is None and reasoning.get("max_tokens") is not None:
                rp["max_tokens"] = reasoning["max_tokens"]
            if "exclude" in reasoning:
                rp["exclude"] = reasoning["exclude"]
            if rp:
                payload["reasoning"] = rp

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    text_parts = []
    reasoning_parts = []
    usage = {}
    finish_reason = None
    gen_id = None
    stats = {}

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread())[:500]
                if resp.status_code == 400 and b"context" in body.lower():
                    raise ContextTooLongError(body.decode("utf-8", "replace"))
                raise LLMError(f"HTTP {resp.status_code}: {body.decode('utf-8', 'replace')}")

            async for kind, chunk in _iter_sse_lines(resp, stats):
                if kind == "done":
                    break
                m = _map_chunk(chunk)
                if m["id"] and gen_id is None:
                    gen_id = m["id"]
                    # ★Emit the id on the very first chunk: an interrupt happens before
                    #   done, so the consumer has to have the handle before then, or what
                    #   the call really cost cannot be looked up.
                    yield {"type": "meta", "generation_id": gen_id}
                if m["usage"]:
                    usage = m["usage"]
                if m["finish"]:
                    finish_reason = m["finish"]
                if m["reasoning"]:
                    reasoning_parts.append(m["reasoning"])
                if m["text"]:
                    text_parts.append(m["text"])
                    yield {"type": "delta", "text": m["text"]}

    content = "".join(text_parts)
    yield {
        "type": "done",
        "message": {"role": "assistant", "content": content,
                    "_meta": {"usage": usage}},
        "usage": usage,
        "finish_reason": finish_reason,
        "reasoning": "".join(reasoning_parts),
        "generation_id": gen_id,
        "stream_stats": stats,      # {comments, comment_samples, other_fields, bad_chunks}
    }


async def text_only(stream):
    """Pass through only the text of the deltas. For consumers that just want characters
    (feeding a sentence buffer, for instance).

    ★Note that it swallows meta and done -- so a consumer that needs usage or
      generation_id cannot use this wrapper and must read the envelopes itself.
    """
    async for ev in stream:
        if ev.get("type") == "delta":
            yield ev["text"]
