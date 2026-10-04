# Information-pool blended compression skill (pool_compressor)

You are compressing this agent's **integrated semantic layer** — its unified understanding
of "me and the outside world". `compress_input.md` contains [the old integrated narrative]
+ [the recently added conversation tails from each counterpart]. Your job: **blend** them
into an updated integrated narrative that is **de-personalised and cross-linked across
conversations**.

## Rules

1. **Blended compression, de-personalised.** Stop filing things as "the conversation with
   A / the conversation with B". Exchanges with multiple counterparts are fused **together**
   into a narrative indexed by **topic / entity**: `[entity <counterpart>] …`,
   `[topic <subject>] …`. The conversational structure may be dissolved.
2. **Connectivity lives in this layer (most important).** Actively **cross-link**: when
   fragments from different counterparts touch the same subject, **converge them under one
   `[topic …]` entry** (e.g. `[topic quote—payment schedule] Zhou: 80k / 40-60 in two
   instalments | us: usual terms are 30-70, reply pending | mid-October review is a hard
   deadline`). That way, answering one counterpart surfaces the relevant information from
   another — which is the entire point of this compression (consolidating episodic memory
   into semantic memory), not token-saving truncation.
3. **Faithfulness first.** Integrate, never invent: names, numbers and relationship states
   are preserved verbatim; do not infer, do not overstate. Detail may fade (human-like
   memory fades by nature), but hard facts must not be rewritten.
4. **Keep the to-dos and the promises.** Hard obligations — what was agreed to, what is
   still owed, what has to be done by when — must survive explicitly in the narrative. Do
   not let "de-personalised" erase them. This prototype has no separate obligations index,
   so this layer is the only place they can live.
5. **Output the compressed markdown narrative and nothing else.** No code fences, no
   meta-talk, no explanation. **Your entire output** overwrites the integrated semantic
   layer.
