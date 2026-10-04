# Single-thread compression skill (thread_compressor)

You are compressing the conversation history between this agent and **one specific
counterpart**. `compress_input.md` contains the **older part** of that conversation (the
most recent exchanges have been left verbatim in place by the system and are not given to
you). Your job: compress this older stretch into a **faithful, compact** summary.

## Rules

1. **Conventional compression, keep the conversational structure.** This is an exchange
   with one particular counterpart, and after compression it must still be possible to see
   "who said what, what was agreed, what is still owed". Do not de-personalise it, do not
   dissolve it into narrative — that is pool_compressor's job.
2. **Faithfulness first.** Compress only, never alter: add nothing, infer nothing,
   overstate nothing. Hard facts — names, numbers, promises, dates — are preserved
   verbatim.
3. **Control the size.** Target ≤ 7000 characters. Keep first: outstanding obligations and
   promises, key decisions, what the counterpart wants and the state of the relationship,
   and any hard fact (numbers, dates, arrangements). Droppable: pleasantries, repetition,
   superseded intermediate steps.
4. **This is spoken conversation.** The input is speech transcription, so it carries verbal
   tics, repetitions, self-corrections and transcription errors. Strip that noise and keep
   only what the speaker actually meant — **but never rewrite an uncertain statement into a
   certain one** ("around eighty thousand" must not become "eighty thousand").
5. **Output the compressed markdown body and nothing else.** No code fences, no "here is
   the summary" meta-talk, no explanation of what you did. **Your entire output** is
   written straight to disk as the replacement for that older stretch of conversation.
