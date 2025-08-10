You are a dataset author creating adversarial prompts to test JSON structural robustness.
Goal: produce prompts that *force* a model to output ONLY valid JSON matching a given schema,
while the prompt itself contains distractions that commonly trigger malformed JSON or extra prose.

Design principles:
- Include subtle traps: conflicting “explain” instructions, mid-prompt quotes, stray braces, and inline examples.
- Vary wording and surface form so these aren’t template clones.
- Keep the *task instruction* crystal clear: output ONLY JSON, no code fences, no explanations.

Target schema (use in your wording, but don’t show as code): object with keys:
- name: string
- age: number

Generate diverse prompts with:
- Minor contradictions (e.g., “explain” vs “only JSON”).
- Punctuation bait: quotes, brackets, commas near the instruction.
- Foreign words occasionally to ensure robustness (don’t require multilingual responses).

Each case must be independent and self-sufficient.
