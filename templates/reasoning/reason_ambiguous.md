Write prompts where the instruction is deliberately ambiguous in wording, but still solvable.

Required output: ONLY JSON with keys name (string) and age (number).

Example of ambiguity:
- "Give me the object about a person" (without explicitly saying 'JSON'), but later clarify in parentheses '(format: JSON with name, age)'.
- Conflicting hints about age format ('age in years' vs 'numeric').

The goal is to see if the model focuses on the clarification and ignores vagueness.
