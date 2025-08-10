# strategies/self_consistency.py
def majority_vote(texts):
    # trivial: choose the most common normalized JSON object
    from collections import Counter
    norm = []
    for t in texts:
        norm.append(t.strip())
    c = Counter(norm)
    return c.most_common(1)[0][0]
