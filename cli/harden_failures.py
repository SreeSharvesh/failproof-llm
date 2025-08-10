from __future__ import annotations
import argparse, asyncio
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from generator.hardener import harden_from_failures

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_failure", type=int, default=1)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()
    asyncio.run(harden_from_failures(args.from_run, args.out, args.per_failure, model=args.model))
    print(f"Hardened suite written → {args.out}")
