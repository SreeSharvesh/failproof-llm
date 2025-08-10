import argparse, json
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from report.metrics import summarize

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    s = summarize(args.run)
    with open(f"{args.run}/summary.json", "w") as f:
        json.dump(s, f, indent=2)
    print(json.dumps(s, indent=2))
