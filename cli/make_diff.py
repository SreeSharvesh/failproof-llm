from __future__ import annotations
import sys, os, argparse, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from report.diff import diff_runs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True, help="run dir with guards OFF")
    ap.add_argument("--on", required=True,  help="run dir with guards ON")
    args = ap.parse_args()
    d = diff_runs(args.off, args.on)
    print(json.dumps(d, indent=2))
