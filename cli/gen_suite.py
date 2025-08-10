import argparse, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generator.csv_html import gen_csv_suite, gen_html_suite

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # write CSV and HTML blocks one after another (JSONL)
    gen_csv_suite(args.out.replace(".jsonl","_csv.jsonl"), n=30)
    gen_html_suite(args.out.replace(".jsonl","_html.jsonl"), n=30)
    print("Suites generated.")
