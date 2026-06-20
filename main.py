"""
AI Company Builder — entry point.
Usage:
    python main.py --idea "I want to build an AI startup for early diabetic retinopathy detection."
    python main.py  # prompts interactively
"""

import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from graph.builder import build_graph
from outputs.report import render_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Company Builder — multi-agent MedTech startup analyser"
    )
    parser.add_argument(
        "--idea",
        type=str,
        default=None,
        help="Startup idea in natural language",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="Healthcare / MedTech",
        help="Target domain (default: Healthcare / MedTech)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/latest_report.md",
        help="Path to write the final Markdown report",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Also dump raw state as JSON alongside the Markdown report",
    )
    return parser.parse_args()


def get_idea(args: argparse.Namespace) -> str:
    if args.idea:
        return args.idea.strip()
    print("\n=== AI Company Builder ===")
    print("Enter your startup idea (press Enter twice when done):\n")
    lines = []
    while True:
        line = input()
        if line == "" and lines:
            break
        lines.append(line)
    return " ".join(lines).strip()


def main() -> None:
    args = parse_args()
    idea = get_idea(args)

    if not idea:
        print("Error: no startup idea provided.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[*] Building company analysis for:\n    \"{idea}\"\n")

    graph = build_graph()

    initial_state = {
        "idea": idea,
        "domain": args.domain,
        "market_output": None,
        "research_output": None,
        "product_output": None,
        "architecture_output": None,
        "synthesis": None,
        "guardrail_flags": [],
        "viability_score": None,
        "final_report": None,
        "errors": [],
    }

    print("[*] Running agent graph...\n")
    final_state = graph.invoke(initial_state)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_report(final_state, output_path)
    print(f"\n[+] Report written to: {output_path}")

    if args.json_output:
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_state, f, indent=2, default=str)
        print(f"[+] JSON state written to: {json_path}")

    score = final_state.get("viability_score")
    if score is not None:
        print(f"\n[*] Viability score: {score}/100")

    flags = final_state.get("guardrail_flags", [])
    if flags:
        print(f"[!] Guardrail flags ({len(flags)}):")
        for flag in flags:
            print(f"    - {flag}")

    print("\n[+] Done.")


if __name__ == "__main__":
    main()