from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Placeholder adapter for Joern/Slither integration. "
            "It keeps the pipeline unblocked when external analyzers are unavailable."
        )
    )
    p.add_argument("--tool", type=str, choices=["joern", "slither"], required=True)
    p.add_argument("--input", type=str, required=True, help="Input solidity file or directory.")
    p.add_argument("--output", type=str, required=True, help="Output json path.")
    p.add_argument("--mock", action="store_true", help="Write mock output schema.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.mock:
        payload = {
            "tool": args.tool,
            "input": str(Path(args.input).resolve()),
            "status": "mock",
            "ast_path": None,
            "cfg_path": None,
            "dfg_path": None,
            "notes": "Replace this file with real Joern/Slither invocation in your environment.",
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[joern_or_slither_adapter] mock output -> {out_path}")
        return
    raise RuntimeError(
        "External analyzer execution is not wired by default. Use --mock first, then replace adapter logic."
    )


if __name__ == "__main__":
    main()

