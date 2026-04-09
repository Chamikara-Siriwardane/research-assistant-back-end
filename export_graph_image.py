"""Export the compiled research graph as a PNG image.

Usage:
    python export_graph_image.py
    python export_graph_image.py --output artifacts/research_graph.png --xray
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.graph import compiled_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the LangGraph research pipeline to a PNG file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research_graph.png"),
        help="Path to the PNG file to create.",
    )
    parser.add_argument(
        "--xray",
        action="store_true",
        help="Include nested subgraphs in the rendered diagram.",
    )
    parser.add_argument(
        "--mermaid-output",
        type=Path,
        default=None,
        help="Optional path to also save the Mermaid source text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    graph = compiled_graph.get_graph(xray=args.xray)

    if args.mermaid_output is not None:
        args.mermaid_output.parent.mkdir(parents=True, exist_ok=True)
        args.mermaid_output.write_text(graph.draw_mermaid(), encoding="utf-8")

    png_bytes = graph.draw_mermaid_png()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(png_bytes)

    print(f"Wrote PNG to {args.output.resolve()}")
    if args.mermaid_output is not None:
        print(f"Wrote Mermaid text to {args.mermaid_output.resolve()}")


if __name__ == "__main__":
    main()