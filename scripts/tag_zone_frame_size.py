#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cat_cannon.config import load_counter_zones, save_counter_zones


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tag an existing pixel-space zones.yaml with its source frame size."
    )
    parser.add_argument("--zones", default="configs/zones.yaml")
    parser.add_argument("--source-width", type=int, required=True)
    parser.add_argument("--source-height", type=int, required=True)
    parser.add_argument(
        "--output",
        default="",
        help="Output path. Defaults to updating --zones in place.",
    )
    args = parser.parse_args()

    zones_path = Path(args.zones)
    output_path = Path(args.output) if args.output else zones_path
    zones = load_counter_zones(zones_path)
    save_counter_zones(
        output_path,
        zones,
        frame_width=args.source_width,
        frame_height=args.source_height,
    )
    print(f"tagged {output_path} as {args.source_width}x{args.source_height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
