#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from cat_cannon.app.algorithm_replay import (
    discover_replay_samples,
    pair_fixed_with_nearest_turret,
    run_paired_algorithm_replay,
    write_replay_csv,
)
from cat_cannon.config import load_counter_zones, load_system_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay reviewed cat detections through the dry-run firing algorithm."
    )
    parser.add_argument(
        "--dataset",
        default="data/review_samples/20260518-221734-reviewed-good/data/cat_training",
        help="Dataset root with images/<camera> and labels/<camera>, or YOLO split dirs.",
    )
    parser.add_argument("--config", default="configs/app.example.yaml")
    parser.add_argument("--zones", default="configs/zones.yaml")
    parser.add_argument("--max-timestamp-delta-ms", type=int, default=5000)
    parser.add_argument(
        "--aim-deadband-px",
        type=float,
        default=None,
        help="Override both horizontal and vertical aim-lock deadbands for what-if replay.",
    )
    parser.add_argument("--include-augmented", action="store_true")
    parser.add_argument(
        "--output-csv",
        default="data/replay_reports/cat_algorithm_replay.csv",
        help="CSV report path. Pass an empty string to skip writing.",
    )
    args = parser.parse_args()

    config = load_system_config(args.config)
    if args.aim_deadband_px is not None:
        config = replace(
            config,
            tracking_calibration=replace(
                config.tracking_calibration,
                horizontal_deadband_px=args.aim_deadband_px,
                vertical_deadband_px=args.aim_deadband_px,
            ),
        )
    zones = load_counter_zones(args.zones)
    samples = discover_replay_samples(
        args.dataset,
        include_augmented=args.include_augmented,
    )
    pairs = pair_fixed_with_nearest_turret(
        samples["fixed"],
        samples["turret"],
        max_timestamp_delta_ms=args.max_timestamp_delta_ms,
    )
    summary = run_paired_algorithm_replay(pairs=pairs, config=config, zones=zones)

    if args.output_csv:
        write_replay_csv(args.output_csv, summary.rows)

    print(f"dataset: {Path(args.dataset)}")
    print(f"config: {Path(args.config)}")
    print(f"zones: {Path(args.zones)}")
    print(f"fixed samples: {len(samples['fixed'])}")
    print(f"turret samples: {len(samples['turret'])}")
    print(f"paired samples: {len(pairs)}")
    print(
        "aim deadband px: "
        f"{config.tracking_calibration.horizontal_deadband_px:g} x "
        f"{config.tracking_calibration.vertical_deadband_px:g}"
    )
    print(f"fire count: {summary.fire_count}")
    if args.output_csv:
        print(f"csv: {Path(args.output_csv)}")
    _print_rows(summary.rows)
    return 0


def _print_rows(rows) -> None:
    if not rows:
        print("no replay rows")
        return
    print(
        "idx fixed_ts turret_ts delta_ms zone counter locked turn state fire "
        "err_x err_y cmd_pan cmd_tilt"
    )
    for row in rows:
        zone = row.active_zone_id or "-"
        print(
            f"{row.index:02d} "
            f"{row.fixed_timestamp_ms} "
            f"{row.turret_timestamp_ms} "
            f"{row.timestamp_delta_ms:>4} "
            f"{zone} "
            f"{_yes_no(row.counter_confirmed)} "
            f"{_yes_no(row.turret_aim_locked)} "
            f"{_yes_no(row.tracking_commanded)} "
            f"{row.state.value} "
            f"{_yes_no(row.fire_commanded)} "
            f"{_format_float(row.turret_error_x_px)} "
            f"{_format_float(row.turret_error_y_px)} "
            f"{_format_float(row.applied_pan_delta)} "
            f"{_format_float(row.applied_tilt_delta)}"
        )


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _format_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
