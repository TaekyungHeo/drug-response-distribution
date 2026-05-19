"""Render experiment reports from Jinja2 templates + data JSON files.

Usage:
    python scripts/render_reports.py                  # render all
    python scripts/render_reports.py --only 03_response_matching
    python scripts/render_reports.py --dry-run         # show what would be rendered

Walks experiments/**/report/README.md.j2, loads report/data/*.json,
renders to report/README.md. Also renders the root experiments/README.md
from experiments/README.md.j2 if present.

Single source of truth: numbers live in JSON, never in prose.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined

ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"


def fmt(value: float, decimals: int = 3) -> str:
    if value != value:  # NaN
        return "NaN"
    return f"{value:.{decimals}f}"


def signed(value: float, decimals: int = 3) -> str:
    if value != value:
        return "NaN"
    return f"{value:+.{decimals}f}"


def load_report_data(report_dir: Path) -> dict:
    data_dir = report_dir / "data"
    if not data_dir.exists():
        return {}
    combined: dict = {}
    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file) as f:
            combined[json_file.stem] = json.load(f)
    return combined


def find_templates(experiments_dir: Path, only: Optional[str] = None) -> list[Path]:
    templates = sorted(experiments_dir.rglob("report/README.md.j2"))
    if only:
        templates = [t for t in templates if only in str(t)]
    return templates


def render_one(template_path: Path, env: Environment, dry_run: bool = False) -> bool:
    report_dir = template_path.parent
    exp_dir = report_dir.parent
    output_path = report_dir / "README.md"

    data = load_report_data(report_dir)
    if not data:
        print(f"  SKIP {exp_dir.relative_to(ROOT)}: no data/*.json files")
        return False

    rel_template = str(template_path.relative_to(EXPERIMENTS_DIR))
    try:
        template = env.get_template(rel_template)
    except Exception as e:
        print(f"  ERROR loading template {rel_template}: {e}")
        return False

    exp_info = {
        "path": str(exp_dir.relative_to(ROOT)),
        "name": exp_dir.name,
        "group": exp_dir.parent.name if exp_dir.parent != EXPERIMENTS_DIR else "",
    }

    try:
        rendered = template.render(exp=exp_info, **data)
    except Exception as e:
        print(f"  ERROR rendering {rel_template}: {e}")
        return False

    if dry_run:
        print(f"  DRY-RUN {output_path.relative_to(ROOT)} ({len(rendered)} chars)")
        return True

    output_path.write_text(rendered + "\n")
    print(f"  RENDERED {output_path.relative_to(ROOT)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Render experiment reports from templates")
    parser.add_argument("--only", help="Filter: only render templates matching this substring")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be rendered")
    args = parser.parse_args()

    env = Environment(
        loader=FileSystemLoader(str(EXPERIMENTS_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["fmt"] = fmt
    env.filters["signed"] = signed
    env.filters["stdev"] = lambda xs: (sum((x - sum(xs)/len(xs))**2 for x in xs) / len(xs)) ** 0.5

    templates = find_templates(EXPERIMENTS_DIR, args.only)
    if not templates:
        print("No templates found.", file=sys.stderr)
        if args.only:
            print(f"  (filter: '{args.only}')", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(templates)} template(s)")
    rendered = 0
    for t in templates:
        if render_one(t, env, dry_run=args.dry_run):
            rendered += 1

    print(f"\n{'Would render' if args.dry_run else 'Rendered'} {rendered}/{len(templates)} reports")


if __name__ == "__main__":
    main()
