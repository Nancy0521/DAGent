#!/usr/bin/env python3
"""Run the DAGent paper-to-poster pipeline."""
import argparse
import os
from pathlib import Path
import sys
import time

from dagent import logged
from dagent.agents import runner
from dagent.core.config import Config

ROOT = Path(__file__).resolve().parent


def project_path(path):
    path = path.expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'benchmark')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/dagent.json')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--generate', action='store_true', help='Call the configured paid API')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        parser.error('--workers must be between 1 and 4')
    args.input_dir = project_path(args.input_dir)
    args.config = project_path(args.config)
    run_dir = args.run_dir or ROOT / 'outputs' / f'run_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    run_dir = project_path(run_dir)
    if not run_dir.is_relative_to(ROOT / 'outputs'):
        parser.error('--run-dir must be inside outputs/')
    if not args.summarize_only:
        if not run_dir.exists() and not (args.input_dir / 'papers').is_dir():
            parser.error(f'Missing input directory: {args.input_dir / "papers"}')
        if not args.config.is_file():
            parser.error(f'Missing config: {args.config}')
        Config.load(args.config)
    command = ['--run-dir', str(run_dir), '--benchmark-root', str(args.input_dir),
               '--config', str(args.config), '--workers', str(args.workers)]
    if args.generate:
        command.append('--allow-paid-generation')
    if args.summarize_only:
        command.append('--summarize-only')
    print(f'Run directory: {run_dir}', flush=True)
    os.chdir(ROOT)
    logged.main(command)


if __name__ == '__main__':
    main()
