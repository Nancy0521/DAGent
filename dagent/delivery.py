"""Check every frozen cohort member; process exit is not proof of delivery.

This is a local artifact check, not a factual/visual quality evaluation. An
offline content replay cannot satisfy a new online generation run's delivery.
"""
import argparse
import json
from pathlib import Path


def read_object(path):
    try:
        value = Path(path).read_text(encoding='utf-8')
        value = json.loads(value)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def check_delivery(run_dir, require_generation=False):
    root = Path(run_dir)
    protocol = read_object(root / 'protocol.json')
    if protocol.get('version') != 'dagent':
        raise ValueError('Expected a DAGent run')
    entries = protocol.get('entries', [])
    ids = [e.get('paper_id') for e in entries if isinstance(e, dict)]
    if (not ids or len(ids) != len(entries) or any(not isinstance(p, str) or not p
            or Path(p).name != p or p in ('.', '..') for p in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('The frozen cohort must contain unique, safe paper IDs')
    ready = complete = 0
    failures = []
    for paper_id in ids:
        errors = []
        preflight = read_object(root / 'input_preflight' / paper_id / 'status.json')
        if preflight.get('status') == 'ready':
            ready += 1
        else:
            errors.append('preflight:' + str(preflight.get('failure_code') or
                          preflight.get('status', 'missing_status')))
        if require_generation:
            folder = root / 'generation' / paper_id
            status = read_object(folder / 'status.json')
            generation_errors = []
            if status.get('status') != 'complete':
                generation_errors.append('generation:' + str(status.get('status', 'missing_status')))
            else:
                for name in ('poster.png', 'poster.pdf', 'poster.pptx'):
                    path = folder / name
                    if not path.is_file() or path.stat().st_size == 0:
                        generation_errors.append('missing_product:' + name)
                geometry = read_object(folder / 'geometry_check.json')
                render = read_object(folder / 'render_checks.json')
                evidence = read_object(folder / 'evidence_route.json')
                selection = read_object(folder / 'candidate_selection.json')
                if (geometry.get('valid') is not True or geometry.get('hard_violations') != 0
                        or geometry.get('missing_atoms') != []):
                    generation_errors.append('geometry_not_passed')
                if render.get('valid') is not True or render.get('defects') != []:
                    generation_errors.append('render_not_passed')
                if evidence.get('status') != 'passed' or evidence.get('errors') != []:
                    generation_errors.append('evidence_not_passed')
                selected = selection.get('selected')
                candidates = selection.get('candidates', [])
                if (selected not in ('candidate_0', 'candidate_1', 'candidate_2')
                        or status.get('chosen') != selected or not isinstance(candidates, list)
                        or not any(isinstance(c, dict) and c.get('id') == selected
                                   and c.get('status') == 'valid' for c in candidates)):
                    generation_errors.append('no_valid_selected_candidate')
            errors.extend(generation_errors)
            if not generation_errors:
                complete += 1
        if errors:
            failures.append(dict(paper_id=paper_id, errors=errors))
    return dict(scope='preflight_and_generation' if require_generation else 'preflight_only',
                denominator=len(ids), preflight_ready=ready,
                generation_complete=complete if require_generation else None,
                passed=not failures, failures=failures,
                note='Artifact delivery check only; not independent quality evaluation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--require-generation', action='store_true')
    args = parser.parse_args()
    try:
        result = check_delivery(args.run_dir, args.require_generation)
    except ValueError as exc:
        parser.exit(2, str(exc) + '\n')
    target = Path(args.run_dir) / 'delivery_check.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"预检通过：{result['preflight_ready']}/{result['denominator']}")
    if args.require_generation:
        print(f"正式海报交付：{result['generation_complete']}/{result['denominator']}")
    else:
        print('本次仅检查输入预检，不代表已经生成最终海报。')
    for row in result['failures']:
        print('未通过：' + row['paper_id'] + ' — ' + ', '.join(row['errors']))
    print('交付核对：' + str(target))
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
