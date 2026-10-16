"""Read-only adapter for the reference Paper2Poster aligned-unique benchmark."""
from __future__ import annotations

from pathlib import Path
import json
import re

from dagent.core.document import sha256
from dagent.core.config import dump_json


def slug(text):
    return re.sub(r'[^\w.-]+','_',text,flags=re.UNICODE).strip('._') or 'paper'


def discover(root: Path, reference: Path, include_parsed=True):
    root=root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Benchmark directory not found: {root}')
    parsed={}
    if include_parsed:
        for parent in sorted((reference/'paper2poster').glob('parsed_papers*')):
            for folder in sorted(parent.iterdir()):
                if not folder.is_dir():
                    continue
                content=sorted(folder.glob('*_content_list.json'))
                if not content:
                    continue
                pdfs=[p for p in folder.glob('*.pdf') if not p.stem.endswith(('_origin','_layout','_span'))]
                for pdf in pdfs:
                    digest=sha256(pdf)
                    parsed.setdefault(digest,str(content[0].resolve()))
    papers=[]
    seen={}
    for folder in sorted(root.iterdir()):
        pdf=folder/'paper.pdf'
        if not folder.is_dir() or not pdf.is_file():
            continue
        digest=sha256(pdf)
        if digest in seen:
            continue
        seen[digest]=folder.name
        gt=folder/'poster.png'
        qa=folder/'o3_qa.json'
        # Preflight input integrity, never remove bad sources from the benchmark.
        try:
            import fitz
            with fitz.open(pdf) as document:
                page_count=len(document)
            input_status='ready' if page_count>0 else 'invalid_pdf_no_pages'
        except Exception as exc:
            page_count=None
            input_status='invalid_pdf_'+type(exc).__name__
        papers.append(dict(paper_id=folder.name,pdf=str(pdf),pdf_sha256=digest,
                           pdf_pages=page_count,input_status=input_status,
                           ground_truth=str(gt) if gt.is_file() else None,
                           qa_path=str(qa) if qa.is_file() else None,
                           parsed_json=parsed.get(digest),
                           source_metadata=(folder/'source.txt').read_text(errors='replace') if (folder/'source.txt').is_file() else None))
    if not papers:
        raise ValueError('No benchmark paper.pdf files found')
    return dict(schema_version=1,benchmark='Paper2Poster-aligned-unique',root=str(root),
                reference_root=str(reference.resolve()),paper_count=len(papers),
                ready_count=sum(p['input_status']=='ready' for p in papers),
                invalid_input_count=sum(p['input_status']!='ready' for p in papers),
                deduplicated_by='full_pdf_sha256',ground_truth_used_for_generation=False,
                parsed_data_verified_by_pdf_sha256=True,papers=papers)


def validate_entry(entry):
    path=Path(entry['pdf'])
    if sha256(path)!=entry['pdf_sha256']:
        raise ValueError(f"Benchmark PDF changed: {entry['paper_id']}; rebuild manifest")
    return entry


def select_papers(manifest, *, papers=None, limit=0, ready_only=False):
    """Explicit smoke-test subset, without deleting bad inputs from the benchmark."""
    if type(limit) is not int or limit < 0:
        raise ValueError('limit must be a nonnegative integer')
    entries = manifest['papers']
    if papers:
        missing = set(papers)-{e['paper_id'] for e in entries}
        if missing:
            raise ValueError(f'Unknown paper IDs: {sorted(missing)}')
        entries = [e for e in entries if e['paper_id'] in papers]
    excluded, selected = [], []
    for entry in entries:
        if ready_only:
            # Recheck current files, rather than trusting possibly stale manifest flags.
            try:
                import fitz
                validate_entry(entry)
                with fitz.open(entry['pdf']) as document:
                    pages = len(document)
                    if not pages:
                        raise ValueError('invalid_pdf_no_pages')
                    if document.needs_pass:
                        raise ValueError('invalid_pdf_password_required')
            except Exception as exc:
                excluded.append(dict(paper_id=entry['paper_id'], reason=str(exc), error_type=type(exc).__name__))
                continue
        selected.append(entry)
    available = len(selected)
    if limit:
        selected = selected[:limit]
    report = dict(ready_only=ready_only, requested_limit=limit, requested_papers=papers,
                  considered=len(entries), available=available, selected_count=len(selected),
                  selected_papers=[e['paper_id'] for e in selected], excluded=excluded,
                  benchmark_manifest_unchanged=True,
                  note='Readability preflight is not a guarantee of text extraction or generation success; '
                       'ready-only is a filtered smoke-test subset, not a full-benchmark success rate.')
    return selected, report
