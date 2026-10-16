"""Read-only source inventory for the final, repository-local benchmark.

The papers and posters have different numeric ordering. Generation therefore
enumerates papers/ directly, without consulting poster/QA metadata or reusing
the historical aligned-unique manifest or parser caches.
"""
from pathlib import Path

from dagent.core.benchmark import slug
from dagent.core.document import sha256


DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / 'benchmark'


def discover_sources(root=DEFAULT_BENCHMARK_ROOT):
    import fitz

    root = Path(root).expanduser().resolve()
    directory = root / 'papers'
    if not directory.is_dir():
        raise FileNotFoundError(f'Benchmark papers directory not found: {directory}')
    if not directory.resolve().is_relative_to(root):
        raise ValueError('Benchmark papers directory points outside the dataset')
    pdfs = sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() == '.pdf')
    if not pdfs:
        raise ValueError(f'No benchmark PDFs found in {directory}')
    papers, ids, hashes = [], set(), {}
    for pdf in pdfs:
        path = pdf.resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError(f'Benchmark PDF points outside papers directory: {pdf.name}')
        paper_id = slug(pdf.stem)
        if paper_id in ids:
            raise ValueError(f'Duplicate benchmark paper ID: {paper_id}')
        ids.add(paper_id)
        digest = sha256(path)
        hashes.setdefault(digest, []).append(paper_id)
        # Keep invalid and duplicate sources in the full cohort, never silently
        # replace them with PDFs from the historical dataset or parser caches.
        try:
            with fitz.open(path) as doc:
                pages = len(doc)
                status = ('invalid_pdf_password_required' if doc.needs_pass else
                          'ready' if pages else 'invalid_pdf_no_pages')
        except (ValueError, RuntimeError, OSError) as exc:
            pages, status = None, 'invalid_pdf_' + type(exc).__name__
        papers.append(dict(paper_id=paper_id, pdf=str(path), pdf_sha256=digest,
                           pdf_pages=pages, input_status=status, parsed_json=None,
                           ground_truth=None, qa_path=None, source_metadata=None))
    return dict(schema_version=1, benchmark='repository-benchmark', root=str(root),
                papers_root=str(directory.resolve()), paper_count=len(papers),
                ready_count=sum(p['input_status'] == 'ready' for p in papers),
                invalid_input_count=sum(p['input_status'] != 'ready' for p in papers),
                deduplicated_by=None,
                duplicate_pdf_groups=[group for group in hashes.values() if len(group) > 1],
                ground_truth_used_for_generation=False,
                benchmark_meta_or_reference_posters_read=False,
                parsed_cache_used=False, papers=papers)
