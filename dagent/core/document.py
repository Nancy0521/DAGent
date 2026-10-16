"""Model-free PDF/MinerU ingestion plus API document understanding.

Existing parsed data are reused only when the benchmark adapter has verified the
source PDF hash. No Docling/Marker/OCR model is loaded or downloaded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import re
import unicodedata

from dagent.core.config import Config, dump_json, writable_path

ROLES = ('problem', 'motivation', 'contribution', 'method', 'experiment', 'conclusion')


def canonical_role(entry):
    """Recover only explicit role/section labels, never infer roles from prose."""
    aliases = {'methods': 'method', 'methodology': 'method', 'experiments': 'experiment',
               'results': 'experiment', 'contributions': 'contribution', 'conclusions': 'conclusion'}
    for key in ('role', 'section'):
        value = entry.get(key)
        if not isinstance(value, str):
            continue
        value = value.strip().lower()
        role = aliases.get(value, value)
        if role in ROLES:
            return role
    return None


def estimate_tokens(text: str) -> int:
    # Stable offline estimate for content planning, NOT billed API accounting.
    return max(1, (len(text.encode('utf-8')) + 3)//4) if text else 0


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?。！？])\s+|\n+', text) if s.strip()]


def extractive_fit(text: str, tokens: int) -> str:
    """Preserve complete source sentences; never invent or truncate a numeric claim."""
    kept = []
    for sentence in sentences(text):
        if estimate_tokens('\n'.join(kept + [sentence])) <= tokens:
            kept.append(sentence)
    return '\n'.join(kept)


@dataclass
class EvidenceCandidate:
    id: str
    section: str
    role: str
    text: str
    source_ids: list[str]
    importance: float = 0.5
    visualizability: float = 0.5
    asset_id: str | None = None
    details: list[dict] = field(default_factory=list)
    asset_kind: str | None = None
    # Optional contract attached to a selected evidence candidate.
    method_contract: dict | None = None

    @property
    def tokens(self):
        return estimate_tokens(self.text)


@dataclass
class SourceDocument:
    paper_id: str
    pdf_path: str
    pdf_sha256: str
    title: str
    authors: str
    blocks: list[dict]
    assets: list[dict]
    candidates: list[EvidenceCandidate] = field(default_factory=list)
    parsing: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def read_mineru(path: Path, *, table_audit=None):
    records = json.loads(path.read_text(encoding='utf-8'))
    blocks, assets = [], []
    source_sha256 = sha256(path)
    from dagent.core.table_evidence import record_table
    for i, record in enumerate(records):
        kind = record.get('type')
        page = int(record.get('page_idx', 0)) + 1
        if kind == 'text' and record.get('text', '').strip():
            blocks.append(dict(id=f'b{i}', page=page, text=record['text'].strip(),
                               bbox=record.get('bbox'), heading=bool(record.get('text_level'))))
        table = None
        if kind == 'table':
            table, table_blocks = record_table(record, i, source_sha256)
            blocks.extend(table_blocks)  # Keep textual evidence even if the image is missing.
            if table_audit is not None:
                table_audit.append(table)
        if kind in ('image', 'table') and record.get('img_path'):
            asset_path = (path.parent / record['img_path']).resolve()
            if not asset_path.is_relative_to(path.parent.resolve()) or not asset_path.is_file():
                continue
            captions = record.get('image_caption' if kind == 'image' else 'table_caption', [])
            caption = '\n'.join(captions) if isinstance(captions, list) else str(captions)
            assets.append(dict(id=f'a{i}', kind='figure' if kind == 'image' else 'table',
                               path=str(asset_path), caption=caption, page=page,
                               extraction='existing_mineru', bbox=record.get('bbox')))
            if table is not None:
                assets[-1]['table_evidence'] = table
    return blocks, assets


def read_pdf(pdf: Path, output: Path):
    import fitz
    from PIL import Image
    blocks, assets = [], []
    source_digest = sha256(Path(pdf))
    output = writable_path(output)
    output.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf) as doc:
        if len(doc)==0:
            raise ValueError('Source PDF has no readable pages; invalid benchmark input, not a generation failure')
        metadata = doc.metadata or {}
        for page_index, page in enumerate(doc):
            raw = [b for b in page.get_text('blocks', sort=True) if b[6] == 0 and b[4].strip()]
            for block in raw:
                text = re.sub(r'(?<=\w)-\n(?=\w)', '', block[4]).strip()
                blocks.append(dict(id=f'p{page_index+1}b{len(blocks)}', page=page_index+1,
                                   bbox=list(block[:4]), text=text,
                                   heading=(len(text) < 110 and not text.rstrip().endswith('.'))))
            from dagent.core.native_assets import caption_graphics_crops
            image_rects = [fitz.Rect(d['bbox']) for d in page.get_image_info()]
            extracted = []
            for crop in caption_graphics_crops(page, raw):
                clip = fitz.Rect(crop['bbox'])
                extracted.append(clip)
                path = writable_path(output / f'page{page_index+1}_asset{len(assets)}.png')
                page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip, alpha=False).save(path)
                assets.append(dict(crop, id=f'a{len(assets)}', path=str(path), page=page_index+1,
                                   source_crop=dict(pdf_sha256=source_digest,page=page_index+1,
                                                    bbox=list(clip),caption_bbox=crop['caption_bbox'])))
            # Raster images without captions remain explicitly marked, not fabricated captions.
            for rect in image_rects:
                if rect.width < 65 or rect.height < 50 or any((rect&r).get_area()/max(1,rect.get_area()) > .8 for r in extracted):
                    continue
                if rect.get_area() > page.rect.get_area()*.8:
                    continue
                path = writable_path(output / f'page{page_index+1}_asset{len(assets)}.png')
                page.get_pixmap(matrix=fitz.Matrix(3,3), clip=rect, alpha=False).save(path)
                assets.append(dict(id=f'a{len(assets)}', kind='figure', path=str(path), caption='',
                                   page=page_index+1, bbox=list(rect), extraction='native_pdf_raster_uncaptioned',
                                   source_crop=dict(pdf_sha256=source_digest,page=page_index+1,bbox=list(rect))))
        title = metadata.get('title', '').strip()
    for asset in assets:
        with Image.open(asset['path']) as im:
            asset['pixel_width'], asset['pixel_height'] = im.size
    return blocks, assets, title


def load_document(pdf: Path, output: Path, parsed_json: Path | None = None, paper_id=None) -> SourceDocument:
    pdf = pdf.resolve()
    table_audit = []
    if parsed_json:
        blocks, assets = read_mineru(parsed_json, table_audit=table_audit)
        from PIL import Image
        for asset in assets:
            with Image.open(asset['path']) as im:
                asset['pixel_width'], asset['pixel_height'] = im.size
        title = blocks[0]['text'] if blocks else pdf.stem
    else:
        blocks, assets, title = read_pdf(pdf, output / 'assets')
    if not blocks or sum(len(b['text']) for b in blocks) < 100:
        raise ValueError('No usable embedded PDF text. Supply existing parsed JSON; local OCR is intentionally disabled.')
    # Remove bibliography only if a standalone heading is found (not a prose mention).
    end = next((i for i,b in enumerate(blocks) if i > len(blocks)//3 and
                re.fullmatch(r'(?:\d+[. ]*)?(References|Bibliography)', b['text'].strip(), re.I)), len(blocks))
    blocks = blocks[:end]
    return SourceDocument(paper_id or pdf.parent.name, str(pdf), sha256(pdf), title or blocks[0]['text'], '', blocks, assets,
                    parsing=dict(backend='existing_mineru' if parsed_json else 'pymupdf_native',
                                 parsed_json=str(parsed_json) if parsed_json else None,
                                 table_audit=table_audit,
                                 local_inference=False, asset_crops_require_visual_review=not bool(parsed_json)))


def sample_evidence_context(doc: SourceDocument, config: Config, byte_limit: int | None = None):
    from dagent.core.content.source_context import sample
    return sample(doc, config, byte_limit)


def select_evidence_pointers(doc: SourceDocument, api, config: Config):
    from dagent.core.content.source_context import VERSION as SOURCE_POLICY, audit as audit_source_context
    # The main method uses a document/context capacity, not remaining API spend.
    input_capacity = config.max_document_chars
    if config.budget.max_tokens is not None:
        # Explicit finite budget mode.
        spent=api.ledger.totals([e for e in api.ledger.events if e['stage'].startswith('content')])['total_tokens']
        remaining=min(config.budget.max_tokens-api.ledger.totals()['total_tokens'],
                      int(config.budget.max_tokens*config.budget.content_fraction)-spent)
        input_capacity=max(0,int(remaining*.7)-config.text_model.max_output_tokens-2500)
    assets=[]
    asset_bytes=0
    for a in doc.assets[:40]:
        item={k:a[k] for k in ('id','kind','caption','page')}
        item['caption']=item['caption'][:1200]
        table = a.get('table_evidence', {})
        if table.get('status') == 'parsed':
            item['table_source_ids'] = table['source_ids']
        size=len(json.dumps(item,ensure_ascii=False).encode('utf-8'))+2
        if asset_bytes+size<=input_capacity*.3:
            assets.append(item)
            asset_bytes+=size
    sampled = sample_evidence_context(doc, config, max(0,input_capacity-asset_bytes))
    if not sampled:
        if config.budget.max_tokens is not None:
            from dagent.core.budget import BudgetExceeded
            raise BudgetExceeded('Insufficient content budget for even one grounded source block')
        raise ValueError('No grounded source block fits the document context capacity')
    payload = dict(blocks=sampled, assets=assets)
    doc.parsing['understanding_source_audit'] = audit_source_context(
        doc, payload, policy=SOURCE_POLICY, byte_capacity=max(0,input_capacity-asset_bytes))
    result = api.json('content_understanding',
        'You are the document-understanding agent for a scientific Poster Tree. Use source IDs only; do not invent facts.',
        'Structured table_row blocks preserve parsed table cells with headers; reference their block IDs for numerical results. '
        'Their transcription is parsed-source evidence, not independently PDF-verified. '
        'Build a candidate pool, NOT the final poster. Cover problem, motivation, contribution, method, experiment, conclusion. '
        'Return {"title":str,"authors":str,"nodes":[{"id":"n1","section":str,"role":str,"source_ids":[block_id],'
        '"importance":0..1,"visualizability":0..1}],"assets":[{"asset_id":str,"section":str,"role":str,"importance":0..1}]}. '
        'For EVERY text node and asset, role MUST be exactly one of these lowercase enum values: '
        '"problem", "motivation", "contribution", "method", "experiment", "conclusion". '
        'role is a category, NEVER a sentence, a summary, or an invented label. For example, '
        '{"section":"Problem","role":"problem","source_ids":["<exact input block id>"]}. '
        f'Use 10-{min(config.evidence_selection.max_candidates,14 if config.evidence_selection.semantic_capacity else 24)} concise, distinct text nodes, each with 1-3 relevant blocks. '
        'The first source ID is the core evidence block; remaining source IDs are optional detail children for EXPAND. '
        'Core evidence MUST be a substantive paragraph, never a bare section heading or an anaphoric fragment '
        'such as "By doing so, this is satisfied". Include its explanatory context in the same node. '
        'Prioritize the actual research question, novel mechanism, main quantitative results, limitations and takeaway. '
        'Do not spend poster space on lists of related work, generic motivation or hardware details. '
        'Use short section headings, not source paragraph headings. '
        'No bibliography, equations-only or author-affiliation nodes. Assets must have captions supporting their scientific meaning. '
        'Include the central method schematic and complementary primary results figures/tables when available; '
        'do not choose appendix detail or redundant plots instead of the main scientific evidence. '
        'Order nodes in scientific reading order. Title/authors must be copied from source, normalizing PDF line breaks, '
        'ligatures and small-cap spacing only. Conference headers are NOT paper titles.'
        + ' Authors are required attribution, not optional poster content: return ALL source author names, '
           'in source visual reading order, without et al. or shortening the list. Keep affiliations out of '
           'the authors field; do not omit names just because affiliations separate them. '
           'If authors cannot be grounded, return an empty authors string rather than inventing names.'
        + '\n' + json.dumps(payload, ensure_ascii=False))
    return validate_evidence_selection(doc, result, payload, config)


def resolve_asset_reference(reference, assets):
    """Resolve a published figure/table number only when the caption is unique.

    The model sometimes echoes ``fig2`` rather than the provided ``a17`` ID.
    Never map by array position, fuzzy title, or number alone: a declared caption
    label and kind must both match, and only assets visible in the request count.
    """
    if reference in assets:
        return reference
    alias = re.fullmatch(r'(figure|fig|table|tab)\.?[\s_:-]*(\d+)', reference, re.I)
    if not alias:
        return None
    kind = 'figure' if alias[1].lower().startswith('fig') else 'table'
    label = int(alias[2])
    matches = []
    for aid, asset in assets.items():
        caption = re.match(r'^\s*(Figure|Fig\.?|Table|Tab\.?)\s+(\d+)(?=\s|[:.\-]|$)',
                           asset.get('caption', ''), re.I)
        if (caption and asset.get('kind') == kind
                and ('figure' if caption[1].lower().startswith('fig') else 'table') == kind
                and int(caption[2]) == label):
            matches.append(aid)
    return matches[0] if len(matches) == 1 else None


def validate_evidence_selection(doc, result, payload, config):
    """Validate a recorded API result without spending or weakening source-ID checks."""
    visible={b['id'] for b in payload['blocks']}
    sources = {b['id']:b for b in doc.blocks if b['id'] in visible}
    seen, nodes = set(), []
    repairs, rejected = [], dict(invalid_schema=0, missing_or_reused_source=0, invalid_role=0)
    entries = result.get('nodes', [])
    if not isinstance(entries, list):
        raise ValueError('Document API nodes must be a list')
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('source_ids'), list):
            rejected['invalid_schema'] += 1
            continue
        ids = list(dict.fromkeys(i for i in entry['source_ids'] if isinstance(i, str) and i in sources))
        ids = [i for i in ids if i not in seen]
        if not ids:
            rejected['missing_or_reused_source'] += 1
            continue
        role = canonical_role(entry)
        if role is None:
            rejected['invalid_role'] += 1
            continue
        if entry.get('role') != role:
            repairs.append(dict(id=entry.get('id'), original_role=entry.get('role'),
                                section=entry.get('section'), normalized_role=role))
        core = ids[:1]
        if config.evidence_selection.semantic_capacity:
            # Recover only from evidence IDs the understanding model actually selected.
            # PDF headings are navigation, not a four-token scientific result.
            usable = [i for i in ids if not (
                len(sources[i]['text']) < 110 and
                (sources[i].get('heading') or
                 (len(sources[i]['text'].split()) < 8 and not re.search(r'[.!?]$', sources[i]['text']))))]
            if not usable:
                rejected.setdefault('heading_only_source', 0)
                rejected['heading_only_source'] += 1
                continue
            core = usable[:1]
            if len(sources[core[0]]['text']) < 100 and len(usable) > 1:
                core = usable[:2]  # retain context for short linked claims
            if core != ids[:1]:
                repairs.append(dict(id=entry.get('id'), evidence_core=core, reason='substantive_core'))
        seen.update(ids)
        text = '\n'.join(sources[i]['text'] for i in core)
        details = [dict(source_id=i, text=sources[i]['text']) for i in ids if i not in core]
        nodes.append(EvidenceCandidate(f'n{len(nodes)}', str(entry.get('section', role))[:100], role,
                               text, core, _unit(entry.get('importance', .5)), _unit(entry.get('visualizability', .5)),
                               details=details))
        if len(nodes) >= config.evidence_selection.max_candidates:
            break
    if not nodes:
        raise ValueError(f'Document API returned no valid source-grounded candidate nodes: {rejected}')
    visible_assets={a['id'] for a in payload['assets']}
    assets = {a['id']:a for a in doc.assets if a['id'] in visible_assets}
    used_assets = set()
    asset_repairs, rejected_assets = [], []
    asset_entries = result.get('assets', [])
    if not isinstance(asset_entries, list):
        asset_entries = []
    for entry in asset_entries:
        if not isinstance(entry, dict):
            continue
        original_aid = str(entry.get('asset_id'))
        aid = resolve_asset_reference(original_aid, assets)
        role = canonical_role(entry)
        if aid is None or aid in used_assets or not assets[aid]['caption'] or role is None:
            rejected_assets.append(dict(reference=original_aid, reason='unresolved_duplicate_or_invalid_asset'))
            continue
        if aid != original_aid:
            asset_repairs.append(dict(reference=original_aid, asset_id=aid,
                                      basis='unique_visible_caption_label_and_kind'))
        if entry.get('role') != role:
            repairs.append(dict(asset_id=aid, original_role=entry.get('role'),
                                section=entry.get('section'), normalized_role=role))
        used_assets.add(aid)
        a = assets[aid]
        nodes.append(EvidenceCandidate('visual_'+aid, str(entry.get('section', role))[:100], role,
                               a['caption'], [aid], _unit(entry.get('importance', .6)), .95, aid, asset_kind=a['kind']))
        if len(nodes) >= config.evidence_selection.max_candidates + config.evidence_selection.max_figures*2:
            break
    full_text = ' '.join(' '.join(b['text'].split()) for b in doc.blocks[:25])
    def canonical_metadata(text, author=False):
        text = re.sub(r'<sup>.*?</sup>', '', text, flags=re.I)
        text = re.sub(r'<[^>]+>', '', text)
        # Affiliation markers interrupt author names in PDFs. Ignore digits ONLY for authors.
        return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold()
                       if c.isalpha() or (c.isdigit() and not author))
    for key in ('title',):
        value = ' '.join(str(result.get(key, '')).split())
        supported = value and canonical_metadata(value, key == 'authors') in canonical_metadata(full_text, key == 'authors')
        is_header = key == 'title' and re.match(r'(?i)(published as|under review|proceedings of|arxiv:)', value)
        if supported and not is_header:
            setattr(doc, key, value)
    from dagent.core.content.source_metadata import validate_authors
    doc.parsing['author_metadata_audit'] = validate_authors(doc, result.get('authors'))
    if config.evidence_selection.semantic_capacity:
        order = {role: i for i, role in enumerate(ROLES)}
        nodes.sort(key=lambda n: (order[n.role], 0 if n.asset_id else 1))
    doc.candidates = nodes
    doc.parsing.update(understanding_blocks=len(payload['blocks']), total_blocks=len(doc.blocks),
                       role_normalizations=repairs, rejected_text_nodes=rejected,
                       asset_reference_normalizations=asset_repairs, rejected_assets=rejected_assets)
    return doc


def _unit(value):
    try:
        value = float(value)
        return min(1., max(0., value)) if value == value else .5
    except (ValueError, TypeError):
        return .5
