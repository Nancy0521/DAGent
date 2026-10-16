"""Lossless provenance for MinerU table bodies, without OCR or model calls.

The source is the supplied parser output, NOT a claim that its transcription is
PDF-verified. Only a rectangular, fully explicit HTML table is accepted. Cell
text is deterministically decoded; raw HTML, exact offsets, spans and hashes are
kept. No values, headers, inequalities or missing cells are inferred/repaired.
"""
from __future__ import annotations

import hashlib
from html import unescape
from html.entities import html5
from html.parser import HTMLParser
import json
import re


class TableParseError(ValueError):
    pass


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class _TableParser(HTMLParser):
    INLINE = {'b', 'strong', 'i', 'em', 'span', 'div', 'p', 'br', 'sup', 'sub'}
    GROUPS = {'thead', 'tbody', 'tfoot'}

    def __init__(self, raw):
        super().__init__(convert_charrefs=False)
        self.raw = raw
        self.lines = [0] + [m.end() for m in re.finditer('\n', raw)]
        self.stack, self.rows, self.cells = [], [], []
        self.cell = None
        self.table_count = 0

    def source_offset(self):
        line, col = self.getpos()
        return self.lines[line - 1] + col

    def handle_starttag(self, tag, attrs):
        if len(attrs) != len({k for k, v in attrs}):
            raise TableParseError('duplicate_attributes')
        attrs = dict(attrs)
        if tag == 'table':
            if self.stack or self.table_count:
                raise TableParseError('nested_or_multiple_tables')
            self.table_count += 1
        elif tag in self.GROUPS:
            if not self.stack or self.stack[-1] != 'table':
                raise TableParseError('invalid_row_group')
        elif tag == 'tr':
            if not self.stack or self.stack[-1] not in {'table'} | self.GROUPS:
                raise TableParseError('invalid_row')
            if len(self.rows) >= 128:
                raise TableParseError('too_many_rows')
            self.rows.append([])
        elif tag in ('td', 'th'):
            if not self.stack or self.stack[-1] != 'tr' or self.cell is not None:
                raise TableParseError('invalid_cell')
            spans = {}
            for key in ('rowspan', 'colspan'):
                value = attrs.get(key, '1')
                if not isinstance(value, str) or not re.fullmatch(r'[1-9]\d?', value) or int(value) > 64:
                    raise TableParseError('invalid_' + key)
                spans[key] = int(value)
            self.cell = dict(row=len(self.rows) - 1, tag=tag, start=self.source_offset(),
                             parts=[], **spans)
        elif tag in self.INLINE:
            if self.cell is None:
                raise TableParseError('inline_outside_cell')
            if tag in ('br', 'p', 'div'):
                self.cell['parts'].append('\n')
            elif tag in ('sup', 'sub'):
                # Explicit notation preserves the distinction from a baseline digit.
                self.cell['parts'].append('^{' if tag == 'sup' else '_{')
        else:
            raise TableParseError('unsupported_tag:' + tag)
        if tag != 'br':
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        if tag != 'br':
            raise TableParseError('unsupported_self_closing_tag')
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            raise TableParseError('unbalanced_markup')
        self.stack.pop()
        if tag in ('sup', 'sub'):
            self.cell['parts'].append('}')
        elif tag in ('p', 'div'):
            self.cell['parts'].append('\n')
        elif tag in ('td', 'th'):
            end = self.raw.find('>', self.source_offset())
            if end < 0:
                raise TableParseError('unterminated_cell')
            cell = self.cell
            cell['end'] = end + 1
            cell['raw_html'] = self.raw[cell['start']:cell['end']]
            cell['raw_sha256'] = digest(cell['raw_html'])
            cell['text'] = ' '.join(''.join(cell.pop('parts')).split())
            self.rows[-1].append(cell)
            self.cells.append(cell)
            self.cell = None

    def handle_data(self, text):
        if self.cell is not None:
            self.cell['parts'].append(text)
        elif text.strip():
            raise TableParseError('text_outside_cell')

    def handle_entityref(self, name):
        raw = '&' + name + ';'
        decoded = unescape(raw)
        if name + ';' not in html5:
            raise TableParseError('unknown_entity')
        self.handle_data(decoded)

    def handle_charref(self, name):
        self.handle_data(unescape('&#' + name + ';'))

    def handle_comment(self, text):
        raise TableParseError('unsupported_comment')

    def handle_decl(self, decl):
        raise TableParseError('unsupported_declaration')

    def handle_pi(self, data):
        raise TableParseError('unsupported_processing_instruction')


def parse_table_html(raw, table_id):
    """Resolve spans, but reject ragged/overlapping/incomplete/malicious markup."""
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 250_000:
        raise TableParseError('missing_or_oversized_html')
    parser = _TableParser(raw)
    parser.feed(raw)
    parser.close()
    if parser.stack or parser.table_count != 1 or not parser.rows or not parser.cells:
        raise TableParseError('incomplete_or_empty_table')
    occupied = {}
    for row, cells in enumerate(parser.rows):
        col = 0
        for cell in cells:
            while (row, col) in occupied:
                col += 1
            if row + cell['rowspan'] > len(parser.rows) or col + cell['colspan'] > 64:
                raise TableParseError('span_outside_table')
            cell.update(col=col, id=f'{table_id}:r{row}c{col}')
            for r in range(row, row + cell['rowspan']):
                for c in range(col, col + cell['colspan']):
                    if (r, c) in occupied:
                        raise TableParseError('overlapping_spans')
                    occupied[r, c] = cell['id']
            col += cell['colspan']
    width = max(c for r, c in occupied) + 1
    if any((r, c) not in occupied for r in range(len(parser.rows)) for c in range(width)):
        raise TableParseError('ragged_table')
    # This is a display convention, explicitly NOT inferred scientific semantics.
    # All top-row rowspans remain intact; th-tagged header rows are also retained.
    header_rows = max(c['rowspan'] for c in parser.rows[0])
    while header_rows < len(parser.rows) and parser.rows[header_rows] and all(
            c['tag'] == 'th' for c in parser.rows[header_rows]):
        header_rows += 1
    return dict(version=1, status='parsed', id=table_id, raw_html=raw,
                raw_sha256=digest(raw), rows=len(parser.rows), columns=width,
                header_rows=header_rows, header_basis='first_row_spans_and_leading_th_rows_not_semantic_inference',
                cells=parser.cells,
                grid=[[occupied[r, c] for c in range(width)] for r in range(len(parser.rows))],
                transforms=['HTML entities decoded', 'cell whitespace collapsed',
                            'sup/sub tagged as ^{...}/_{...}; no TeX or numeric rewriting'],
                verification='parsed_source_only_not_pdf_verified')


def row_evidence(table, caption, record_index, page, source_sha256):
    """One row per source block, with its complete header context and provenance.

    JSON tuples prevent a literal pipe/newline in a cell from changing a column.
    A span is repeated by reference in the grid, not silently filled with zeros.
    Header context never crosses a table identity. Empty cells remain empty.
    """
    cells = {c['id']: c for c in table['cells']}
    headers = []
    for r in range(table['header_rows']):
        headers.append([cells[c]['text'] for c in table['grid'][r]])
    blocks = []
    context_cell = None
    for r in range(table['rows']):
        row_ids = list(dict.fromkeys(table['grid'][r]))
        if r >= table['header_rows'] and len(row_ids) == 1:
            candidate = cells[row_ids[0]]
            if candidate['colspan'] == table['columns'] and candidate['rowspan'] == 1:
                context_cell = candidate
        row_cells = [cells[c]['text'] for c in table['grid'][r]]
        identity = f'b{record_index}_table_r{r}'
        content = json.dumps(dict(caption=caption, headers=headers, row_index=r,
                                  cells=row_cells, footnotes=table.get('footnotes', []),
                                  preceding_full_width_row=context_cell['text'] if context_cell else None), ensure_ascii=False, separators=(',', ':'))
        cell_ids = list(dict.fromkeys(c for row in table['grid'][:table['header_rows']] for c in row))
        cell_ids = list(dict.fromkeys(cell_ids + table['grid'][r] + ([context_cell['id']] if context_cell else [])))
        blocks.append(dict(id=identity, page=page, text=content, heading=False, kind='table_row',
                           asset_id=table['id'], table_row=r, cell_ids=cell_ids,
                           provenance=dict(kind='deterministic_table_serialization',
                                           parsed_json_sha256=source_sha256, record_index=record_index,
                                           table_body_sha256=table['raw_sha256'],
                                           block_sha256=digest(content),
                                           verification=table['verification']),
                           source_cells=[{k:cells[c][k] for k in ('id', 'start', 'end', 'raw_html', 'raw_sha256')}
                                         for c in cell_ids]))
    table['source_ids'] = [b['id'] for b in blocks]
    return blocks


def record_table(record, record_index, source_sha256):
    """Always retain rejected HTML in the returned audit, never guess a grid."""
    raw = record.get('table_body')
    if not raw:
        return dict(status='absent', id=f'a{record_index}'), []
    try:
        table = parse_table_html(raw, f'a{record_index}')
    except TableParseError as exc:
        return dict(status='rejected', id=f'a{record_index}', reason=str(exc), raw_html=raw,
                    raw_sha256=digest(raw) if isinstance(raw, str) else None), []
    captions = record.get('table_caption', [])
    caption = '\n'.join(captions) if isinstance(captions, list) else str(captions)
    table['parsed_json_sha256'] = source_sha256
    table['record_index'] = record_index
    # Footnotes are source qualifiers, not ornaments. They stay with the table.
    notes = record.get('table_footnote', [])
    notes = notes if isinstance(notes, list) else [str(notes)]
    if any(not isinstance(n, str) for n in notes):
        return dict(status='rejected', id=f'a{record_index}', reason='invalid_footnotes',
                    raw_html=raw, raw_sha256=digest(raw), raw_footnotes=notes), []
    table['footnotes'] = notes
    table['footnotes_sha256'] = digest(json.dumps(notes, ensure_ascii=False))
    blocks = row_evidence(table, caption, record_index, int(record.get('page_idx', 0)) + 1, source_sha256)
    for i, text in enumerate(notes):
        if not isinstance(text, str) or not text.strip():
            continue
        sid = f'b{record_index}_table_note{i}'
        blocks.append(dict(id=sid, page=int(record.get('page_idx', 0))+1,
                           text=text, heading=False, kind='table_footnote', asset_id=table['id'],
                           provenance=dict(parsed_json_sha256=source_sha256, record_index=record_index,
                                           field='table_footnote', field_index=i, block_sha256=digest(text))))
        table['source_ids'].append(sid)
    return table, blocks


def route_table_sources(selected, document):
    """Pre-generation routing by exact selected asset identity, never fuzzy prose.

    Each selected table's full rows/qualifiers reach its caption and the unique
    consolidated text section of the same role. No winner rows are selected, no
    cross-table borrowing, no reinterpretation of an archived model answer.
    The added context is online inference input, NOT excluded skill training.
    """
    from copy import deepcopy
    from dagent.core.document import estimate_tokens
    result = deepcopy(selected)
    assets = {a['id']: a for a in document.assets}
    sources = {b['id']: b for b in document.blocks}
    audit = []
    for node in result:
        asset = assets.get(node.get('asset_id'), {})
        table = asset.get('table_evidence', {})
        if table.get('status') != 'parsed':
            continue
        ids = table['source_ids']
        if any(s not in sources or sources[s].get('asset_id') != asset['id'] for s in ids):
            raise ValueError('Table evidence routing source identity mismatch')
        targets = [node]
        narratives = [n for n in result if not n.get('asset_id') and n['role'] == node['role']]
        # No arbitrary choice among multiple same-role sections.
        if len(narratives) == 1:
            targets.extend(narratives)
        for target in targets:
            added = [s for s in ids if s not in target['source_ids']]
            target['source_ids'] = list(dict.fromkeys(target['source_ids'] + ids))
            audit.append(dict(asset_id=asset['id'], target_node=target['id'], added_source_ids=added,
                              basis='exact_selected_table_identity_and_same_role',
                              table_body_sha256=table['raw_sha256'],
                              added_source_tokens_estimate=sum(estimate_tokens(sources[s]['text']) for s in added),
                              token_accounting='normal_online_input_not_skill_generation',
                              verification=table['verification']))
    return result, audit
