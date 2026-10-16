"""Conservative, source-reference-driven visual routing; no claim rewriting."""
from copy import deepcopy
import re

VERSION = 'provenance_visual_binding'
LABEL = re.compile(r'\b(Fig(?:ure)?\.?|Table)\s*(\d+)(?![\d.])', re.I)


def labels(text):
    return {('table' if k.lower() == 'table' else 'figure', n) for k, n in LABEL.findall(text)}


def reroute_visual_bindings(story, document, assets):
    result = deepcopy(story)
    blocks = {b['id']: b.get('text', '') for b in document['blocks']}
    actions = []
    for module in result['modules']:
        if module.get('role') != 'theory':
            continue
        for visual in list(module['visuals']):
            aid = visual['asset_id']
            caption = assets[aid]['original']['caption']
            # A theoretical diagram is valid. Do not move it based merely on role.
            if re.search(r'\b(theorem|theoretical|proof|mechanistic similarity|framework)\b', caption, re.I):
                continue
            first = LABEL.search(caption)
            key = (('table' if first.group(1).lower() == 'table' else 'figure'), first.group(2)) if first else None
            if not key:
                continue
            candidates = []
            for other in result['modules']:
                for index, bullet in enumerate(other['bullets']):
                    supporting = [sid for sid in bullet['source_ids'] if key in labels(blocks.get(sid, ''))]
                    if supporting:
                        candidates.append((len(supporting), other['id'] != module['id'], other, index, supporting))
            candidates.sort(key=lambda row: (row[0], not row[1]), reverse=True)
            if not candidates or candidates[0][2]['id'] == module['id']:
                continue
            _, _, target, index, refs = candidates[0]
            if target.get('role') not in ('results', 'analysis', 'experiments'):
                continue
            module['visuals'].remove(visual)
            target['visuals'].append(visual)
            for group in result['modules']:
                for bullet in group['bullets']:
                    if aid in bullet.get('asset_ids', []):
                        bullet['asset_ids'] = [a for a in bullet['asset_ids'] if a != aid]
            # The cited source explicitly names this figure, not an unrelated table.
            bullet = target['bullets'][index]
            mentioned = set().union(*(labels(blocks.get(sid, '')) for sid in bullet['source_ids']))
            previous = list(bullet.get('asset_ids', []))
            bullet['asset_ids'] = [a for a in previous if labels(assets[a]['original']['caption']) & mentioned]
            if aid not in bullet['asset_ids']:
                bullet['asset_ids'].append(aid)
            actions.append(dict(asset_id=aid, from_module=module['id'], to_module=target['id'],
                                target_bullet=index, source_ids=refs, previous_bullet_bindings=previous,
                                evidence_label=list(key), reason='Explicit figure reference in cited empirical source; no corresponding theoretical support'))
    return result, dict(version=VERSION, actions=actions, words_unchanged=True,
                        text_only_modules=[m['id'] for m in result['modules'] if not m['visuals']])

