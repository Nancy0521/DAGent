"""Grounded revision of the Writing Agent's poster storyboard.

The revision uses verified source blocks and prepared visual assets. It keeps
the latest usable storyboard after bounded feedback and validation.
"""
from copy import deepcopy
import json
from pathlib import Path

from dagent.core.config import dump_json
from dagent.core.api import APIError,safe_error_metadata
from dagent.core.budget import BudgetExceeded
from dagent.agents.writing_agent import SYSTEM,validate_provenance_storyboard,body_bullets,word_count,assemble_evidence_package
from dagent.agents.evidence_agent import visual_mode
from dagent.agents.retry import ATTEMPT_LIMIT, failure_feedback
from dagent.agents.source_refs import normalize_source_references, source_reference_errors

VERSION='grounded_storyboard_revision'


def assemble_revision_evidence(document, notes, prepared):
    ids=set()
    for section in [notes.get('intro',{}),*notes.get('modules',[]),notes.get('conclusion',{})]:
        if not isinstance(section, dict):
            continue
        for key in ('bullets', 'visuals'):
            entries = section.get(key, [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict) and isinstance(entry.get('source_ids'), list):
                    ids.update(s for s in entry['source_ids'] if isinstance(s,str))
    # Reuse original evidence for both headline claims and essential qualifications.
    for c in document.get('candidates',[]):
        if c.get('role') in ('problem','conclusion','limitation'):
            ids.update(c.get('source_ids',[]))
    # A malformed first draft must not starve the bounded repair of authoritative evidence.
    _, source_ids = assemble_evidence_package(document)
    ids.update(source_ids)
    blocks=[dict(id=b['id'],text=b['text'],page=b['page']) for b in document['blocks'] if b['id'] in ids]
    aliases=[dict(id='asset:'+aid,page=a['original']['page'],text=a['original']['caption'],origin='asset_caption_alias') for aid,a in prepared.items()]
    allowed={b['id'] for b in blocks+aliases}
    packet=dict(title=document['title'],previous_draft_for_revision=notes,visual_mode=visual_mode(len(prepared)),
                source_blocks=blocks+aliases,invalid_previous_source_ids=sorted(ids-allowed),
                available_visuals=[dict(asset_id=aid,caption=a['original']['caption'],
                    caption_source_id='asset:'+aid,aspect=round(a['aspect'],3)) for aid,a in prepared.items()])
    checked=deepcopy(document);checked['blocks']+=aliases
    checked['assets']=[a for a in document['assets'] if a['id'] in prepared]
    return packet,checked,allowed


def normalize_storyboard_structure(candidate):
    """Mechanical schema fixes only; do not rewrite scientific claims."""
    story=deepcopy(candidate);changes=[]
    if not isinstance(story,dict) or not isinstance(story.get('modules'),list):return story,changes
    for i,m in enumerate(story.get('modules',[])):
        if not isinstance(m,dict):continue
        wanted=f'm{i+1}'
        if m.get('id')!=wanted:
            changes.append(dict(kind='module_id',old=m.get('id'),new=wanted));m['id']=wanted
        for v in m.get('visuals',[]) if isinstance(m.get('visuals'),list) else []:
            if isinstance(v,dict) and not v.get('source_ids') and v.get('asset_id'):
                v['source_ids']=['asset:'+v['asset_id']]
                changes.append(dict(kind='caption_alias',asset_id=v['asset_id']))
    return story,changes


def revision_prompt(packet):
    n=len(packet['available_visuals'])
    return '''Rewrite the previous long draft into a VISUAL POSTER, not a paper summary.
The old draft may contain mistakes. The original source blocks and captions are
the ONLY factual authority. Some old figures failed crop preflight: only use
available_visuals. This request is about aggressive editing, not adding coverage.

HARD EDITORIAL BUDGET (verify your own word count before returning):
- 200–260 TOTAL BODY words, never above 270. Title, headings, captions excluded.
- Introduction: ONE bullet, about 25 words.
- Conclusion & Scope: TWO bullets, about 20 words each.
- Choose 2–4 middle modules for the argument, INDEPENDENT of the number of figures.
- Usually TWO bullets per module, ~20–25 words each; THREE only if two modules.
- Every bullet <=30 words. Select core claims; remove secondary dataset numbers
  and architecture minutiae before removing the mechanism or critical caveat.
- Do NOT increase total words to address a review issue: replace existing words.
- At least one method/theory module and one results/analysis module.
- Each middle module may have 0–2 figures. There is NO minimum figure quota.
  Use visuals: [] and asset_ids: [] for text-only modules. Never force an unrelated plot
  into theory. When no images are available, ALL modules are text-only, using grounded
  claims, plain-notation formulas and assumptions, not invented figures or results.
  With one image, use it at most once in the most directly supported module.
  With more images, choose only useful evidence. NEVER repeat a source image.
- Every figure has a meaningful local bullet with asset_ids linking to it.
- Prefer results-bearing plots over decorative/conceptual images.
- Captions: 8–14 words, not a copy of the paper caption.
- Preserve meaning and scope: only empirically supported comparisons, explicit
  crucial theoretical assumptions, core method operations, and an honest caveat.
- Cite source IDs in JSON metadata ONLY, never inline in displayed bullet text.
- Captions may cite the supplied asset:aN caption source. Use only source_blocks IDs.
- Use plain scientific notation, no LaTeX. Do not rewrite authors/title.

Return EXACTLY this object, with all required arrays:
{"intro":{"heading":"Introduction","bullets":[{"text":"...","source_ids":["..."]}]},
"modules":[{"id":"m1","heading":"Specific finding, at most six words","role":"method",
"bullets":[{"text":"...","source_ids":["..."],"asset_ids":["a0"]}],
"visuals":[{"asset_id":"a0","caption":"...","source_ids":["asset:a0"]}]}],
"conclusion":{"heading":"Conclusion & Scope","bullets":[{"text":"...","source_ids":["..."]}]},
"planning_notes":"Why this structure and these figures suit this paper"}

SOURCE PACKET:
''' + json.dumps(packet,ensure_ascii=False) + f'\nThere are {n} preflighted images. FINAL CHECK: body <=260 words; each bullet <=30 words; all visual IDs unique and bound to a local bullet. Return the complete short JSON only.'


def revise_grounded_storyboard(document,notes,prepared,api,output):
    output=Path(output)
    notes,seed_reference_changes=normalize_source_references(notes,document)
    dump_json(output/'seed_source_reference_normalization.json',seed_reference_changes)
    packet,checked,allowed=assemble_revision_evidence(document,notes,prepared)
    dump_json(output/'focused_evidence.json',packet)
    dump_json(output/'document_tree.json',checked)
    history=[];draft=None
    seed, seed_changes = normalize_storyboard_structure(notes)
    seed_errors, seed_warnings = validate_provenance_storyboard(seed, checked, allowed)
    latest = deepcopy(seed) if not seed_errors else None
    chosen = 'previous_draft' if latest is not None else None
    dump_json(output/'seed_check.json',dict(errors=seed_errors,warnings=seed_warnings,
        structural_normalization=seed_changes,usable=latest is not None))
    base=revision_prompt(packet)
    for attempt in range(ATTEMPT_LIMIT):
        if attempt == 2 and latest is not None:
            break
        prompt=base if attempt==0 else base+'''\nGROUNDED CHECK + BOUNDED FEEDBACK REPAIR (AT MOST THREE ATTEMPTS TOTAL):
Inspect the draft below against source_blocks. Fix unsupported numbers, missing
critical mechanism/scope and wrong figure correspondences. Then return JSON
{"review":{"issues":["..."],"scope_preserved":true},"storyboard":<COMPLETE REVISED STORYBOARD>}.
Keep total body 200–260 words, captions <=14, each bullet <=30. Do NOT expand the
story when correcting it. If already correct, keep it unchanged. Structural
errors and word counts below are deterministic; fix them before responding.
'''+json.dumps(dict(draft=draft,checks=history[-1]),ensure_ascii=False)
        try:
            raw=api.json('short_draft' if attempt==0 else f'grounded_check_and_revision_{attempt}',SYSTEM,prompt,max_tokens=4800)
        except (APIError, BudgetExceeded) as exc:
            failure=failure_feedback(exc,api)
            history.append(dict(attempt=attempt,**failure))
            dump_json(output/'writer_history.json',history)
            if not failure['retryable']:break
            continue
        dump_json(output/f'refine_response_{attempt}.json',raw)
        if attempt>=1 and isinstance(raw,dict) and 'storyboard' in raw:
            candidate=raw['storyboard'];dump_json(output/'grounded_review.json',raw.get('review',{}))
        else:candidate=raw
        candidate,reference_changes=normalize_source_references(candidate,checked)
        dump_json(output/f'source_reference_normalization_{attempt}.json',reference_changes)
        draft,changes=normalize_storyboard_structure(candidate)
        errors,warnings=validate_provenance_storyboard(draft,checked,allowed)
        wc=sum(word_count(b['text']) for b in body_bullets(draft)) if not errors else None
        history.append(dict(attempt=attempt,errors=errors,warnings=warnings,body_words=wc,structural_normalization=changes,
                            reference_errors=source_reference_errors(draft,allowed)))
        dump_json(output/f'candidate_{attempt}.json',draft)
        if not errors:
            latest=draft;chosen=f'candidate_{attempt}';dump_json(output/'storyboard.json',latest)
    dump_json(output/'writer_history.json',history)
    if latest is None:raise ValueError('No usable grounded draft after bounded refinement; preserve attempts for diagnosis')
    dump_json(output/'storyboard.json',latest)
    _,warnings=validate_provenance_storyboard(latest,checked,allowed)
    dump_json(output/'completion.json',dict(policy='last_usable_candidate',repairs_max=2,requests_max=3,requests_used=len(history),
        warnings=warnings,chosen=chosen,visual_mode=visual_mode(len(prepared)),grounded_review_independent=False,last_usable_is_final_attempt=latest==draft,
        source_anchor_note='asset:aN explicitly resolves to original asset caption, not invented PDF block ID'))
    return latest,checked
