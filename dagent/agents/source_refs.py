"""Exact asset-ID namespace normalization, never fuzzy evidence substitution."""
from copy import deepcopy


def caption_aliases(document):
    return [dict(id='asset:'+a['id'],page=a['page'],text=a['caption'],origin='asset_caption_alias')
            for a in document['assets'] if a.get('caption','').strip()]


def normalize_source_references(candidate, document):
    result=deepcopy(candidate); changes=[]
    blocks={b['id'] for b in document['blocks']}
    aliases={a['id']:'asset:'+a['id'] for a in document['assets'] if a.get('caption','').strip()}
    if not isinstance(result,dict):return result,changes
    modules=result.get('modules',[])
    if not isinstance(modules,list):return result,changes
    sections=[result.get('intro'),*modules,result.get('conclusion')]
    for si,section in enumerate(sections):
        if not isinstance(section,dict):continue
        for kind in ('bullets','visuals'):
            entries=section.get(kind,[])
            if not isinstance(entries,list):continue
            for index,entry in enumerate(entries):
                if not isinstance(entry,dict) or not isinstance(entry.get('source_ids'),list):continue
                resolved=[]
                for sid in entry['source_ids']:
                    # If this exact ID is a real text block, keep its original meaning.
                    mapped=aliases.get(sid,sid) if isinstance(sid,str) and sid not in blocks else sid
                    if mapped!=sid:
                        changes.append(dict(section_index=si,kind=kind,index=index,original=sid,
                                            canonical=mapped,reason='exact_known_asset_caption_namespace'))
                    if mapped not in resolved:resolved.append(mapped)
                entry['source_ids']=resolved
    return result,changes


def source_reference_errors(candidate, allowed):
    if not isinstance(candidate,dict) or not isinstance(candidate.get('modules',[]),list):return []
    errors=[]
    for si,section in enumerate([candidate.get('intro'),*candidate.get('modules',[]),candidate.get('conclusion')]):
        if not isinstance(section,dict):continue
        for kind in ('bullets','visuals'):
            entries=section.get(kind,[])
            if not isinstance(entries,list):continue
            for i,entry in enumerate(entries):
                if not isinstance(entry,dict) or not isinstance(entry.get('source_ids'),list):continue
                unknown=[s for s in entry['source_ids'] if not isinstance(s,str) or s not in allowed]
                if unknown:errors.append(dict(section_index=si,kind=kind,index=i,unknown_source_ids=unknown))
    return errors
