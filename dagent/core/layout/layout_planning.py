"""Figure-first, bounded deterministic topology search for atomic story modules.

No prose/figure deletion, no image distortion, no arbitrary raster enlargement.
Real source PDF crops are prepared by the existing guarded visual pipeline.
"""
from __future__ import annotations

from copy import deepcopy
import itertools
import math
from pathlib import Path

from PIL import Image

from dagent.core.config import dump_json
from dagent.core.layout.geometry import Element, Layout, text_height
from dagent.core.layout.guarded_visuals import prepare_guarded_visuals

VERSION = 'poster_module_layout'
NAVY = [22, 46, 66]
TEAL = [0, 99, 112]
INK = [28, 44, 58]
SLATE = [72, 90, 104]
WHITE = [255, 255, 255]


def prepare_assets(story, document, output, cfg):
    """Reuse guarded crop recovery, not the flawed cached extraction pixels."""
    by_id = {a['id']: a for a in document['assets']}
    children = []
    for m in story['modules']:
        nodes = []
        for v in m['visuals']:
            a = by_id[v['asset_id']]
            nodes.append(dict(id=a['id'], panel_id=a['id'], node_type='figure',
                              kind=a['kind'], asset_path=a['path'], content=a['caption'],
                              source_ids=v['source_ids'], figure_aspect=a['pixel_width']/a['pixel_height']))
        children.append(dict(id=m['id'], children=nodes))
    tree = dict(children=children, source_pdf_sha256=document['pdf_sha256'])
    updated, audit = prepare_guarded_visuals(
        tree, document, Path(output), target_width_inches=cfg.width_inches-2*cfg.margin_inches,
        minimum_dpi=cfg.minimum_image_dpi)
    assets = {}
    for section in updated['children']:
        for node in section['children']:
            with Image.open(node['asset_path']) as image:
                size = image.size
            aid = node['panel_id']
            assets[aid] = dict(path=node['asset_path'], aspect=size[0]/size[1], pixels=list(size),
                               original=deepcopy(by_id[aid]), node=node)
    dump_json(Path(output)/'prepared_assets.json', assets)
    return assets


def compositions(n):
    if n == 0:
        yield []
    for k in range(1, min(n, 3)+1):
        for rest in compositions(n-k):
            yield [k] + rest


def figure_arrangement(module, assets, width):
    figs = module['visuals']
    if len(figs) == 1:
        return 'single', [width], width/assets[figs[0]['asset_id']]['aspect']
    # Joint aspect-ratio allocation: common image height, not stretched pixels.
    aspects = [assets[v['asset_id']]['aspect'] for v in figs]
    available = width - .38*(len(figs)-1)
    common = available / sum(aspects)
    widths = [common*a for a in aspects]
    if min(widths) >= 7 and common >= 2.8 and not (width < 30 and len(figs)==2 and any(assets[v['asset_id']].get('original',{}).get('kind')=='table' for v in figs)):
        return 'side_by_side', widths, common
    return 'stacked', [width]*len(figs), sum(width/a for a in aspects)


def metrics(module, assets, width, body_font, letters, sidecar=False):
    heading = text_height('01  '+module['heading'], width, 37, True) + .10
    text_width = min(15.5, max(8.0, width*.34)) if sidecar else width
    image_width = width-text_width-.45 if sidecar else width
    bullets = []
    for b in module['bullets']:
        links = ', '.join(letters[a] for a in b.get('asset_ids', []))
        text = '• '+b['text'].lstrip('• ').strip() + (f' ({links})' if links else '')
        bullets.append((text, text_height(text, text_width, body_font)+.09, b))
    mode, widths, ideal_image_height = figure_arrangement(module, assets, image_width)
    caps = [text_height(letters[v['asset_id']]+'  '+v['caption'], w, 26)+.07
            for v, w in zip(module['visuals'], widths)]
    cap_height = max(caps) if mode=='side_by_side' else sum(caps)
    text_height_total = heading+.18+sum(h for _,h,_ in bullets)+.20
    fixed = heading + .18 + cap_height + .22
    if not sidecar:
        fixed += sum(h for _,h,_ in bullets) + .20
    if mode == 'stacked':
        fixed += .38*(len(caps)-1)
    return dict(heading=heading, bullets=bullets, mode=mode, widths=widths,
                fixed=fixed, caps=caps, ideal=ideal_image_height, sidecar=sidecar,
                text_width=text_width, image_width=image_width, text_height=text_height_total)


def make_layout(story, document, assets, cfg):
    width, height, margin = cfg.width_inches, cfg.height_inches, cfg.margin_inches
    usable, gap = width-2*margin, .60
    visuals = [v['asset_id'] for m in story['modules'] for v in m['visuals']]
    letters = {aid: chr(65+i) for i,aid in enumerate(visuals)}
    if len(visuals) > 26:
        raise ValueError('Too many visuals for figure bindings')
    elements = []
    def add(id, kind, x, y, w, h, text='', pt=40, bold=False, fill=None, color=INK, sid='', sources=None, path=None):
        elements.append(Element(id,kind,x,y,w,h,text,pt,path,list(sources or []),sid,fill,color,bold))
    title_pt = 64
    th = text_height(document['title'], usable-.36, title_pt, True)+.34
    if th>3.4:
        title_pt=57; th=text_height(document['title'],usable-.36,title_pt,True)+.34
    add('title','title',margin,margin,usable,th,document['title'],title_pt,True,NAVY,WHITE)
    ay=margin+th+.12
    authors=document['authors']
    if isinstance(authors,list):
        authors=', '.join(str(a) for a in authors)
    ah=text_height(authors,usable,27)+.06
    add('authors','authors',margin,ay,usable,ah,authors,27,color=SLATE)

    def anchor_size(section):
        count=len(section['bullets']); bw=(usable-gap*(count-1))/count
        return .72+.14+max(text_height('• '+b['text'],bw,36)+.10 for b in section['bullets'])
    ih=anchor_size(story['intro']); ch=anchor_size(story['conclusion'])
    iy=ay+ah+.26; cy=height-margin-ch
    top=iy+ih+.45; available=cy-.45-top
    if available<10:
        raise ValueError('Insufficient canvas for grounded story; preserve text, do not silently truncate')

    def anchor(section, sid, label, y):
        add(sid+'_heading','heading',margin,y,usable,.72,label,34,True,TEAL,WHITE,sid=sid)
        count=len(section['bullets']); bw=(usable-gap*(count-1))/count
        for i,b in enumerate(section['bullets']):
            text='• '+b['text'].lstrip('• ').strip(); bh=text_height(text,bw,36)+.10
            add(f'{sid}_b{i}','text',margin+i*(bw+gap),y+.86,bw,bh,text,36,sid=sid,sources=b['source_ids'])
    anchor(story['intro'],'intro','Introduction',iy)
    anchor(story['conclusion'],'conclusion','Conclusion & Scope',cy)

    candidates=[]; modules=story['modules']
    for font in (42, 40, 38, 36):
        for counts in compositions(len(modules)):
            if len(counts)>3:
                continue
            for weighted in (False, True):
                for orientations in itertools.product((False, True), repeat=len(modules)):
                    rows=[]; start=0
                    for count in counts:
                        group=modules[start:start+count]; modes=orientations[start:start+count]; start+=count
                        weights=[min(1.45,max(.75,math.sqrt(sum(assets[v['asset_id']]['aspect'] for v in m['visuals'])/2)))
                                 if weighted else 1 for m in group]
                        widths=[(usable-gap*(count-1))*v/sum(weights) for v in weights]
                        ms=[metrics(m,assets,w,font,letters,sidecar=mode) for m,w,mode in zip(group,widths,modes)]
                        rows.append(dict(modules=group,widths=widths,metrics=ms))
                    if any(m['sidecar'] and w<21 for r in rows for m,w in zip(r['metrics'],r['widths'])):
                        continue
                    row_space=available-gap*(len(rows)-1)
                    bases=[max(max(m['text_height'],m['fixed']+min(3.5,m['ideal'])) for m in r['metrics']) for r in rows]
                    if sum(bases)>row_space:
                        continue
                    desires=[max(max(m['text_height'],m['fixed']+m['ideal']) for m in r['metrics']) for r in rows]
                    extra=row_space-sum(bases); want=sum(max(0,d-b) for d,b in zip(desires,bases))
                    heights=[b+extra*(max(0,d-b)/want if want else 1/len(rows)) for b,d in zip(bases,desires)]
                    area=0; scales=[]; minimum_dims=[]; legibility=[]
                    for row,rh in zip(rows,heights):
                        row['height']=rh
                        for module,m in zip(row['modules'],row['metrics']):
                            scale=min(1,(rh-m['fixed'])/m['ideal'])
                            scales.append(scale)
                            for v,fw in zip(module['visuals'],m['widths']):
                                aspect=assets[v['asset_id']]['aspect']; fw=fw*scale; fh=fw/aspect
                                area+=fw*fh; minimum_dims.append(min(fw,fh))
                                original=assets[v['asset_id']].get('original',{})
                                caption=original.get('caption','').lower()
                                target=28 if original.get('kind')=='table' else (24 if any(k in caption for k in ('plot','accuracy','result','error','estimat')) else 18)
                                legibility.append(min(1,fw/target))
                    # Prioritize real visual area, readable text and balanced use of canvas.
                    score=.45*area/(usable*available)+.30*sum(legibility)/len(legibility)+.15*min(legibility)+.006*(font-36)
                    if min(minimum_dims)<2.5:
                        score-=.4*(2.5-min(minimum_dims))
                    candidates.append(dict(score=score,rows=rows,font=font,counts=counts,weighted=weighted,orientations=list(orientations),legibility_proxy=sum(legibility)/len(legibility),
                                           figure_area=area,min_figure_dimension=min(minimum_dims)))
    if not candidates:
        raise ValueError('No readable topology fits; no zero-sized figure fallback')
    best=max(candidates,key=lambda c:c['score'])
    y=top; index=0; binding=[]
    for row in best['rows']:
        x=margin; rh=row['height']
        for module,w,m in zip(row['modules'],row['widths'],row['metrics']):
            index+=1; mid=module['id']; my=y
            add(mid+'_heading','heading',x,my,w,m['heading'],f'{index:02d}  '+module['heading'],37,True,NAVY,WHITE,sid=mid)
            my+=m['heading']+.18
            for bi,(text,bh,b) in enumerate(m['bullets']):
                add(f'{mid}_b{bi}','text',x,my,m['text_width'],bh,text,best['font'],sid=mid,sources=b['source_ids']);my+=bh
                binding.append(dict(bullet_id=f'{mid}_b{bi}',asset_ids=b.get('asset_ids',[]),module_id=mid))
            my+=.20
            scale=min(1,(rh-m['fixed'])/m['ideal']); fig_x=x
            if m['sidecar']:
                my=y+m['heading']+.18;fig_x=x+m['text_width']+.45
            for vi,(v,fw,chh) in enumerate(zip(module['visuals'],m['widths'],m['caps'])):
                aid=v['asset_id']; asset=assets[aid]; imw=fw*scale; imh=imw/asset['aspect']
                actual_x=fig_x+(fw-imw)/2
                add(aid+'_image','figure',actual_x,my,imw,imh,sid=mid,sources=v['source_ids'],path=asset['path'])
                cap=letters[aid]+'  '+v['caption']
                cap_y=my+imh+.12
                add(aid+'_caption','caption',fig_x,cap_y,fw,chh,cap,26,color=SLATE,sid=mid,sources=v['source_ids'])
                if m['mode']=='side_by_side':
                    fig_x+=fw+.38
                else:
                    my=cap_y+chh+.24
            x+=w+gap
        y+=rh+gap
    layout=Layout(width,height,elements,columns=max(best['counts']))
    report=dict(version=VERSION,search_candidates=len(candidates),rows=best['counts'],
                weighted_widths=best['weighted'],body_font_pt=best['font'],
                module_sidecar_modes=best['orientations'],legibility_width_proxy=best['legibility_proxy'],
                body_visual_area_fraction=best['figure_area']/(usable*available),
                canvas_visual_area_fraction=best['figure_area']/(width*height),
                min_figure_dimension=best['min_figure_dimension'],bindings=binding,
                no_content_deletion=True,figure_letters=letters)
    return layout,report


def audit_layout(story, layout, assets, cfg):
    """Geometry is checked independently of old fixed-column topology rules."""
    from dataclasses import replace
    from dagent.core.layout.verifier import verify
    report=verify(layout,replace(cfg,design='classic'))
    missing=[]
    for m in story['modules']:
        for v in m['visuals']:
            image=layout.get(v['asset_id']+'_image');caption=layout.get(v['asset_id']+'_caption')
            if not image or not caption or image.section_id!=m['id'] or caption.section_id!=m['id']:
                missing.append(v['asset_id'])
        for bi,_ in enumerate(m['bullets']):
            if not layout.get(f'{m["id"]}_b{bi}'):
                missing.append(f'{m["id"]}_b{bi}')
    report['missing_atoms']=missing
    report['source_fidelity_note']='Guarded PDF crops preserve original graphics; linkage is writer-reviewed, not independent semantic evaluation.'
    report['usable']=not missing
    return report
