"""Evidence-aware rectangular packing with independently sized modules.

A bounded dynamic program optimizes guillotine partitions, local text placement
and figure arrangements. It uses source-label size proxies rather than counting
figures or forcing all modules in a row to share a height. No text is removed.
"""
from functools import lru_cache
from dataclasses import replace
import math
from dagent.core.layout.geometry import Element, Layout, text_height
from dagent.core.layout.layout_planning import NAVY, TEAL, INK, SLATE, WHITE, compositions, audit_layout

VERSION = 'dagent_layout_safety'


def make_layout(story, document, assets, cfg, grid=24, preferences=None, text_safety=None):
    text_safety = text_safety or {}
    def measured_height(eid, text, width, pt, bold=False):
        repairs = min(2, max(0, text_safety.get(eid, 0)))
        base = text_height(text, width - .12 * repairs, pt, bold)
        # Actual failing glyph targets get extra allocated space, never clipped
        # or made invisible. Repack and re-run every geometry/render check.
        return base * (1 + .12 * repairs) + .10 * repairs
    width, height, margin = cfg.width_inches, cfg.height_inches, cfg.margin_inches
    usable, gap = width - 2*margin, .48
    modules = story['modules']; body_font = 40; caption_font = 26
    has_visuals = any(m['visuals'] for m in modules)
    if not has_visuals:
        body_font = 48  # Source-only theory cards, not empty figure placeholders.
    preferences = preferences or {}
    letters = {v['asset_id']: chr(65+i) for i, v in enumerate(v for m in modules for v in m['visuals'])}
    elements = []
    def item(eid, kind, x, y, w, h, text='', pt=40, bold=False, fill=None, color=INK, sid='', refs=(), path=None):
        return Element(eid, kind, x, y, w, h, text, pt, path, list(refs), sid, fill, color, bold)
    title_pt = 64
    th = measured_height('title', document['title'], usable-.36, title_pt, True)+.34
    if th > 3.4:
        title_pt=57; th=measured_height('title', document['title'], usable-.36, title_pt, True)+.34
    elements.append(item('title','title',margin,margin,usable,th,document['title'],title_pt,True,NAVY,WHITE))
    authors=document['authors']; authors=', '.join(str(a) for a in authors) if isinstance(authors,list) else authors
    ay=margin+th+.12; ah=measured_height('authors',authors,usable,27)+.06
    elements.append(item('authors','authors',margin,ay,usable,ah,authors,27,color=SLATE))
    def anchor_size(section,sid):
        bw=(usable-gap*(len(section['bullets'])-1))/len(section['bullets'])
        return .86+max(measured_height(f'{sid}_b{i}','• '+b['text'],bw,36)+.10 for i,b in enumerate(section['bullets']))
    ih,ch=anchor_size(story['intro'],'intro'),anchor_size(story['conclusion'],'conclusion')
    iy=ay+ah+.26; cy=height-margin-ch; top=iy+ih+.34; available=cy-.34-top
    def anchor(section,sid,label,y):
        elements.append(item(sid+'_heading','heading',margin,y,usable,.72,label,34,True,TEAL,WHITE,sid))
        bw=(usable-gap*(len(section['bullets'])-1))/len(section['bullets'])
        for i,b in enumerate(section['bullets']):
            text='• '+b['text'].lstrip('• ').strip(); bh=measured_height(f'{sid}_b{i}',text,bw,36)+.10
            elements.append(item(f'{sid}_b{i}','text',margin+i*(bw+gap),y+.86,bw,bh,text,36,sid=sid,refs=b['source_ids']))
    anchor(story['intro'],'intro','Introduction',iy)
    anchor(story['conclusion'],'conclusion','Conclusion & Scope',cy)
    if available<10: raise ValueError('Insufficient canvas; no content deletion allowed')
    physical_w=lambda units:(usable+gap)*units/grid-gap
    physical_h=lambda units:(available+gap)*units/grid-gap
    visits=0

    def image_target(aid):
        a=assets[aid]; profile=a.get('label_profile',{})
        target_pt=24 if a['original'].get('kind')=='table' or any(v['asset_id']==aid for m in modules if m['role'] in ('results','analysis') for v in m['visuals']) else 20
        if profile.get('points_per_inch'):
            return max(8,min(42,target_pt/profile['points_per_inch']))
        return profile.get('target_width_inches',15)

    @lru_cache(None)
    def configs(index, wu):
        module=modules[index]; w=physical_w(wu)
        hh=measured_height(module['id']+'_heading',f'{index+1:02d} '+module['heading'],w,36,True)+.08
        options=[]
        places=[('top',1),('side',.27),('side',.35),('side',.43)] if module['visuals'] and w>=17 else [('top',1)]
        if w>=25 and len(module['bullets'])>1: places.append(('top',2))
        if not module['visuals'] and w>=34 and len(module['bullets'])>=3: places.append(('top',3))
        for mode,param in places:
            tw=w*param if mode=='side' else (w-.38*(param-1))/param
            iw=w-tw-.42 if mode=='side' else w
            if mode=='side' and (tw<6.5 or iw<8):continue
            bullets=[]; col_heights=[0.]*(1 if mode=='side' else int(param))
            for bi,b in enumerate(module['bullets']):
                label=', '.join(letters[a] for a in b.get('asset_ids',[]))
                text='• '+b['text'].lstrip('• ').strip()+(f' ({label})' if label else '')
                bh=measured_height(f'{module["id"]}_b{bi}',text,tw,body_font)+.08
                col=0 if mode=='side' else min(int(param)-1,int(bi*int(param)/len(module['bullets'])))
                bullets.append((bi,col*(tw+.38),col_heights[col],tw,bh,text,b))
                col_heights[col]+=bh+.06
            bh=max(col_heights,default=0)
            figure_top=hh+.15 if mode=='side' else hh+.15+bh+.18
            for arrangement in compositions(len(module['visuals'])) if module['visuals'] else [[]]:
                groups=[];start=0
                for count in arrangement:
                    vs=module['visuals'][start:start+count];start+=count
                    # Compare aspect-equal and label-prioritized sizing below.
                    groups.append(vs)
                for weighted in ([False,True] if any(n>1 for n in arrangement) else [False]):
                    rows=[]
                    for group in groups:
                        weights=[image_target(v['asset_id']) if weighted else assets[v['asset_id']]['aspect'] for v in group]
                        total=iw-.38*(len(group)-1)
                        widths=[total*q/sum(weights) for q in weights]
                        rows.append((group,widths))
                    options.append(dict(header=hh,bullets=bullets,text_height=bh,text_width=tw,
                                        image_x=tw+.42 if mode=='side' else 0, image_top=figure_top,
                                        image_width=iw,mode=mode,rows=rows,weighted=weighted,columns=param if mode=='top' else 1))
        return options

    def image_geometry(option,scale):
        rows=[];height=0.
        for group,widths in option['rows']:
            boxes=[];x=0.
            for v,slot in zip(group,widths):
                aid=v['asset_id'];a=assets[aid];fw=slot*scale;fh=fw/a['aspect']
                caption=letters[aid]+'  '+v['caption']
                ch=measured_height(aid+'_caption',caption,slot,caption_font)+.06
                boxes.append((v,x+(slot-fw)/2,height,fw,fh,x,slot,ch,caption))
                x+=slot+.38
            rh=max((b[4]+.12+b[7] for b in boxes),default=0)
            rows.extend(boxes);height+=rh+.23
        return max(0,height-.23),rows

    def resolution_limited_scale(option):
        """Keep raster figures at or above the configured effective DPI."""
        minimum_dpi = float(getattr(cfg, 'minimum_image_dpi', 0) or 0)
        if minimum_dpi <= 0:
            return 1.0
        scale = 1.0
        for group, widths in option['rows']:
            for v, slot in zip(group, widths):
                asset = assets[v['asset_id']]
                pixels = asset.get('pixels') or ()
                if len(pixels) < 2:
                    continue
                try:
                    pixel_width, pixel_height = float(pixels[0]), float(pixels[1])
                    aspect = float(asset['aspect'])
                except (KeyError, TypeError, ValueError):
                    continue
                if pixel_width <= 0 or pixel_height <= 0 or slot <= 0 or aspect <= 0:
                    continue
                # Apply a small numerical headroom so the audit does not reject a
                # candidate on a DPI value that rounds to the threshold.
                headroom = .995
                scale = min(scale, headroom * pixel_width / (minimum_dpi * slot))
                intrinsic_height = slot / aspect
                scale = min(scale, headroom * pixel_height /
                            (minimum_dpi * intrinsic_height))
        return min(1.0, scale)

    @lru_cache(None)
    def leaf(index,wu,hu):
        nonlocal visits
        visits+=1; w,h=physical_w(wu),physical_h(hu)
        if w<8 or h<1.3:return None
        module=modules[index];best=None
        preference=preferences.get(module['id'], {})
        for option in configs(index,wu):
            if option['header']+.15+option['text_height']>h-.03:continue
            remaining=h-option['image_top']
            if not module['visuals']:
                # A theory card must be compact rather than a blank pseudo-image.
                if has_visuals and h-(option['header']+.15+option['text_height'])>2.4:continue
                scale=1; image_h=0;boxes=[]
            else:
                if remaining<1.5:continue
                image_h,boxes=image_geometry(option,1)
                scale=resolution_limited_scale(option)
                if scale<.08:continue
                image_h,boxes=image_geometry(option,scale)
                if image_h>remaining:
                    lo,hi=.08,scale
                    for _ in range(9):
                        mid=(lo+hi)/2
                        if image_geometry(option,mid)[0]<=remaining:lo=mid
                        else:hi=mid
                    scale=lo;image_h,boxes=image_geometry(option,scale)
                if image_h>remaining+.002 or any(b[3]<3 or b[4]<1.25 for b in boxes):continue
            text_area=sum(b[3]*b[4] for b in option['bullets'])
            visual_area=sum(b[3]*b[4] for b in boxes)
            cap_area=sum(b[6]*b[7] for b in boxes)
            quality=0
            for v,_,_,fw,fh,*_ in boxes:
                q=fw/image_target(v['asset_id'])
                quality+=(10*min(1,q)+min(1.2,q)-6*max(0,.7-q)**2)*preference.get('visual_priority',1)
            occupied=option['header']*w+text_area+visual_area+cap_area
            score=quality+.009*occupied-.003*max(0,w*h-occupied)
            # Discourage long unused sidecar columns without compressing prose.
            if option['mode']=='side':
                score-=.008*option['text_width']*max(0,image_h-option['text_height']-1)
            if preference.get('placement')==option['mode']:
                score+=4
            if best is None or score>best['score']:
                best=dict(score=score,option=option,boxes=boxes,scale=scale,w=w,h=h,
                          occupied=occupied,visual_area=visual_area,quality=quality)
        return best

    @lru_cache(None)
    def solve(lo,hi,wu,hu):
        if hi-lo==1:
            fit=leaf(lo,wu,hu)
            return (fit['score'],('leaf',lo,wu,hu)) if fit else None
        best=None
        for mid in range(lo+1,hi):
            # Integer grid keeps the search finite and allows unequal module heights.
            for direction,extent,minimum in [('vertical',wu,5),('horizontal',hu,2)]:
                for first in range(minimum,extent-minimum+1):
                    if direction=='vertical':
                        a=solve(lo,mid,first,hu);b=solve(mid,hi,wu-first,hu)
                    else:
                        a=solve(lo,mid,wu,first);b=solve(mid,hi,wu,hu-first)
                    if a is None or b is None:continue
                    score=a[0]+b[0]-.012
                    if best is None or score>best[0]:
                        best=(score,(direction,first,a[1],b[1]))
        return best
    solution=solve(0,len(modules),grid,grid)
    if solution is None:raise ValueError('No complete layout at readable text size; retain previous render')
    regions=[];bindings=[]
    def emit(tree,x,y):
        if tree[0]!='leaf':
            direction,first,a,b=tree;emit(a,x,y)
            emit(b,x+(physical_w(first)+gap if direction=='vertical' else 0),
                   y+(physical_h(first)+gap if direction=='horizontal' else 0));return
        _,index,wu,hu=tree;fit=leaf(index,wu,hu);option=fit['option'];m=modules[index];mid=m['id'];w=fit['w']
        elements.append(item(mid+'_heading','heading',x,y,w,option['header'],f'{index+1:02d} '+m['heading'],36,True,NAVY,WHITE,mid))
        for bi,bx,by,bw,bh,text,b in option['bullets']:
            elements.append(item(f'{mid}_b{bi}','text',x+bx,y+option['header']+.15+by,bw,bh,text,body_font,sid=mid,refs=b['source_ids']))
            bindings.append(dict(bullet_id=f'{mid}_b{bi}',asset_ids=b.get('asset_ids',[]),module_id=mid))
        for v,bx,by,fw,fh,cx,cw,ch,caption in fit['boxes']:
            aid=v['asset_id'];fx=x+option['image_x'];fy=y+option['image_top']
            elements.append(item(aid+'_image','figure',fx+bx,fy+by,fw,fh,sid=mid,refs=v['source_ids'],path=assets[aid]['path']))
            elements.append(item(aid+'_caption','caption',fx+cx,fy+by+fh+.12,cw,ch,caption,caption_font,color=SLATE,sid=mid,refs=v['source_ids']))
        regions.append(dict(module_id=mid,x=x,y=y,width=w,height=fit['h'],text_position=option['mode'],
                            text_columns=option['columns'],figure_scale=fit['scale'],
                            occupied_area=fit['occupied'],unused_area=max(0,w*fit['h']-fit['occupied'])))
    emit(solution[1],margin,top)
    layout=Layout(width,height,elements,columns=2)
    return layout,dict(version=VERSION,body_font_pt=body_font,grid=grid,search_states=solve.cache_info().currsize,
                       leaf_candidates=visits,score=solution[0],regions=regions,partition=solution[1],
                       no_content_deletion=True,figure_letters=letters,bindings=bindings,
                       text_only_modules=[m['id'] for m in modules if not m['visuals']],
                       visual_mode='with_source_visuals' if has_visuals else 'text_only',
                       source_label_sizes_are_proxies=True,independent_semantic_evaluation=False,
                       applied_preferences=preferences,text_safety_repairs=text_safety)
