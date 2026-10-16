"""Source-preserving column composition with section continuation.

A long method/results section can cross a column boundary at a content-block
boundary. A figure and its caption are indivisible. Dynamic programming minimizes
the maximum occupied column height and preserves scientific reading order.
"""
from __future__ import annotations
from dagent.core.layout.geometry import Element, Layout, text_height
NAVY = [20, 44, 68]
TEAL = [0, 111, 119]
INK = [29, 43, 56]


def balanced_layout(tree, cfg):
    w,h,m,gap = cfg.width_inches,cfg.height_inches,cfg.margin_inches,cfg.gap_inches
    cw=(w-2*m-(cfg.columns-1)*gap)/cfg.columns
    title_font=min(78,w*1.6)
    title_h=max(2.,text_height(tree['title'],w-2*m,title_font,True)+.3)
    header=[Element('title','title',m,m,w-2*m,title_h,tree['title'],title_font,
                    fill=NAVY,color=[255,255,255],bold=True)]
    top=m+title_h+gap
    if tree.get('authors'):
        size=max(cfg.minimum_font_pt,30)
        ah=text_height(tree['authors'],w-2*m,size)+.15
        header.append(Element('authors','authors',m,top,w-2*m,ah,tree['authors'],size,color=INK))
        top+=ah+gap
    available=h-m-top
    sections=tree['children']

    def specs(body_size,figure_scale):
        units=[]
        heading_size=max(36,min(48,body_size*1.12))
        caption_size=max(cfg.minimum_font_pt,min(32,body_size*.75))
        for i,section in enumerate(sections):
            sid=section['panel_id']
            for child in section['children']:
                elements=[]
                if child.get('asset_path'):
                    aspect=max(.05,child['figure_aspect'])
                    fw=min(cw,child['figure_size'][0]/cfg.minimum_image_dpi,available*.43*aspect)*figure_scale
                    elements.append(Element(child['panel_id']+'_image','figure',0,0,fw,fw/aspect,
                        asset_path=child['asset_path'],source_ids=child['source_ids'],section_id=sid))
                    elements.append(Element(child['panel_id']+'_caption','caption',0,0,cw,
                        text_height(child['content'],cw,caption_size)+.12,child['content'],caption_size,
                        source_ids=child['source_ids'],section_id=sid,color=INK))
                else:
                    elements.append(Element(child['panel_id'],'text',0,0,cw,
                        text_height(child['content'],cw,body_size,spacing=1.17)+.12,
                        child['content'],body_size,source_ids=child['source_ids'],section_id=sid,
                        color=INK,line_spacing=1.17))
                units.append(dict(section=i,sid=sid,label=section['section_name'],elements=elements,
                                  height=sum(e.height for e in elements)+gap*(len(elements)-1)))
        def heading(index,continuation=False,col=0):
            unit=units[index]
            text=f"{unit['section']+1:02d}  {unit['label']}"+(' · continued' if continuation else '')
            return Element(unit['sid']+(f'_cont{col}' if continuation else ''),'heading',0,0,cw,
                           text_height(text,cw,heading_size,True)+.20,text,heading_size,
                           section_id=unit['sid'],fill=TEAL,color=[255,255,255],bold=True)
        costs={}
        for start in range(len(units)):
            height=0
            for end in range(start,len(units)):
                if end==start or units[end]['sid']!=units[end-1]['sid']:
                    height+=heading(end, end==start and start>0 and units[start]['sid']==units[start-1]['sid']).height+gap
                    if end>start:
                        height+=gap*.8
                height+=units[end]['height']+gap
                costs[start,end+1]=height-gap
        n=len(units);columns=min(cfg.columns,max(1,n))
        dp={(0,0):(0,[])}
        for k in range(1,columns+1):
            for end in range(k,n+1):
                choices=[]
                for start in range(k-1,end):
                    if (k-1,start) in dp:
                        prior,groups=dp[k-1,start]
                        choices.append((max(prior,costs[start,end]),groups+[(start,end)]))
                dp[k,end]=min(choices,key=lambda x:x[0])
        max_height,groups=dp[columns,n] if n else (0,[])
        return units,heading,costs,groups,max_height

    best=None
    for scale in (1.,.9,.8,.7):
        for size in range(int(min(48,max(cfg.body_font_pt,44))),int(cfg.minimum_font_pt)-1,-1):
            candidate=specs(size,scale)
            if candidate[-1]<=available:
                best=candidate;break
        if best is not None:break
    units,heading,costs,groups,_=best if best is not None else specs(cfg.minimum_font_pt,.7)
    elements=list(header)
    for col,(start,end) in enumerate(groups):
        y=top;x=m+col*(cw+gap)
        # Distribute only a small amount of spare room to actual content intervals.
        extra=min(.20,max(0,available-costs[start,end])/max(1,end-start-1))
        for j in range(start,end):
            if j==start or units[j]['sid']!=units[j-1]['sid']:
                if j>start:y+=gap*.8
                e=heading(j,j==start and start>0 and units[start]['sid']==units[start-1]['sid'],col)
                e.x,e.y=x,y;elements.append(e);y+=e.height+gap
            for e in units[j]['elements']:
                e.x=x+((cw-e.width)/2 if e.kind=='figure' else 0);e.y=y
                elements.append(e);y+=e.height+gap
            if j<end-1:y+=extra
    return Layout(w,h,elements,cfg.columns)


def reflow_balanced(layout, cfg):
    """Repair typed edits without losing section topology or detaching captions.

    Scientific blocks retain their canonical list order. Geometry edits may change
    size/typography, but cannot leave a heading behind in a different column.
    """
    from copy import deepcopy
    from dagent.core.layout.verifier import verify
    result = deepcopy(layout)
    headers = [e for e in result.elements if e.kind in ('title', 'authors')]
    prototypes = {}
    for e in result.elements:
        if e.kind == 'heading':
            prototypes.setdefault(e.section_id, e)
    # Do not invent sections for malformed or externally supplied layouts.
    body = [e for e in result.elements if e.kind not in ('title', 'authors', 'heading')]
    if not body or any(e.section_id not in prototypes for e in body):
        return result
    cw = (layout.width - 2*cfg.margin_inches - (cfg.columns-1)*cfg.gap_inches)/cfg.columns
    gap = cfg.gap_inches
    top = max((e.y+e.height for e in headers), default=cfg.margin_inches)+gap
    available = layout.height-cfg.margin_inches-top
    paired = set()
    units = []
    for e in body:
        if e.id in paired:
            continue
        group = [e]
        if e.kind == 'figure':
            caption = next((x for x in body if x.id == e.id.removesuffix('_image')+'_caption'), None)
            if caption is not None:
                group.append(caption); paired.add(caption.id)
        units.append(group)
    for attempt in range(12):
        for group in units:
            for e in group:
                if e.kind == 'figure':
                    from PIL import Image
                    with Image.open(e.asset_path) as im:
                        e.width = min(e.width, cw, im.width/cfg.minimum_image_dpi)
                        e.height = e.width*im.height/im.width
                else:
                    e.width = cw
                    e.font_pt = max(cfg.minimum_font_pt, e.font_pt)
                    e.height = text_height(e.text, cw, e.font_pt, e.bold, e.line_spacing)+.12
        def make_heading(index, continuation=False, column=0):
            sid = units[index][0].section_id
            e = deepcopy(prototypes[sid]);e.id=sid+(f'_cont{column}' if continuation else '')
            e.text = e.text.removesuffix(' · continued')+(' · continued' if continuation else '')
            e.width = cw;e.height = text_height(e.text,cw,e.font_pt,True)+.20
            return e
        costs={};n=len(units);columns=min(cfg.columns,n)
        for start in range(n):
            height=0
            for end in range(start,n):
                if end==start or units[end][0].section_id!=units[end-1][0].section_id:
                    height+=make_heading(end,end==start and start>0 and units[start][0].section_id==units[start-1][0].section_id).height+gap
                    if end>start:height+=gap*.8
                height+=sum(e.height+gap for e in units[end])
                costs[start,end+1]=height-gap
        dp={(0,0):(0,[])}
        for k in range(1,columns+1):
            for end in range(k,n+1):
                choices=[(max(dp[k-1,start][0],costs[start,end]),dp[k-1,start][1]+[(start,end)])
                         for start in range(k-1,end) if (k-1,start) in dp]
                dp[k,end]=min(choices,key=lambda v:v[0])
        max_height,groups=dp[columns,n]
        elements=list(headers)
        for col,(start,end) in enumerate(groups):
            x=cfg.margin_inches+col*(cw+gap);y=top
            extra=min(.35,max(0,available-costs[start,end])/max(1,end-start-1))
            for j in range(start,end):
                if j==start or units[j][0].section_id!=units[j-1][0].section_id:
                    if j>start:y+=gap*.8
                    e=make_heading(j,j==start and start>0 and units[start][0].section_id==units[start-1][0].section_id,col)
                    e.x,e.y=x,y;elements.append(e);y+=e.height+gap
                for e in units[j]:
                    e.x=x+(cw-e.width)/2 if e.kind=='figure' else x
                    e.y=y;elements.append(e);y+=e.height+gap
                if j<end-1:y+=extra
        result.elements=elements
        if verify(result,cfg)['valid']:
            return result
        if max_height<=available:
            return result  # not an overflow; do not hide a different defect by shrinking
        for group in units:
            for e in group:
                if e.kind=='text':e.font_pt=max(cfg.minimum_font_pt,e.font_pt-1)
                elif e.kind=='figure':e.width*=.95
    return result
