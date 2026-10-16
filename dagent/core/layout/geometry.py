from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from functools import lru_cache
import math
import re

from dagent.core.poster_config import TITLE_FONT_SIZE, SECTION_TITLE_FONT_SIZE, THEME_TITLE_FILL_COLOR


@dataclass
class Element:
    id: str
    kind: str
    x: float
    y: float
    width: float
    height: float
    text: str = ''
    font_pt: float = 36
    asset_path: str | None = None
    source_ids: list[str] = field(default_factory=list)
    section_id: str = ''
    fill: list[int] | None = None
    color: list[int] = field(default_factory=lambda:[30,40,55])
    bold: bool = False
    line_spacing: float = 1.12


@dataclass
class Layout:
    width: float
    height: float
    elements: list[Element]
    columns: int = 3

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(data['width'],data['height'],[Element(**e) for e in data['elements']],data.get('columns',3))

    def get(self, element_id):
        return next((e for e in self.elements if e.id == element_id),None)


@lru_cache(maxsize=16)
def font(bold=False):
    from PIL import ImageFont
    names = [f'/usr/share/fonts/truetype/liberation2/LiberationSans-{ "Bold" if bold else "Regular" }.ttf',
             f'/usr/share/fonts/truetype/dejavu/DejaVuSans{ "-Bold" if bold else "" }.ttf']
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name,100)
    return ImageFont.load_default()


@lru_cache(maxsize=8192)
def _wrapped_lines(text, width_inches, font_pt, bold=False):
    f = font(bold)
    max_width = max(.1, width_inches-.16)*72/font_pt*100/1.04
    lines = []
    for paragraph in text.split('\n'):
        line = ''
        for word in paragraph.split():
            candidate = (line+' '+word).strip()
            if f.getlength(candidate) <= max_width:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = word
        if line or not paragraph:
            lines.append(line)
    return tuple(lines or [''])


def wrap_lines(text, width_inches, font_pt, bold=False):
    # Return a fresh list: callers cannot corrupt cached font measurements.
    return list(_wrapped_lines(text, width_inches, font_pt, bold))


def text_height(text, width, font_pt, bold=False, spacing=1.12):
    return .16 + len(wrap_lines(text,width,font_pt,bold))*font_pt/72*spacing


def initial_layout(tree, cfg):
    """Fit Poster Tree sections into contiguous columns without content deletion."""
    if cfg.design == 'balanced':
        from dagent.core.layout.balanced import balanced_layout
        return balanced_layout(tree, cfg)
    width,height,margin,gap = cfg.width_inches,cfg.height_inches,cfg.margin_inches,cfg.gap_inches
    col_width = (width-2*margin-(cfg.columns-1)*gap)/cfg.columns
    title_font = min(TITLE_FONT_SIZE, width*1.45)
    title_h = text_height(tree['title'],width-2*margin,title_font,True)+.25
    elements = [Element('title','title',margin,margin,width-2*margin,title_h,tree['title'],title_font,
                        fill=list(THEME_TITLE_FILL_COLOR),color=[255,255,255],bold=True)]
    top = margin+title_h+gap
    if tree.get('authors'):
        author_h = text_height(tree['authors'],width-2*margin,cfg.minimum_font_pt)
        elements.append(Element('authors','authors',margin,top,width-2*margin,author_h,tree['authors'],cfg.minimum_font_pt))
        top += author_h+gap
    sections = tree['children']
    heading_font = min(SECTION_TITLE_FONT_SIZE,max(30,cfg.body_font_pt*1.12))
    section_heights = []
    specs = []
    for section in sections:
        parts = [('heading',section['section_name'],None,section['panel_id'],[],heading_font,True)]
        for child in section['children']:
            if child.get('asset_path'):
                parts.append(('figure','',child,child['panel_id']+'_image',child['source_ids'],0,False))
                parts.append(('caption',child['content'],None,child['panel_id']+'_caption',child['source_ids'],cfg.minimum_font_pt,False))
            else:
                parts.append(('text',child['content'],None,child['panel_id'],child['source_ids'],cfg.body_font_pt,False))
        heights=[]
        for kind,text,child,eid,sources,size,bold in parts:
            if kind == 'figure':
                heights.append(min(col_width/max(.15,child['figure_aspect']), (height-top-margin)*.35))
            else:
                heights.append(text_height(text,col_width,size,bold)+(.12 if kind == 'heading' else 0))
        specs.append((section,parts,heights))
        section_heights.append(sum(heights)+gap*(len(parts)-1)+gap)
    # Dynamic programming contiguous partition: faithful column-major reading order.
    partitions = partition_sections(section_heights, min(cfg.columns,max(1,len(sections))))
    for col, indices in enumerate(partitions):
        y = top
        for index in indices:
            section,parts,heights = specs[index]
            for (kind,text,child,eid,sources,size,bold),h in zip(parts,heights):
                x = margin+col*(col_width+gap)
                w=col_width
                if kind=='figure':
                    w=min(col_width,h*child['figure_aspect'])
                    x+=(col_width-w)/2
                elements.append(Element(eid,kind,x,y,w,h,text,size or cfg.minimum_font_pt,
                    child['asset_path'] if child else None,sources,section['panel_id'],
                    list(THEME_TITLE_FILL_COLOR) if kind=='heading' else None,
                    [255,255,255] if kind=='heading' else [30,40,55],bold))
                y+=h+gap
    return Layout(width,height,elements,cfg.columns)


def partition_sections(heights, columns):
    n=len(heights)
    if not n:
        return []
    sums=[0]
    for h in heights:
        sums.append(sums[-1]+h)
    dp={(0,0):(0,[])}
    for k in range(1,columns+1):
        for end in range(k,n+1):
            choices=[]
            for start in range(k-1,end):
                if (k-1,start) in dp:
                    cost,groups=dp[k-1,start]
                    choices.append((max(cost,sums[end]-sums[start]),groups+[list(range(start,end))]))
            dp[k,end]=min(choices,key=lambda x:x[0])
    return dp[columns,n][1]


def intersection(a,b):
    return max(0,min(a.x+a.width,b.x+b.width)-max(a.x,b.x))*max(0,min(a.y+a.height,b.y+b.height)-max(a.y,b.y))
