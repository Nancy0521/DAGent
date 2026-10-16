"""Prepare verified source figure and table assets for poster layout."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PIL import Image

from dagent.core.config import dump_json
from dagent.core.layout.guarded_visuals import prepare_guarded_visuals

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


