"""Exact PDF crop resampling and projected source-glyph size, without OCR.

A higher raster sampling rate is NOT evidence of greater underlying source
resolution. Raster-embedded labels and missing crop provenance stay unmeasured.
"""
from copy import deepcopy
from pathlib import Path
import math

from dagent.core.config import writable_path, dump_json
from dagent.core.document import sha256


def prepare_visuals(tree, document, output, *, target_width_inches, minimum_dpi):
    import fitz
    from PIL import Image
    if not all(math.isfinite(x) and x>0 for x in (target_width_inches,minimum_dpi)):
        raise ValueError('Positive finite sampling dimensions required')
    out = writable_path(Path(output))
    if out.exists():
        raise FileExistsError('Source visual audit is immutable; choose a fresh output')
    pdf_path = Path(document['pdf_path'])
    digest = sha256(pdf_path)
    if digest != document['pdf_sha256'] or tree.get('source_pdf_sha256') != digest:
        raise ValueError('Source PDF identity mismatch')
    assets = {str(Path(a['path']).resolve()): a for a in document['assets']}
    result = deepcopy(tree); audit = []
    out.mkdir(parents=True)
    with fitz.open(pdf_path) as pdf:
        for section in result['children']:
            for node in section['children']:
                if not node.get('asset_path'):
                    continue
                asset = assets.get(str(Path(node['asset_path']).resolve()))
                if asset is None:
                    raise ValueError('Selected asset not in source document')
                entry = dict(node_id=node['panel_id'],asset_id=asset['id'],original_path=node['asset_path'],
                    original_sha256=sha256(Path(node['asset_path'])),source_caption=asset['caption'],
                    source_pdf_sha256=digest,rendered_path=node['asset_path'],source_crop=asset.get('source_crop'))
                crop = asset.get('source_crop')
                if not crop:
                    audit.append(dict(entry,status='retained_no_verified_pdf_crop',label_sizes_known=False))
                    continue
                if crop['pdf_sha256'] != digest or type(crop['page']) is not int or not 1<=crop['page']<=len(pdf):
                    raise ValueError('Invalid source crop binding')
                values = crop['bbox']
                if len(values)!=4 or not all(type(x) in (int,float) and math.isfinite(x) for x in values):
                    raise ValueError('Invalid source crop coordinates')
                box, page = fitz.Rect(values), pdf[crop['page']-1]
                if box.is_empty or not page.rect.contains(box):
                    raise ValueError('Source crop outside original PDF page')
                # Ceiling limits sampling memory, not semantic selection. It is
                # explicitly reported; low effective DPI still fails the verifier.
                scale = min(10., max(3.,target_width_inches*(minimum_dpi+1)/box.width))
                path = out/f'visual_{len(audit):03d}.png'
                page.get_pixmap(matrix=fitz.Matrix(scale,scale),clip=box,alpha=False).save(path)
                sizes, partial = [], 0
                raw = page.get_text('rawdict',flags=fitz.TEXTFLAGS_RAWDICT & ~fitz.TEXT_PRESERVE_IMAGES)
                for block in raw['blocks']:
                    for line in block.get('lines',[]):
                        for span in line.get('spans',[]):
                            for char in span.get('chars',[]):
                                if not char['c'].strip():
                                    continue
                                cb = fitz.Rect(char['bbox'])
                                if box.contains(cb):
                                    sizes.append(span['size'])
                                elif box.intersects(cb):
                                    partial += 1
                raster_regions = sum(box.intersects(fitz.Rect(i['bbox'])) for i in page.get_image_info())
                node['asset_path'] = str(path.resolve())
                with Image.open(path) as image:
                    node['figure_size'] = list(image.size)
                    node['figure_aspect'] = image.width/image.height
                sizes.sort()
                audit.append(dict(entry,status='resampled_identical_pdf_crop',rendered_path=node['asset_path'],
                    rendered_sha256=sha256(path),pdf_sampling_dpi=72*scale,pixel_size=node['figure_size'],
                    source_text_size_pt=dict(minimum=min(sizes),p10=sizes[int((len(sizes)-1)*.1)],
                        median=sizes[len(sizes)//2]) if sizes else None,
                    source_glyph_count=len(sizes),partially_clipped_source_glyphs=partial,
                    raster_regions_present=raster_regions,label_sizes_known=bool(sizes),
                    raster_embedded_text_assessed=False,crop_bbox_changed=False))
    dump_json(out/'audit.json',audit)
    return result,audit


def projected_labels(layout, audit, *, threshold_pt=24):
    by_path = {str(Path(a['rendered_path']).resolve()):a for a in audit}
    rows = []
    for e in layout.elements:
        if e.kind!='figure':
            continue
        a = by_path.get(str(Path(e.asset_path).resolve()))
        row = dict(element_id=e.id,width_inches=e.width,height_inches=e.height,
                   threshold_pt=threshold_pt,semantic_legibility_verified=False)
        if a is None or not a.get('source_text_size_pt'):
            rows.append(dict(row,status='unknown',passes_measured_vector_label_threshold=None))
            continue
        b = a['source_crop']['bbox']
        scale = min(e.width*72/(b[2]-b[0]),e.height*72/(b[3]-b[1]))
        sizes = {k:v*scale for k,v in a['source_text_size_pt'].items()}
        rows.append(dict(row,status='projected_vector_glyph_sizes',projected_text_size_pt=sizes,
            source_glyph_count=a['source_glyph_count'],raster_regions_present=a['raster_regions_present'],
            partially_clipped_source_glyphs=a['partially_clipped_source_glyphs'],
            passes_measured_vector_label_threshold=sizes['p10']>=threshold_pt,
            all_embedded_text_assessed=False))
    return rows
