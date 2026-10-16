"""Native source pixels only; a damaged crop must not discard all source text."""
from pathlib import Path
import re
from dagent.core.config import writable_path, dump_json
from dagent.core.document import sha256


def _diagnose(diagnostics, page, stage, exc, **extra):
    diagnostics.append(dict(page=page, stage=stage, error_type=type(exc).__name__,
                            reason=str(exc)[:250], **extra))


def pixel_audit(path):
    """Cheap full-resolution blank/degenerate gate, not semantic figure QA."""
    from PIL import Image, ImageStat
    with Image.open(path) as image:
        image.load()
        gray = image.convert('L')
        histogram = gray.histogram()
        count = gray.width * gray.height
        stats = ImageStat.Stat(gray)
        nonwhite = sum(histogram[:245]) / max(1, count)
        mean, variance = stats.mean[0], stats.var[0]
        failures = []
        if min(gray.size) < 16:
            failures.append('too_small')
        if nonwhite < .001:
            failures.append('blank_crop')
        if variance < 1.0:
            failures.append('degenerate_pixel_variance')
        return dict(policy='source_visual_pixel_check', valid=not failures,
                    failures=failures, pixel_width=gray.width, pixel_height=gray.height,
                    nonwhite_fraction=nonwhite, mean_brightness=mean, pixel_variance=variance,
                    thresholds=dict(min_nonwhite_fraction=.001, min_variance=1.0),
                    semantic_figure_completeness_verified=False)


def render_crop(page, clip, path, provenance, diagnostics):
    """Record warnings even when MuPDF silently writes a blank PNG."""
    import fitz
    fitz.TOOLS.mupdf_warnings(reset=True)
    report = dict(source_crop=provenance, path=str(path), status='failed')
    try:
        page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip, alpha=False).save(path)
        pixels = pixel_audit(path)
        report['pixel_audit'] = pixels
        if not pixels['valid']:
            raise ValueError('Native crop failed pixel gate: ' + ', '.join(pixels['failures']))
        report['status'] = 'available'
    except Exception as exc:
        report.update(error_type=type(exc).__name__, reason=str(exc)[:250])
    warnings = fitz.TOOLS.mupdf_warnings(reset=True)
    report['mupdf_warnings'] = warnings
    if re.search(r'(?i)(cannot find.*(?:xobject|resource)|(?:missing|invalid|broken).*xobject)', warnings):
        report.update(status='failed', error_type='DamagedXObject',
                      reason='Source rendering reported missing/damaged graphics resources')
    dump_json(path.with_suffix('.audit.json'), report)
    if report['status'] != 'available':
        diagnostics.append(dict(stage='crop_preflight', page=provenance['page'], **report))
        return None
    return report


def read_pdf(pdf: Path, output: Path, diagnostics=None):
    diagnostics = diagnostics if diagnostics is not None else []
    import fitz
    blocks, assets = [], []
    source_digest = sha256(Path(pdf))
    output = writable_path(output)
    output.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf) as doc:
        if len(doc)==0:
            raise ValueError('Source PDF has no readable pages; invalid benchmark input, not a generation failure')
        metadata = doc.metadata or {}
        for page_index in range(len(doc)):
            fitz.TOOLS.mupdf_warnings(reset=True)
            try:
                page = doc[page_index]
                raw = [b for b in page.get_text('blocks', sort=True) if b[6] == 0 and b[4].strip()]
            except Exception as exc:
                _diagnose(diagnostics, page_index + 1, 'page_text', exc)
                warnings = fitz.TOOLS.mupdf_warnings(reset=True)
                if warnings:
                    diagnostics[-1]['mupdf_warnings'] = warnings
                continue
            for block in raw:
                text = re.sub(r'(?<=\w)-\n(?=\w)', '', block[4]).strip()
                blocks.append(dict(id=f'p{page_index+1}b{len(blocks)}', page=page_index+1,
                                   bbox=list(block[:4]), text=text,
                                   heading=(len(text) < 110 and not text.rstrip().endswith('.'))))
            from dagent.core.native_assets import caption_graphics_crops
            try:
                image_rects = [fitz.Rect(d['bbox']) for d in page.get_image_info()]
            except Exception as exc:
                _diagnose(diagnostics, page_index + 1, 'image_inventory', exc)
                image_rects = []
            try:
                crops = caption_graphics_crops(page, raw)
            except Exception as exc:
                _diagnose(diagnostics, page_index + 1, 'caption_crops', exc)
                crops = []
            # Missing XObjects can prevent any crop seed from being found.
            # Capture inventory warnings as well as per-pixmap warnings; a
            # zero-length crop list is not evidence that the PDF is healthy.
            warnings = fitz.TOOLS.mupdf_warnings(reset=True)
            caption_boxes = [list(b[:4]) for b in raw if re.match(r'^\s*(?:Figure|Fig\.|Table)\s*\d+', b[4], re.I)]
            if warnings or (caption_boxes and not crops and not image_rects):
                diagnostics.append(dict(stage='page_graphics_inventory', page=page_index+1,
                    pdf_sha256=source_digest, mupdf_warnings=warnings,
                    caption_bboxes=caption_boxes, crop_seeds=len(crops), raster_seeds=len(image_rects),
                    extraction='native_pdf_graphics_inventory',
                    reason='source_resource_warnings' if warnings else 'captions_without_reliable_native_graphics'))
            extracted = []
            for crop_index, crop in enumerate(crops):
                clip = fitz.Rect(crop['bbox'])
                extracted.append(clip)
                # Use crop attempt IDs, not success counts: a failed image and
                # its diagnostics must not be overwritten by the next success.
                path = writable_path(output / f'page{page_index+1}_caption{crop_index}.png')
                provenance = dict(pdf_sha256=source_digest, page=page_index+1,
                                  bbox=list(clip), caption_bbox=crop['caption_bbox'],
                                  extraction=crop['extraction'],
                                  graphics_fragments_merged=crop.get('graphics_count', 0))
                report = render_crop(page, clip, path, provenance, diagnostics)
                if report is None:
                    continue
                assets.append(dict(crop, id=f'a{len(assets)}', path=str(path), page=page_index+1,
                                   source_crop=provenance, pixel_audit=report['pixel_audit'],
                                   pixel_width=report['pixel_audit']['pixel_width'],
                                   pixel_height=report['pixel_audit']['pixel_height']))
            # Raster images without captions remain explicitly marked, not fabricated captions.
            for image_index, rect in enumerate(image_rects):
                if rect.width < 65 or rect.height < 50 or any((rect&r).get_area()/max(1,rect.get_area()) > .8 for r in extracted):
                    continue
                if rect.get_area() > page.rect.get_area()*.8:
                    continue
                path = writable_path(output / f'page{page_index+1}_raster{image_index}.png')
                provenance = dict(pdf_sha256=source_digest, page=page_index+1, bbox=list(rect),
                                  caption_bbox=None, extraction='native_pdf_raster_uncaptioned')
                report = render_crop(page, rect, path, provenance, diagnostics)
                if report is None:
                    continue
                assets.append(dict(id=f'a{len(assets)}', kind='figure', path=str(path), caption='',
                                   page=page_index+1, bbox=list(rect), extraction='native_pdf_raster_uncaptioned',
                                   source_crop=provenance, pixel_audit=report['pixel_audit'],
                                   pixel_width=report['pixel_audit']['pixel_width'],
                                   pixel_height=report['pixel_audit']['pixel_height']))
        title = metadata.get('title', '').strip()
    return blocks, assets, title
