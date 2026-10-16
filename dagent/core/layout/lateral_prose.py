"""Recover lateral column contamination only with a rendered-ink preservation check."""
from PIL import Image, ImageDraw


def recover(page, crop):
    import fitz
    old = fitz.Rect(crop['bbox'])
    caption = fitz.Rect(crop['caption_bbox']) if crop.get('caption_bbox') else None
    if not caption:
        return list(old), None
    blocks = page.get_text('rawdict', flags=fitz.TEXTFLAGS_RAWDICT & ~fitz.TEXT_PRESERVE_IMAGES)['blocks']
    excluded = []
    for block in blocks:
        box = fitz.Rect(block['bbox'])
        lines = block.get('lines', [])
        texts = [''.join(char['c'] for span in line['spans'] for char in span['chars']) for line in lines]
        if (old.intersects(box) and box.x0 < old.x0 < box.x1 < caption.x0 - 2
                and sum(len(text.strip()) >= 45 for text in texts) >= 3):
            excluded.append(block)
    if not excluded:
        return list(old), None
    new = fitz.Rect(old)
    new.x0 = max(block['bbox'][2] for block in excluded) + .5
    if new.width < old.width * .65:
        return list(old), None
    excluded_ids = {id(block) for block in excluded}
    for block in blocks:
        if id(block) in excluded_ids:
            continue
        for line in block.get('lines', []):
            box = fitz.Rect(line['bbox'])
            if (old & box).get_area() > (new & box).get_area() + .01:
                return list(old), None
    strip = fitz.Rect(old.x0, old.y0, new.x0, old.y1)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=strip, alpha=False)
    pixels = Image.frombytes('RGB', [pixmap.width, pixmap.height], pixmap.samples)
    mask = Image.new('L', pixels.size)
    draw = ImageDraw.Draw(mask)
    for block in excluded:
        for line in block['lines']:
            for span in line['spans']:
                for char in span['chars']:
                    if not char['c'].strip():
                        continue
                    box = fitz.Rect(char['bbox']) + (-.5, -.5, .5, .5)
                    draw.rectangle([box.x0 * 2 - pixmap.x, box.y0 * 2 - pixmap.y,
                                    box.x1 * 2 - pixmap.x, box.y1 * 2 - pixmap.y], fill=255)
    residual = sum(min(pixel) < 250 and not covered for pixel, covered in zip(pixels.getdata(), mask.getdata()))
    if residual:
        return list(old), None
    return list(new), dict(operation='exclude_lateral_prose_with_rendered_ink_guard',
        original_bbox=list(old), bbox=list(new), excluded_prose=[dict(bbox=block['bbox'],
        line_count=len(block['lines'])) for block in excluded], raster_dpi=144,
        unexplained_removed_ink_pixels=residual, no_inpainting=True)
