"""PPTX primitives used by DAGent's editable poster renderer."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.dml.color import RGBColor

def _parse_font_size(font_size):
    """
    Internal helper to convert a numeric font size (e.g., 12) 
    to a python-pptx Pt object. If it's already a Pt, return as-is.
    """
    if font_size is None:
        return None
    if isinstance(font_size, (int, float)):
        return Pt(font_size)
    return font_size

def _parse_alignment(alignment):
    """
    Internal helper to convert a string alignment (e.g., "left", "center") 
    to the corresponding PP_ALIGN constant. 
    Default to PP_ALIGN.LEFT if unrecognized or None.
    """
    if not isinstance(alignment, str):
        # If user passed None or something else, default to PP_ALIGN.LEFT
        return PP_ALIGN.LEFT

    alignment = alignment.lower().strip()
    alignment_map = {
        "left": PP_ALIGN.LEFT,
        "center": PP_ALIGN.CENTER,
        "right": PP_ALIGN.RIGHT,
        "justify": PP_ALIGN.JUSTIFY,
    }
    return alignment_map.get(alignment, PP_ALIGN.LEFT)

def create_poster(width_inch=48, height_inch=36):
    """
    Create a new Presentation object, set its slide size (e.g., 48x36 inches).
    
    :param width_inch: Float or int specifying width in inches (default 48).
    :param height_inch: Float or int specifying height in inches (default 36).
    :return: A python-pptx Presentation object.
    """
    prs = Presentation()
    prs.slide_width = Inches(width_inch)
    prs.slide_height = Inches(height_inch)
    return prs

def add_blank_slide(prs):
    """
    Add a blank slide to the Presentation (layout index 6 is typically blank).
    
    :param prs: The Presentation object to add a slide to.
    :return: The newly added slide object.
    """
    blank_layout = prs.slide_layouts[6]
    return prs.slides.add_slide(blank_layout)

def add_textbox(
    slide, 
    name, 
    left_inch, 
    top_inch, 
    width_inch, 
    height_inch, 
    text="", 
    word_wrap=True,
    font_size=40,
    bold=False,
    italic=False,
    alignment="left",
    fill_color=None,
    font_name="Arial"
):
    """
    Create a textbox shape on the given slide, optionally fill its background with
    a color if fill_color is specified as (r, g, b).
    
    :param slide: Slide object to place the textbox on.
    :param name: Name for the shape (shape.name).
    :param left_inch: Left coordinate (in inches).
    :param top_inch: Top coordinate (in inches).
    :param width_inch: Width (in inches).
    :param height_inch: Height (in inches).
    :param text: Text to display in the textbox.
    :param word_wrap: If True, wrap text in the textbox.
    :param font_size: Numeric font size (e.g. 40).
    :param bold: Boolean to set run.font.bold.
    :param italic: Boolean to set run.font.italic.
    :param alignment: String alignment: "left", "center", "right", or "justify".
    :param fill_color: (r, g, b) tuple for solid fill background color, or None to skip.
    :param font_name: String font name (e.g., "Arial").
    :return: The newly created textbox shape.
    """
    shape = slide.shapes.add_textbox(
        Inches(left_inch), Inches(top_inch),
        Inches(width_inch), Inches(height_inch)
    )
    
    shape.name = name
    
    # If a fill color is specified, apply a solid fill
    if fill_color is not None:
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(*fill_color)
    else:
        # Otherwise, set "no fill" if you want it transparent
        shape.fill.background()

    text_frame = shape.text_frame
    # Turn off auto-size to ensure stable font size, etc.
    text_frame.auto_size = MSO_AUTO_SIZE.NONE
    text_frame.word_wrap = word_wrap
    
    # Clear any default paragraphs
    text_frame.clear()
    
    # Add a new paragraph
    p = text_frame.add_paragraph()
    # Instead of setting p.text, explicitly create a Run
    run = p.add_run()
    run.text = text
    
    # Parse alignment and set it
    p.alignment = _parse_alignment(alignment)
    
    # Set the font formatting on the run
    font = run.font
    font.size = _parse_font_size(font_size)
    font.bold = bold
    font.italic = italic
    font.name = font_name
    
    return shape

def add_image(slide, name, left_inch, top_inch, width_inch, height_inch, image_path):
    """
    Add an image to the slide at the specified position and size.
    
    :param slide: The slide object where the image should be placed.
    :param name: A string name/label for the shape.
    :param left_inch: Left position in inches.
    :param top_inch: Top position in inches.
    :param width_inch: Width in inches.
    :param height_inch: Height in inches.
    :param image_path: File path to the image.
    :return: The newly created picture shape object.
    """
    shape = slide.shapes.add_picture(
        image_path,
        Inches(left_inch), Inches(top_inch),
        width=Inches(width_inch), height=Inches(height_inch)
    )
    shape.name = name
    return shape

def save_presentation(prs, file_name="poster.pptx"):
    """
    Save the current Presentation object to disk.
    
    :param prs: The Presentation object.
    :param file_name: The file path/name for the saved pptx file.
    """
    prs.save(file_name)
