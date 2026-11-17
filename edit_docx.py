import docx

doc = docx.Document('/home/zangetsu/Downloads/Project template.docx')

# Project Title Replacements
old_title_caps = "24FPS: MULTI-VENDOR DVR/NVR\nFORENSIC ANALYSIS PLATFORM"
new_title_caps = "FRAME: A RADIOMETRICALLY AND SPATIALLY CONSISTENT\nSUPER-RESOLUTION FRAMEWORK FOR SENTINEL-2"

old_title_caps_inline = "24FPS: MULTI-VENDOR DVR/NVR FORENSIC ANALYSIS PLATFORM"
new_title_caps_inline = "FRAME: A RADIOMETRICALLY AND SPATIALLY CONSISTENT SUPER-RESOLUTION FRAMEWORK FOR SENTINEL-2"

old_title_mixed = "24FPS: Multi-Vendor DVR/NVR Forensic Analysis Platform"
new_title_mixed = "FRAME: A Radiometrically and Spatially Consistent Super-Resolution Framework for Sentinel-2"

old_title_short = "24FPS"
new_title_short = "FRAME"

def replace_text_in_paragraph(p, old_text, new_text):
    if old_text in p.text:
        # Simple replace might break run formatting, but for title it's fine
        inline = p.runs
        # A simple run-safe replacement is hard, we just replace the whole text if it's mostly the title
        if p.text.strip() == old_text:
            p.text = new_text
        else:
            # We try to find and replace in text, then assign to first run, clear others
            text = p.text.replace(old_text, new_text)
            p.text = text

for p in doc.paragraphs:
    replace_text_in_paragraph(p, old_title_caps, new_title_caps)
    replace_text_in_paragraph(p, old_title_caps_inline, new_title_caps_inline)
    replace_text_in_paragraph(p, old_title_mixed, new_title_mixed)

# Let's save a copy to check the replacements
doc.save('/home/zangetsu/Documents/Projects/FRAME/temp1.docx')
print("Saved temp1.docx")
