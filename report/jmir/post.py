import sys, re, copy
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

TITLE = ('Media Coverage and Online Search Interest During French Health Awareness Campaigns: '
         'Granger Causality and Transfer Entropy Beyond the Campaign Calendar')

def sty(d, sid):
    for st in d.styles:
        if st.style_id == sid:
            return st
    raise KeyError(sid)

def run(src, dst, title, original_paper, line_numbers):
    d = Document(src)
    d.styles['Normal'].paragraph_format.space_after = Pt(8)
    for sect in d.sections:
        sect.page_width = Inches(8.5); sect.page_height = Inches(11)
        sect.left_margin = Inches(1.25); sect.right_margin = Inches(1.25)
        sect.top_margin = Inches(1); sect.bottom_margin = Inches(1)
    paras = d.paragraphs
    for p in paras:
        t = p.text.strip()
        if t == '@@ORIGINALPAPER@@':
            p.text = 'Original Paper'; p.style = sty(d,'Heading2')
        elif t == '@@TITLE@@':
            p.text = title; p.style = sty(d,'Heading1')
    # remove empty paragraphs that pandoc may leave (not those holding images/math)
    for p in list(d.paragraphs):
        if not p.text.strip() and not p._p.xpath('.//w:drawing') and not p._p.xpath('.//m:oMath') and not p._p.xpath('.//w:br'):
            p._p.getparent().remove(p._p)
    # tables
    for t in d.tables:
        tblPr = t._tbl.tblPr
        b = OxmlElement('w:tblBorders')
        for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            e = OxmlElement('w:' + side)
            e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '4'); e.set(qn('w:space'), '0'); e.set(qn('w:color'), '808080')
            b.append(e)
        tblPr.append(b)
        ncol = len(t.columns)
        wts = []
        for ci in range(ncol):
            m = max((len(c.text.strip()) for c in t.columns[ci].cells), default=4)
            wts.append(min(max(m, 7), 40) + 3)
        tot = float(sum(wts))
        t.autofit = False
        for ci in range(ncol):
            wi = Inches(6.0 * wts[ci] / tot)
            t.columns[ci].width = wi
            for c in t.columns[ci].cells:
                c.width = wi
        fs = Pt(8) if ncol >= 9 else Pt(9)
        for r_i, row in enumerate(t.rows):
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.space_before = Pt(0)
                    for r in p.runs:
                        r.font.size = fs
                        if r_i == 0:
                            r.font.bold = True
    # placeholders highlight
    for p in d.paragraphs:
        if p.text.startswith('[To be completed'):
            for r in p.runs:
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    # line numbers
    if line_numbers:
        for sect in d.sections:
            sp = sect._sectPr
            ln = OxmlElement('w:lnNumType')
            ln.set(qn('w:countBy'), '1'); ln.set(qn('w:restart'), 'continuous')
            # lnNumType must come before cols/docGrid in the sectPr sequence: insert after pgMar
            pg = sp.find(qn('w:pgMar'))
            pg.addnext(ln)
    d.core_properties.title = title
    d.core_properties.author = ''
    d.core_properties.last_modified_by = ''
    d.core_properties.comments = ''
    d.save(dst)

run('build/manuscript.docx', 'build/JMIR_manuscript.docx', TITLE, True, True)
run('build/appendix_full.docx', 'build/JMIR_Multimedia_Appendix_1.docx', 'Multimedia Appendix 1', False, False)
