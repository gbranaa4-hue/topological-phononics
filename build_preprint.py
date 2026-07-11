#!/usr/bin/env python3
"""Render PREPRINT.md -> preprint.pdf with reportlab (no pandoc/latex needed).
Handles headings, paragraphs, bullets, pipe-tables, **bold**/*italic*/`code`, and
sanitizes Unicode that reportlab's base fonts cannot draw. Run: python build_preprint.py
"""
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable)

SAN = {'—': '--', '–': '-', '→': '->', '≈': '~', '×': 'x',
       '≤': '<=', '≥': '>=', '±': '+/-', '²': '^2', '³': '^3',
       '’': "'", '‘': "'", '“': '"', '”': '"', '…': '...',
       '·': '.', '‑': '-', ' ': ' ', '≠': '!=', '√': 'sqrt',
       'ρ': 'rho', 'ψ': 'psi', 'φ': 'phi', 'θ': 'theta',
       'ω': 'omega', 'λ': 'lambda', 'γ': 'gamma', 'β': 'beta'}

def san(s):
    for k, v in SAN.items(): s = s.replace(k, v)
    return s

def inline(s):
    s = san(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', s)
    s = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', s)
    return s

ss = getSampleStyleSheet()
S = {
    'title':  ParagraphStyle('t', parent=ss['Title'], fontSize=15, leading=18, spaceAfter=6),
    'h1':     ParagraphStyle('h1', parent=ss['Heading1'], fontSize=12, leading=15, spaceBefore=10, spaceAfter=4),
    'h2':     ParagraphStyle('h2', parent=ss['Heading2'], fontSize=10.5, leading=13, spaceBefore=7, spaceAfter=3),
    'body':   ParagraphStyle('b', parent=ss['BodyText'], fontSize=9.5, leading=13, alignment=TA_JUSTIFY, spaceAfter=0),
    'bullet': ParagraphStyle('bul', parent=ss['BodyText'], fontSize=9.5, leading=13, leftIndent=16, bulletIndent=4, spaceAfter=2),
    'cell':   ParagraphStyle('c', parent=ss['BodyText'], fontSize=8, leading=10),
    'cellh':  ParagraphStyle('ch', parent=ss['BodyText'], fontSize=8, leading=10, fontName='Helvetica-Bold'),
}

def make_table(tbl):
    rows = [[c.strip() for c in ln.strip().strip('|').split('|')] for ln in tbl]
    rows = [r for r in rows if not all(set(c) <= set('-: ') for c in r)]  # drop the |---| separator
    head, body = rows[0], rows[1:]
    data = [[Paragraph(inline(c), S['cellh']) for c in head]]
    data += [[Paragraph(inline(c), S['cell']) for c in r] for r in body]
    w = 470.0 / len(head)
    t = Table(data, colWidths=[w] * len(head), repeatRows=1)
    t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
                           ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ececec')),
                           ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                           ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                           ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    return t

def build(md='PREPRINT.md', pdf='preprint.pdf'):
    lines = open(md, encoding='utf-8').read().split('\n')
    story, buf, i = [], [], 0
    def flush():
        if buf:
            txt = ' '.join(buf).strip()
            if txt: story.append(Paragraph(inline(txt), S['body'])); story.append(Spacer(1, 5))
            buf.clear()
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip(): flush(); i += 1; continue
        if ln.startswith('# '): flush(); story.append(Paragraph(inline(ln[2:]), S['title'])); i += 1; continue
        if ln.startswith('## '): flush(); story.append(Paragraph(inline(ln[3:]), S['h1'])); i += 1; continue
        if ln.startswith('### '): flush(); story.append(Paragraph(inline(ln[4:]), S['h2'])); i += 1; continue
        if ln.strip() == '---':
            flush(); story.append(Spacer(1, 4)); story.append(HRFlowable(width='100%', color=colors.grey)); story.append(Spacer(1, 4)); i += 1; continue
        if ln.lstrip().startswith('- '):
            flush(); story.append(Paragraph(inline(ln.lstrip()[2:]), S['bullet'], bulletText=chr(8226))); i += 1; continue
        if ln.startswith('|'):
            flush(); tbl = []
            while i < len(lines) and lines[i].strip().startswith('|'): tbl.append(lines[i]); i += 1
            story.append(make_table(tbl)); story.append(Spacer(1, 8)); continue
        buf.append(ln); i += 1
    flush()
    def foot(canv, doc):
        canv.setFont('Helvetica', 8); canv.setFillColor(colors.grey)
        canv.drawCentredString(letter[0] / 2, 0.5 * inch, str(doc.page))
    SimpleDocTemplate(pdf, pagesize=letter, topMargin=0.9 * inch, bottomMargin=0.9 * inch,
                      leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                      title="Topological protection as a defect-tolerant reservoir primitive",
                      author="gbranaa4-hue").build(story, onFirstPage=foot, onLaterPages=foot)
    print(f"wrote {pdf}")

if __name__ == "__main__":
    build()
