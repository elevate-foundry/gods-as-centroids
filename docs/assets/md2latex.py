#!/usr/bin/env python3
r"""
md2latex.py -- Deterministic Markdown-to-LaTeX converter for academic papers.

Handles:
  - Headings (#, ##, ###) -> section, subsection, subsubsection
  - Bold (**text**) → \textbf{text}
  - Italic (*text*) → \textit{text}
  - Bold-italic (***text***) → \textbf{\textit{text}}
  - Inline math ($...$) → $...$  (passthrough)
  - Display math ($$...$$) → \[ ... \]
  - Ordered lists (1. ...) → \begin{enumerate} ... \end{enumerate}
  - Unordered lists (- ...) → \begin{itemize} ... \end{itemize}
  - Markdown tables (| ... |) → \begin{tabular} ... \end{tabular}
  - Horizontal rules (---) → (skipped, sections handle breaks)
  - Block quotes (> ...) → \begin{quote} ... \end{quote}
  - Inline code (`...`) → \texttt{...}
  - **Keywords:** line → parsed into \keywords{}
  - Abstract section → \begin{abstract} ... \end{abstract}
  - References section → \begin{thebibliography} with \bibitem entries
  - Author/year line → \author{} / \date{}
  - LaTeX special chars escaped in text: &, %, #, _, ~
  - Parenthetical citations like (Author 2020) left as-is (no bibtex)

Usage:
    python3 md2latex.py input.md [output.tex]
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional


# ─── LaTeX document template ─────────────────────────────────────────

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{cleveref}
\usepackage{natbib}
\usepackage{graphicx}
\usepackage{xcolor}

\theoremstyle{definition}
\newtheorem{definition}{Definition}[section]
\newtheorem{hypothesis}{Hypothesis}[section]
\newtheorem{corollary}{Corollary}[section]

\hypersetup{
    colorlinks=true,
    linkcolor=blue!60!black,
    citecolor=green!50!black,
    urlcolor=blue!70!black,
}

"""


# ─── Escape helpers ──────────────────────────────────────────────────

LATEX_SPECIAL = {
    '&': r'\&',
    '%': r'\%',
    '#': r'\#',
    '_': r'\_',
    '~': r'\textasciitilde{}',
}

# Characters that should NOT be escaped inside math mode
MATH_PLACEHOLDER = '\x00MATH\x00'
CODE_PLACEHOLDER = '\x00CODE\x00'


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters in text, preserving math and code spans."""
    # Extract math spans and code spans first
    segments: List[Tuple[str, bool]] = []  # (text, is_protected)

    # Split on inline math $...$ and code `...`
    # We process the string character by character to handle nesting
    result = []
    i = 0
    while i < len(text):
        # Display math $$...$$
        if text[i:i+2] == '$$':
            end = text.find('$$', i + 2)
            if end != -1:
                result.append(text[i:end+2])  # keep as-is
                i = end + 2
                continue
        # Inline math $...$
        if text[i] == '$' and (i == 0 or text[i-1] != '\\'):
            end = text.find('$', i + 1)
            if end != -1:
                result.append(text[i:end+1])  # keep as-is
                i = end + 1
                continue
        # Inline code `...`
        if text[i] == '`':
            end = text.find('`', i + 1)
            if end != -1:
                code_content = text[i+1:end]
                result.append(r'\texttt{' + code_content + '}')
                i = end + 1
                continue
        # Regular character — escape if needed
        ch = text[i]
        if ch in LATEX_SPECIAL:
            result.append(LATEX_SPECIAL[ch])
        else:
            result.append(ch)
        i += 1

    return ''.join(result)


def process_inline_formatting(text: str) -> str:
    """Convert **bold**, *italic*, ***bold-italic*** to LaTeX commands."""
    # Bold-italic first (***...***) 
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\\textbf{\\textit{\1}}', text)
    # Bold (**...**)
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    # Italic (*...*)  — but not inside math $...$
    # We need to be careful not to match * inside math
    text = re.sub(r'(?<!\$)\*([^*\n]+?)\*(?!\$)', r'\\textit{\1}', text)
    return text


def process_line(line: str) -> str:
    """Apply escaping and inline formatting to a single line."""
    # Don't process lines that are pure math
    stripped = line.strip()
    if stripped.startswith('$$') or stripped.startswith('\\[') or stripped.startswith('\\]'):
        return line

    line = escape_latex(line)
    line = process_inline_formatting(line)
    return line


# ─── Section detection ───────────────────────────────────────────────

def detect_heading(line: str) -> Optional[Tuple[int, str]]:
    """Return (level, title) if line is a heading, else None."""
    m = re.match(r'^(#{1,4})\s+(.+)$', line.strip())
    if m:
        return len(m.group(1)), m.group(2)
    return None


HEADING_COMMANDS = {
    1: 'title',
    2: 'section',
    3: 'subsection',
    4: 'subsubsection',
}


# ─── Table parsing ───────────────────────────────────────────────────

def is_table_row(line: str) -> bool:
    return line.strip().startswith('|') and line.strip().endswith('|')


def is_table_separator(line: str) -> bool:
    return bool(re.match(r'^\|[\s\-:|]+\|$', line.strip()))


def parse_table(lines: List[str], start: int) -> Tuple[str, int]:
    """Parse a markdown table starting at `start`, return (latex, end_index)."""
    rows = []
    i = start
    while i < len(lines) and is_table_row(lines[i]):
        if not is_table_separator(lines[i]):
            cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
            rows.append(cells)
        i += 1

    if not rows:
        return '', start

    ncols = len(rows[0])
    col_spec = 'l' * ncols

    latex_lines = []
    latex_lines.append(r'\begin{table}[htbp]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\begin{tabular}{' + col_spec + '}')
    latex_lines.append(r'\toprule')

    for idx, row in enumerate(rows):
        # Pad row to ncols
        while len(row) < ncols:
            row.append('')
        processed = [process_line(cell) for cell in row]
        latex_lines.append(' & '.join(processed) + r' \\')
        if idx == 0:
            latex_lines.append(r'\midrule')

    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')

    return '\n'.join(latex_lines), i


# ─── List parsing ────────────────────────────────────────────────────

def is_ordered_item(line: str) -> Optional[str]:
    m = re.match(r'^\d+\.\s+(.+)$', line.strip())
    return m.group(1) if m else None


def is_unordered_item(line: str) -> Optional[str]:
    m = re.match(r'^[-*]\s+(.+)$', line.strip())
    return m.group(1) if m else None


def parse_ordered_list(lines: List[str], start: int) -> Tuple[str, int]:
    """Parse ordered list items, allowing blank lines, display math, and
    'where' continuation lines between items."""
    # Each item is a list of strings (main line + continuations)
    items: List[List[str]] = []
    i = start
    while i < len(lines):
        content = is_ordered_item(lines[i])
        if content is not None:
            items.append([process_line(content)])
            i += 1
            # Collect continuation: display math and 'where ...' lines
            while i < len(lines):
                sl = lines[i].strip()
                if sl == '':
                    # Check if next non-blank is another ordered item
                    j = i + 1
                    while j < len(lines) and lines[j].strip() == '':
                        j += 1
                    if j < len(lines) and is_ordered_item(lines[j]) is not None:
                        i = j  # skip blanks, outer loop picks up next item
                        break
                    else:
                        break
                elif sl.startswith('$$') and sl.endswith('$$') and len(sl) > 4:
                    items[-1].append('\\[')
                    items[-1].append(sl[2:-2].strip())
                    items[-1].append('\\]')
                    i += 1
                elif sl.startswith('$$'):
                    # Multi-line display math
                    math_buf = []
                    rest_s = sl[2:].strip()
                    if rest_s:
                        math_buf.append(rest_s)
                    i += 1
                    while i < len(lines):
                        ml = lines[i].strip()
                        if ml.endswith('$$'):
                            rest_e = ml[:-2].strip()
                            if rest_e:
                                math_buf.append(rest_e)
                            i += 1
                            break
                        math_buf.append(lines[i])
                        i += 1
                    items[-1].append('\\[')
                    items[-1].append('\n'.join(math_buf))
                    items[-1].append('\\]')
                elif sl.startswith('where ') or sl.startswith('where$'):
                    items[-1].append(process_line(sl))
                    i += 1
                elif is_ordered_item(sl) is not None:
                    break  # next item, handled by outer loop
                else:
                    break
        elif lines[i].strip() == '' and i + 1 < len(lines) and is_ordered_item(lines[i + 1]) is not None:
            i += 1
        else:
            break

    latex = '\\begin{enumerate}\n'
    for item_parts in items:
        latex += '  \\item ' + '\n'.join(item_parts) + '\n'
    latex += '\\end{enumerate}'
    return latex, i


def parse_unordered_list(lines: List[str], start: int) -> Tuple[str, int]:
    """Parse unordered list items, allowing blank lines between items."""
    items = []
    i = start
    while i < len(lines):
        content = is_unordered_item(lines[i])
        if content is not None:
            items.append(process_line(content))
            i += 1
        elif lines[i].strip() == '' and i + 1 < len(lines) and is_unordered_item(lines[i + 1]) is not None:
            # Skip blank line between consecutive unordered items
            i += 1
        else:
            break

    latex = '\\begin{itemize}\n'
    for item in items:
        latex += f'  \\item {item}\n'
    latex += '\\end{itemize}'
    return latex, i


# ─── Reference parsing ──────────────────────────────────────────────

def parse_reference(line: str) -> Optional[Tuple[str, str]]:
    """Parse a reference line into (key, formatted_entry)."""
    line = line.strip()
    if not line or line.startswith('#') or line.startswith('---'):
        return None

    # Generate a cite key from first author + year
    # Match patterns like "Author, A. (2020)." or "Author, A., et al. (2020)."
    m = re.match(r'^([A-Z][a-z]+)', line)
    year_m = re.search(r'\((\d{4})\)', line)
    if m and year_m:
        key = m.group(1).lower() + year_m.group(1)
        # Escape the entry text
        entry = escape_latex(line)
        entry = process_inline_formatting(entry)
        return key, entry

    return None


# ─── Detect definition/hypothesis/corollary blocks ───────────────────

def detect_theorem_env(line: str) -> Optional[Tuple[str, str, str]]:
    """Detect **Definition N (Title).** or **Hypothesis N (Title).** etc.
    Returns (env_name, label, title) or None."""
    patterns = [
        (r'\*\*Definition\s+(\d+)\s*\(([^)]+)\)\.\*\*', 'definition'),
        (r'\*\*Hypothesis\s+(\d+)\s*\(([^)]+)\)\.\*\*', 'hypothesis'),
        (r'\*\*Corollary\s+(\d+)\s*\(([^)]+)\)\.\*\*', 'corollary'),
    ]
    for pattern, env in patterns:
        m = re.match(pattern, line.strip())
        if m:
            num = m.group(1)
            title = m.group(2)
            rest = line.strip()[m.end():].strip()
            return env, f'{env}:{num}', title
    return None


# ─── Main converter ──────────────────────────────────────────────────

def convert(md_text: str) -> str:
    lines = md_text.split('\n')
    output: List[str] = []

    # Extract metadata from first few lines
    title = ''
    author = ''
    date = ''
    keywords = ''

    # State
    in_abstract = False
    in_references = False
    in_display_math = False
    display_math_buf: List[str] = []
    abstract_buf: List[str] = []
    ref_entries: List[Tuple[str, str]] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Skip horizontal rules ──
        if re.match(r'^---+$', stripped):
            i += 1
            continue

        # ── Display math ($$...$$) ──
        if stripped.startswith('$$') and not in_display_math:
            # Check if it's a single-line display math
            if stripped.endswith('$$') and len(stripped) > 4:
                math_content = stripped[2:-2].strip()
                output.append('\\[')
                output.append(math_content)
                output.append('\\]')
                i += 1
                continue
            else:
                in_display_math = True
                display_math_buf = []
                rest = stripped[2:].strip()
                if rest:
                    display_math_buf.append(rest)
                i += 1
                continue

        if in_display_math:
            if stripped.endswith('$$'):
                rest = stripped[:-2].strip()
                if rest:
                    display_math_buf.append(rest)
                output.append('\\[')
                output.append('\n'.join(display_math_buf))
                output.append('\\]')
                in_display_math = False
                i += 1
                continue
            else:
                display_math_buf.append(line)
                i += 1
                continue

        # ── Headings ──
        heading = detect_heading(line)
        if heading:
            level, htitle = heading

            # Title (# ...)
            if level == 1:
                title = htitle
                i += 1
                continue

            # Abstract
            if htitle.strip().lower() == 'abstract':
                in_abstract = True
                i += 1
                continue

            # References
            if htitle.strip().lower() == 'references':
                in_references = True
                i += 1
                continue

            # End abstract if we hit a new section
            if in_abstract:
                in_abstract = False
                output.append('\\begin{abstract}')
                for aline in abstract_buf:
                    output.append(aline)
                output.append('\\end{abstract}')
                abstract_buf = []

            # Regular section
            cmd = HEADING_COMMANDS.get(level, 'subsubsection')
            # Process the title for inline formatting
            processed_title = process_line(htitle)
            # Remove numbering like "1. " or "2.1 " from title if present
            processed_title = re.sub(r'^\d+(\.\d+)*\.?\s+', '', processed_title)
            output.append(f'\\{cmd}{{{processed_title}}}')
            i += 1
            continue

        # ── Abstract content ──
        if in_abstract:
            if stripped:
                # Check for keywords line
                km = re.match(r'\*\*Keywords?:\*\*\s*(.+)', stripped)
                if km:
                    keywords = km.group(1)
                else:
                    abstract_buf.append(process_line(stripped))
            i += 1
            continue

        # ── References content ──
        if in_references:
            ref = parse_reference(stripped)
            if ref:
                ref_entries.append(ref)
            i += 1
            continue

        # ── Author / date lines (right after title) ──
        if not author and not title == '' and stripped.startswith('**') and stripped.endswith('**'):
            author = stripped.strip('*').strip()
            i += 1
            continue
        if not date and re.match(r'^\d{4}$', stripped):
            date = stripped
            i += 1
            continue

        # ── Tables ──
        if is_table_row(stripped):
            table_latex, i = parse_table(lines, i)
            output.append(table_latex)
            continue

        # ── Ordered lists ──
        if is_ordered_item(stripped) is not None:
            list_latex, i = parse_ordered_list(lines, i)
            output.append(list_latex)
            continue

        # ── Unordered lists ──
        if is_unordered_item(stripped) is not None:
            list_latex, i = parse_unordered_list(lines, i)
            output.append(list_latex)
            continue

        # ── Definition / Hypothesis / Corollary blocks ──
        thm = detect_theorem_env(stripped)
        if thm:
            env, label, thm_title = thm
            # Get the rest of the line after the **Definition N (Title).**
            pattern_map = {
                'definition': r'\*\*Definition\s+\d+\s*\([^)]+\)\.\*\*\s*',
                'hypothesis': r'\*\*Hypothesis\s+\d+\s*\([^)]+\)\.\*\*\s*',
                'corollary': r'\*\*Corollary\s+\d+\s*\([^)]+\)\.\*\*\s*',
            }
            rest = re.sub(pattern_map[env], '', stripped).strip()

            output.append(f'\\begin{{{env}}}[{process_line(thm_title)}]')
            output.append(f'\\label{{{label}}}')
            if rest:
                output.append(process_line(rest))

            # Collect continuation lines: include display math, list items,
            # and text separated by single blank lines. Stop at headings,
            # new theorem envs, or double blank lines.
            i += 1
            consecutive_blanks = 0
            while i < len(lines):
                nline = lines[i].strip()

                # Stop conditions
                if detect_heading(lines[i]):
                    break
                if detect_theorem_env(nline):
                    break
                if nline.startswith('**Simulation result:**') or nline.startswith('**Historical prediction:**') or nline.startswith('**Implication:**'):
                    break

                # Track blank lines — stop after 2 consecutive
                if nline == '':
                    consecutive_blanks += 1
                    if consecutive_blanks >= 2:
                        break
                    i += 1
                    continue
                else:
                    consecutive_blanks = 0

                # Display math
                if nline.startswith('$$') and nline.endswith('$$') and len(nline) > 4:
                    output.append('\\[')
                    output.append(nline[2:-2].strip())
                    output.append('\\]')
                    i += 1
                    continue
                if nline.startswith('$$'):
                    # Multi-line display math
                    math_buf = []
                    rest_start = nline[2:].strip()
                    if rest_start:
                        math_buf.append(rest_start)
                    i += 1
                    while i < len(lines):
                        ml = lines[i].strip()
                        if ml.endswith('$$'):
                            rest_end = ml[:-2].strip()
                            if rest_end:
                                math_buf.append(rest_end)
                            break
                        math_buf.append(lines[i])
                        i += 1
                    output.append('\\[')
                    output.append('\n'.join(math_buf))
                    output.append('\\]')
                    i += 1
                    continue

                # Skip stray horizontal rules inside theorems
                if re.match(r'^---+$', nline):
                    i += 1
                    continue

                # Table inside theorem
                if is_table_row(nline):
                    table_latex, i = parse_table(lines, i)
                    output.append(table_latex)
                    continue

                # Ordered list items inside theorem
                if is_ordered_item(nline) is not None:
                    list_latex, i = parse_ordered_list(lines, i)
                    output.append(list_latex)
                    continue

                # Unordered list items inside theorem (- ...)
                if is_unordered_item(nline) is not None:
                    list_latex, i = parse_unordered_list(lines, i)
                    output.append(list_latex)
                    continue

                # Regular text
                output.append(process_line(nline))
                i += 1

            output.append(f'\\end{{{env}}}')
            continue

        # ── Simulation result blocks ──
        if stripped.startswith('**Simulation result:**'):
            content = stripped.replace('**Simulation result:**', '').strip()
            output.append('')
            output.append('\\medskip')
            output.append('\\noindent\\textbf{Simulation result.} ' + process_line(content))
            i += 1
            continue

        # ── Historical prediction blocks ──
        if stripped.startswith('**Historical prediction:**'):
            content = stripped.replace('**Historical prediction:**', '').strip()
            output.append('')
            output.append('\\medskip')
            output.append('\\noindent\\textbf{Historical prediction.} ' + process_line(content))
            i += 1
            continue

        # ── Implication blocks ──
        if stripped.startswith('**Implication:**'):
            content = stripped.replace('**Implication:**', '').strip()
            output.append('')
            output.append('\\medskip')
            output.append('\\noindent\\textbf{Implication.} ' + process_line(content))
            i += 1
            continue

        # ── Regular paragraph ──
        if stripped == '':
            output.append('')
        else:
            output.append(process_line(stripped))

        i += 1

    # ── Flush abstract if still open ──
    if in_abstract and abstract_buf:
        output.append('\\begin{abstract}')
        for aline in abstract_buf:
            output.append(aline)
        output.append('\\end{abstract}')

    # ── Assemble document ──
    doc = []
    doc.append(PREAMBLE)

    if title:
        doc.append(f'\\title{{{process_line(title)}}}')
    if author:
        doc.append(f'\\author{{{escape_latex(author)}}}')
    if date:
        doc.append(f'\\date{{{date}}}')

    if keywords:
        doc.append(f'\\newcommand{{\\keywords}}[1]{{\\par\\medskip\\noindent\\textbf{{Keywords:}} #1}}')

    doc.append('')
    doc.append('\\begin{document}')
    doc.append('\\maketitle')
    doc.append('')

    # Insert body
    doc.extend(output)

    # Keywords
    if keywords:
        doc.append('')
        doc.append(f'\\keywords{{{process_line(keywords)}}}')

    # References
    if ref_entries:
        doc.append('')
        doc.append(f'\\begin{{thebibliography}}{{{len(ref_entries)}}}')
        for key, entry in ref_entries:
            doc.append(f'\\bibitem{{{key}}} {entry}')
        doc.append('\\end{thebibliography}')

    doc.append('')
    doc.append('\\end{document}')

    return '\n'.join(doc)


# ─── CLI ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} input.md [output.tex]', file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.with_suffix('.tex')

    md_text = input_path.read_text(encoding='utf-8')
    latex_text = convert(md_text)
    output_path.write_text(latex_text, encoding='utf-8')

    print(f'Converted {input_path} → {output_path}')
    print(f'  Lines: {len(md_text.splitlines())} MD → {len(latex_text.splitlines())} LaTeX')


if __name__ == '__main__':
    main()
