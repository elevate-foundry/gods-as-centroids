#!/usr/bin/env python3
"""Convert the Gods as Centroids markdown paper to a styled HTML page
for GitHub Pages. Uses KaTeX for math rendering (client-side)."""

import re
import sys
from pathlib import Path


def convert(md_text: str) -> str:
    lines = md_text.split('\n')
    html_parts = []
    i = 0

    # Skip title block (handled in template)
    # Find where the abstract starts
    while i < len(lines) and not lines[i].strip().startswith('## Abstract'):
        i += 1

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip horizontal rules
        if re.match(r'^---+$', stripped):
            i += 1
            continue

        # Section headings
        if stripped.startswith('## References'):
            html_parts.append('<h2 id="references">References</h2>')
            html_parts.append('<div class="references">')
            i += 1
            # Collect all reference lines
            while i < len(lines):
                ref = lines[i].strip()
                if ref == '' or ref.startswith('---'):
                    i += 1
                    continue
                if ref.startswith('## ') or ref.startswith('# '):
                    break
                html_parts.append(f'<p>{process_inline(ref)}</p>')
                i += 1
            html_parts.append('</div>')
            continue

        if stripped.startswith('## Abstract'):
            html_parts.append('<div class="abstract" id="abstract">')
            html_parts.append('<h2>Abstract</h2>')
            i += 1
            # Collect abstract text
            abstract_lines = []
            while i < len(lines) and not lines[i].strip().startswith('**Keywords:'):
                if lines[i].strip():
                    abstract_lines.append(lines[i].strip())
                i += 1
            html_parts.append(f'<p>{process_inline(" ".join(abstract_lines))}</p>')
            html_parts.append('</div>')
            # Keywords
            if i < len(lines) and lines[i].strip().startswith('**Keywords:'):
                kw_text = lines[i].strip().replace('**Keywords:**', '').strip()
                keywords = [k.strip() for k in kw_text.split(',')]
                html_parts.append('<div class="keywords">')
                for kw in keywords:
                    html_parts.append(f'<span>{kw}</span>')
                html_parts.append('</div>')
                i += 1
            continue

        heading_match = re.match(r'^(#{2,4})\s+(.+)$', stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            # Generate ID from text
            anchor = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
            # Map to nav anchors
            anchor = map_anchor(text, anchor)
            tag = f'h{level}'
            html_parts.append(f'<{tag} id="{anchor}">{process_inline(text)}</{tag}>')
            i += 1
            continue

        # Definition / Hypothesis / Corollary blocks
        env_match = re.match(
            r'^\*\*(?:Definition|Hypothesis|Corollary)\s+\d+\w?\s*\((.+?)\)\.\*\*\s*(.*)',
            stripped
        )
        if env_match:
            env_type = 'definition'
            if 'Hypothesis' in stripped:
                env_type = 'hypothesis'
            elif 'Corollary' in stripped:
                env_type = 'corollary'

            label_num = re.search(r'(\d+\w?)', stripped).group(1)
            title = env_match.group(1)
            rest = env_match.group(2)

            html_parts.append(f'<div class="env-block {env_type}">')
            html_parts.append(f'<div class="env-label">{env_type.title()} {label_num} ({title})</div>')
            if rest:
                html_parts.append(f'<p>{process_inline(rest)}</p>')
            i += 1

            # Collect body until next heading, env, or sim result
            while i < len(lines):
                sl = lines[i].strip()
                if sl == '':
                    i += 1
                    continue
                if sl.startswith('## ') or sl.startswith('### '):
                    break
                if re.match(r'^\*\*(?:Definition|Hypothesis|Corollary)\s+\d+', sl):
                    break
                if sl.startswith('**Simulation result:**') or sl.startswith('**Historical prediction:**') or sl.startswith('**Implication:**') or sl.startswith('**Remark.**'):
                    break
                if re.match(r'^---+$', sl):
                    i += 1
                    break

                # Display math
                if sl.startswith('$$') and sl.endswith('$$') and len(sl) > 4:
                    html_parts.append(f'<p>$${sl[2:-2]}$$</p>')
                    i += 1
                    continue
                if sl.startswith('$$'):
                    math_buf = [sl[2:]]
                    i += 1
                    while i < len(lines):
                        ml = lines[i].strip()
                        if ml.endswith('$$'):
                            math_buf.append(ml[:-2])
                            i += 1
                            break
                        math_buf.append(lines[i])
                        i += 1
                    html_parts.append(f'<p>$${"".join(math_buf)}$$</p>')
                    continue

                # Lists inside env
                if sl.startswith('- ') or sl.startswith('* '):
                    html_parts.append('<ul>')
                    while i < len(lines) and (lines[i].strip().startswith('- ') or lines[i].strip().startswith('* ')):
                        item = lines[i].strip()[2:]
                        html_parts.append(f'<li>{process_inline(item)}</li>')
                        i += 1
                    html_parts.append('</ul>')
                    continue

                if re.match(r'^\d+\.\s', sl):
                    html_parts.append('<ol>')
                    while i < len(lines) and re.match(r'^\d+\.\s', lines[i].strip()):
                        item = re.sub(r'^\d+\.\s+', '', lines[i].strip())
                        html_parts.append(f'<li>{process_inline(item)}</li>')
                        i += 1
                    html_parts.append('</ol>')
                    continue

                # Table inside env
                if '|' in sl and sl.startswith('|'):
                    html_parts.append(parse_table(lines, i))
                    while i < len(lines) and '|' in lines[i].strip():
                        i += 1
                    continue

                html_parts.append(f'<p>{process_inline(sl)}</p>')
                i += 1

            html_parts.append('</div>')
            continue

        # Simulation result blocks
        if stripped.startswith('**Simulation result:**'):
            content = stripped.replace('**Simulation result:**', '').strip()
            html_parts.append(f'<div class="sim-result"><strong>Simulation Result</strong><br>{process_inline(content)}</div>')
            i += 1
            continue

        # Historical prediction blocks
        if stripped.startswith('**Historical prediction:**'):
            content = stripped.replace('**Historical prediction:**', '').strip()
            html_parts.append(f'<div class="hist-pred"><strong>Historical Prediction:</strong> {process_inline(content)}</div>')
            i += 1
            continue

        # Implication blocks
        if stripped.startswith('**Implication:**'):
            content = stripped.replace('**Implication:**', '').strip()
            html_parts.append(f'<div class="hist-pred"><strong>Implication:</strong> {process_inline(content)}</div>')
            i += 1
            continue

        # Remark blocks
        if stripped.startswith('**Remark.**'):
            content = stripped.replace('**Remark.**', '').strip()
            html_parts.append(f'<div class="sim-result"><strong>Remark.</strong> {process_inline(content)}</div>')
            i += 1
            continue

        # Display math
        if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 4:
            html_parts.append(f'<p>$${stripped[2:-2]}$$</p>')
            i += 1
            continue
        if stripped.startswith('$$'):
            math_buf = [stripped[2:]]
            i += 1
            while i < len(lines):
                ml = lines[i].strip()
                if ml.endswith('$$'):
                    math_buf.append(ml[:-2])
                    i += 1
                    break
                math_buf.append(lines[i])
                i += 1
            html_parts.append(f'<p>$${"".join(math_buf)}$$</p>')
            continue

        # Ordered list
        if re.match(r'^\d+\.\s', stripped):
            html_parts.append('<ol>')
            while i < len(lines):
                sl = lines[i].strip()
                if re.match(r'^\d+\.\s', sl):
                    item = re.sub(r'^\d+\.\s+', '', sl)
                    html_parts.append(f'<li>{process_inline(item)}</li>')
                    i += 1
                elif sl == '':
                    # Check if next non-blank is another item
                    j = i + 1
                    while j < len(lines) and lines[j].strip() == '':
                        j += 1
                    if j < len(lines) and re.match(r'^\d+\.\s', lines[j].strip()):
                        i = j
                    else:
                        break
                else:
                    break
            html_parts.append('</ol>')
            continue

        # Unordered list
        if stripped.startswith('- ') or stripped.startswith('* '):
            html_parts.append('<ul>')
            while i < len(lines) and (lines[i].strip().startswith('- ') or lines[i].strip().startswith('* ')):
                item = lines[i].strip()
                item = item[2:] if item.startswith('- ') else item[2:]
                html_parts.append(f'<li>{process_inline(item)}</li>')
                i += 1
            html_parts.append('</ul>')
            continue

        # Table
        if '|' in stripped and stripped.startswith('|'):
            html_parts.append(parse_table(lines, i))
            while i < len(lines) and '|' in lines[i].strip():
                i += 1
            continue

        # Regular paragraph
        if stripped:
            para_lines = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('#') \
                    and not lines[i].strip().startswith('**Definition') \
                    and not lines[i].strip().startswith('**Hypothesis') \
                    and not lines[i].strip().startswith('**Corollary') \
                    and not lines[i].strip().startswith('**Simulation') \
                    and not lines[i].strip().startswith('**Historical') \
                    and not lines[i].strip().startswith('**Implication') \
                    and not lines[i].strip().startswith('**Remark') \
                    and not lines[i].strip().startswith('$$') \
                    and not lines[i].strip().startswith('- ') \
                    and not lines[i].strip().startswith('* ') \
                    and not re.match(r'^\d+\.\s', lines[i].strip()) \
                    and not re.match(r'^---+$', lines[i].strip()) \
                    and not (lines[i].strip().startswith('|') and '|' in lines[i]):
                para_lines.append(lines[i].strip())
                i += 1
            html_parts.append(f'<p>{process_inline(" ".join(para_lines))}</p>')
            continue

        i += 1

    return '\n'.join(html_parts)


def process_inline(text: str) -> str:
    """Process inline markdown: bold, italic, inline math, links."""
    # Protect math first
    math_spans = []
    def save_math(m):
        math_spans.append(m.group(0))
        return f'MATH_PLACEHOLDER_{len(math_spans) - 1}'

    text = re.sub(r'\$[^$]+\$', save_math, text)

    # Bold + italic
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    # Restore math
    for idx, m in enumerate(math_spans):
        text = text.replace(f'MATH_PLACEHOLDER_{idx}', m)

    # Escape special HTML chars (but not in math or tags)
    # We'll leave this simple since KaTeX handles math client-side

    return text


def parse_table(lines, start):
    """Parse a markdown table into HTML."""
    rows = []
    i = start
    while i < len(lines) and '|' in lines[i].strip():
        cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
        rows.append(cells)
        i += 1

    if len(rows) < 2:
        return ''

    # First row is header, second is separator (skip it)
    header = rows[0]
    data_rows = rows[2:] if len(rows) > 2 else []

    html = '<table><thead><tr>'
    for h in header:
        html += f'<th>{process_inline(h)}</th>'
    html += '</tr></thead><tbody>'
    for row in data_rows:
        html += '<tr>'
        for cell in row:
            html += f'<td>{process_inline(cell)}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html


def map_anchor(text: str, default: str) -> str:
    """Map section titles to nav anchor IDs."""
    mappings = {
        'Abstract': 'abstract',
        '1. Introduction': 'introduction',
        '2. The Model': 'model',
        '2.1 Belief Space': 'belief-space',
        '2.2 Deity Priors': 'deity-priors',
        '2.3 Interaction Dynamics': 'dynamics',
        '2.4 Clustering': 'clustering',
        '3. A Calculus of Religious Change': 'calculus',
        '3.1 Fusion': 'fusion',
        '3.2 Fission': 'fission',
        '3.3 Perturbation': 'perturbation',
        '4. Phase Transitions': 'phase',
        '4.1 The Coercion': 'coercion',
        '4.2 The Disordered Phase': 'polytheism',
        '4.3 The Ordered Phase': 'monotheism',
        '4.4 Hysteresis': 'hysteresis',
        '5. Corollaries and Predictions': 'corollaries',
        '5.1 Corollary A': 'accessibility',
        '5.2 Corollary B': 'ritual',
        '5.3 Corollary C': 'prestige',
        '5.4 The Discrete Semantic Substrate': 'braille',
        '6. Historical Backtesting': 'backtesting',
        '7. Discussion': 'discussion',
        '8. Related Work': 'related',
        '9. Limitations': 'limitations',
        '10. Conclusion': 'conclusion',
    }
    for key, val in mappings.items():
        if text.startswith(key) or key in text:
            return val
    return default


def main():
    md_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'gods_as_centroids_v2.md'
    template_path = Path(__file__).parent.parent / 'docs' / 'template.html'
    output_path = Path(__file__).parent.parent / 'docs' / 'index.html'

    md_text = md_path.read_text()
    content_html = convert(md_text)

    template = template_path.read_text()

    # Always inject content into the clean template
    # The template has <!-- Content will be injected by the build script --> before </main>
    comment = '  <!-- Content will be injected by the build script -->'
    if comment in template:
        final = template.replace(comment, content_html)
    else:
        # Fallback: insert before </main>
        final = template.replace('</main>', content_html + '\n</main>')

    output_path.write_text(final)
    print(f'Generated {output_path} ({len(final)} bytes)')


if __name__ == '__main__':
    main()
