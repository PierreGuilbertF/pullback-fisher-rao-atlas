"""Build the figure gallery from the published HTML article (standard library only)."""
from html import escape
from pathlib import Path
import re

TARGET = Path(__file__).resolve().parents[1] / 'docs/papers/pullback-fisher-rao-atlas'


def build_gallery():
    article = (TARGET / 'full-text.html').read_text()
    cards = []
    for figure_id, content in re.findall(r'<figure\b[^>]*id="([^"]+)"[^>]*>(.*?)</figure>', article, re.S):
        image = re.search(r'<img\b[^>]*src="([^"]+)"[^>]*>', content)
        caption = re.search(r'<figcaption>(.*?)</figcaption>', content, re.S)
        if not image or not caption:
            raise ValueError(f'Missing image or caption: {figure_id}')
        src = image.group(1)
        if not (TARGET / src).is_file():
            raise FileNotFoundError(src)
        number = re.match(r'Figure (\d+)\.', caption.group(1)).group(1)
        cards.append(
            f'<figure class="figure-card" id="{escape(figure_id)}">\n'
            f'<a class="figure-image" href="{escape(src)}" aria-label="Open Figure {number} at full size">'
            f'<img loading="lazy" decoding="async" src="{escape(src)}" alt="Figure {number}; description below" /></a>\n'
            f'<figcaption>{caption.group(1)}</figcaption>\n'
            f'<p class="figure-links"><a href="{escape(src)}">Full-size image</a> · '
            f'<a href="full-text.html#{escape(figure_id)}">Read in context</a></p>\n</figure>')
    if not cards:
        raise ValueError('No figures found in the article')
    page = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Figure gallery — Pullback Fisher–Rao Atlas</title>
  <meta name="description" content="All figures from the Pullback Fisher–Rao Atlas, with original captions and links to the paper.">
  <link rel="canonical" href="https://pierreguilbertf.github.io/pullback-fisher-rao-atlas/papers/pullback-fisher-rao-atlas/gallery.html">
  <link rel="stylesheet" href="../../styles.css">
</head>
<body>
  <a class="skip" href="#main">Skip to content</a>
  <main id="main" class="gallery">
    <nav aria-label="Breadcrumb"><a href="../../">Publications</a> · <a href="./">Abstract and citation</a> · <a href="full-text.html">Full text</a> · <a href="paper.pdf">PDF</a></nav>
    <p class="eyebrow">Pullback Fisher–Rao Atlas</p>
    <h1>Figure gallery</h1>
'''
    page += f'    <p>Browse all {len(cards)} figures in paper order, with their original captions. Select an image to view it at full size.</p>\n'
    page += '    <p>Figures by Pierre Guilbert, licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Please credit the author and paper when reusing or adapting a figure.</p>\n'
    page += '    <div class="figure-grid">\n' + '\n'.join(cards) + '\n    </div>\n'
    page += '  </main>\n  <footer>Pierre Guilbert · <a href="./">Abstract and citation</a> · <a href="https://github.com/PierreGuilbertF/pullback-fisher-rao-atlas">Source repository</a></footer>\n</body>\n</html>\n'
    (TARGET / 'gallery.html').write_text(page)
    print(f'Generated gallery.html with {len(cards)} figures.')


if __name__ == '__main__':
    build_gallery()
