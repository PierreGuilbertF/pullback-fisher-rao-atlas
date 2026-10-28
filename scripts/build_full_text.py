"""Rebuild the HTML article using the local (untracked) TeX and Pandoc.
Run with the bundled Python, or Python with Pillow installed.
"""
from pathlib import Path
import re
import subprocess
import tempfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'docs/papers/pullback-fisher-rao-atlas'
source = (ROOT / 'paper_fr.tex').read_text()
# Print-layout overrides redefine section commands and confuse Pandoc's reader.
source = re.sub(r'\\makeatletter.*?\\makeatother', '', source, flags=re.S)
source = source.replace(r'\date{\today}', r'\date{September 1, 2026}')

# Preserve the manuscript's manually numbered bibliography and citation links.
keys = re.findall(r'\\bibitem\{([^}]+)\}', source)
numbers = {key: str(i + 1) for i, key in enumerate(keys)}
def citation(match):
    return ', '.join(r'\hyperlink{ref-' + key.strip() + '}{[' + numbers[key.strip()] + ']}'
                     for key in match.group(1).split(','))
source = re.sub(r'\\cite\{([^}]+)\}', citation, source)
source = source.replace(r'\begin{thebibliography}{99}', r'\section*{References}')
source = source.replace(r'\end{thebibliography}', '')
source = re.sub(r'\\bibitem\{([^}]+)\}',
                lambda m: '\n\n' + r'\hypertarget{ref-' + m.group(1) + '}{}[' + numbers[m.group(1)] + '] ', source)

with tempfile.TemporaryDirectory() as directory:
    clean = Path(directory) / 'paper.tex'
    clean.write_text(source)
    result = subprocess.run(['pandoc', str(clean), '--from=latex', '--to=html5',
                             '--mathml', '--section-divs', '--number-sections', '--shift-heading-level-by=1'],
                            check=True, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr)
    body = result.stdout
    count = [0]
    def caption(match):
        count[0] += 1
        return '<figcaption>Figure ' + str(count[0]) + '. '
    body = re.sub(r'<figcaption>', caption, body)
assets = TARGET / 'figures'
assets.mkdir(exist_ok=True)
for name in sorted(set(re.findall(r'src="data/([^"<>]+)"', body))):
    image_source = ROOT / 'figures' / name
    if not image_source.is_file():
        # This figure is embedded in the PDF but absent from figures/.
        if name != '34_gibbs_degenerate_displacement.png':
            raise FileNotFoundError(image_source)
        from pypdf import PdfReader
        reader = PdfReader(ROOT / 'fisher_rao_hal_v1.pdf')
        image = reader.pages[44].images[0].image.convert('RGB')
        image.thumbnail((1600, 1600))
        output_name = Path(name).with_suffix('.jpg').name
        image.save(assets / output_name, quality=90, optimize=True)
        body = body.replace('src="data/' + name + '"', 'loading="lazy" src="figures/' + output_name + '"')
        continue
    output_name = Path(name).with_suffix('.jpg').name
    with Image.open(image_source) as image:
        image = image.convert('RGB')
        image.thumbnail((1600, 1600))
        image.save(assets / output_name, quality=90, optimize=True)
    body = body.replace('src="data/' + name + '"', 'loading="lazy" src="figures/' + output_name + '"')
# The abstract page holds the authoritative shared metadata and title.
landing = (TARGET / 'index.html').read_text()
head = landing.split('<body>', 1)[0]
head = head.replace('href="https://pierreguilbertf.github.io/pullback-fisher-rao-atlas/papers/pullback-fisher-rao-atlas/"',
                    'href="https://pierreguilbertf.github.io/pullback-fisher-rao-atlas/papers/pullback-fisher-rao-atlas/full-text.html"', 1)
heading = re.search(r'<h1.*?</h1>', landing, re.S).group(0)
abstract = re.search(r'<h2>Abstract</h2>\s*<p>.*?</p>', landing, re.S).group(0)
article = (head + '<body><a class="skip" href="#main">Skip to content</a><main id="main" class="full-text">'
           '<nav aria-label="Breadcrumb"><a href="../../">Publications</a> · <a href="./">Abstract and citation</a> · '
           '<a href="paper.pdf">PDF</a></nav>' + heading +
           '<p class="byline">Pierre Guilbert · September 1, 2026 · Preprint</p>' + abstract + body +
           '</main><footer><a href="./">Abstract and citation</a> · <a href="paper.pdf">PDF version</a></footer></body></html>')
(TARGET / 'full-text.html').write_text(article)
print('Generated full-text.html with', len(list(assets.glob('*.jpg'))), 'figures.')
