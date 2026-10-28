# Pullback Fisher–Rao Atlas

Code, numerical experiments, and reusable figures accompanying the paper

**Pierre Guilbert — _Pullback Fisher–Rao Geometries of Parametric Probability Families: An Expository Atlas_ (2026).**

This repository contains the material used to produce the numerical experiments and figures of the paper.

## Scope

The paper studies pullback Fisher–Rao geometries for:

- univariate and multivariate Gaussian families;
- elliptical Laplace families;
- linear and nonlinear Gibbs families;
- push-forward families.

For a parametric family $\theta \mapsto p_\theta$, the pullback Fisher–Rao metric is

```math
g_{ij}(\theta)
=
\int
\frac{\partial_i p_\theta(x)\,\partial_j p_\theta(x)}
     {p_\theta(x)}
\,dx.
```

The repository includes both closed-form examples and numerical approximations of Riemannian geodesics.

## Repository structure

```text
pullback-fisher-rao-atlas/
├── code/
├── figures/
├── CITATION.cff
├── LICENSE
└── README.md
```

The organization of `code/` and `figures/` follows the main probability families studied in the paper.

## Paper

**Pullback Fisher–Rao Geometries of Parametric Probability Families: An Expository Atlas**  
Pierre Guilbert, 2026.

Publication website (after enabling GitHub Pages):

- [Abstract and citation](https://pierreguilbertf.github.io/pullback-fisher-rao-atlas/papers/pullback-fisher-rao-atlas/)
- [Searchable PDF](https://pierreguilbertf.github.io/pullback-fisher-rao-atlas/papers/pullback-fisher-rao-atlas/paper.pdf)
- [HTML full text](https://pierreguilbertf.github.io/pullback-fisher-rao-atlas/papers/pullback-fisher-rao-atlas/full-text.html)
- [Author ORCID](https://orcid.org/0009-0002-1582-4435)

## Figures

Unless otherwise stated, the original figures in this repository were created by **Pierre Guilbert** and are released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You may reuse, redistribute, and adapt them with appropriate attribution.

Suggested attribution for an unchanged figure:

> Figure reproduced from P. Guilbert,  
> _Pullback Fisher–Rao Geometries of Parametric Probability Families: An Expository Atlas_, 2026.  
> CC BY 4.0.

For an adapted figure, replace **“reproduced”** with **“adapted”**.

## Citation

If you use the code, figures, or numerical experiments from this repository, please cite the associated paper.

Until the HAL/arXiv identifiers are available:

```bibtex
@misc{Guilbert2026PullbackFisherRao,
  author    = {Pierre Guilbert},
  title     = {Pullback Fisher--Rao Geometries of Parametric Probability Families:
               An Expository Atlas},
  year      = {2026},
  note      = {Preprint},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22921871},
  url       = {https://doi.org/10.5281/zenodo.22921871}
}
```

A `CITATION.cff` file is also provided for GitHub's **Cite this repository** feature.

## License

- **Source code:** MIT License
- **Figures and graphical material:** CC BY 4.0

See `LICENSE` and the license notice in the `figures/` directory.

## Author

**Pierre Guilbert**  
Independent researcher

## Publishing the research website

The static website lives in `docs/`; no build step, JavaScript, or package
installation is needed to publish it.

1. Commit and push `docs/`, `scripts/`, `.gitignore`, `README.md`, and
   `CITATION.cff`. The original local PDF and `paper_fr.tex` are ignored;
   the optimized public PDF in `docs/` is intended to be committed.
2. In the GitHub repository, open **Settings → Pages**. Select **Deploy
   from a branch**, select the branch you pushed, choose **/docs**, and save.
3. Open <https://pierreguilbertf.github.io/pullback-fisher-rao-atlas/> after
   GitHub finishes deploying. Check the PDF, full text, and figure links.
4. Verify the site in [Google Search Console](https://search.google.com/search-console/)
   using its URL-prefix property and the verification method Google provides.
   For HTML-file verification, put Google's supplied file in `docs/` and push it.
   Submit the site's `sitemap.xml` and request indexing of the paper page.
   Search Console concerns Google Search, not a Google Scholar submission.
5. Add the paper to your Scholar profile and periodically search Scholar for
   its exact title. Eligibility does not guarantee indexing.

The paper page includes Highwire citation metadata, an absolute PDF URL in
its own directory, the complete author-written abstract, and a BibTeX download.
Navigation uses ordinary HTML links. The HTML full text uses native MathML
and local figures, without a third-party equation-rendering dependency.
See [Google Scholar's inclusion requirements](https://scholar.google.com/intl/en/scholar/inclusion.html).

This is a GitHub Pages **project** site. A `docs/robots.txt` would be served
below the project path, not at the host root, so none is added. Ensure any
host-level robots rules you manage allow this project to be crawled.

### Updating publication details

The Zenodo DOI is recorded in the citation above. Add that identifier to
both HTML versions, their citation metadata, BibTeX, and `CITATION.cff`
when updating the publication details. Keep the existing DOI when updating
a version. To add a second paper, create its own directory under `docs/papers/`
with its abstract page, full text, and PDF, then link it from the homepage
and add it to the sitemap. Do not reuse this paper's citation metadata.

The absolute URLs assume the repository keeps its current owner and name.
If you use another address or a custom domain, update canonical links,
citation URLs, structured data, BibTeX, and sitemap URLs together.

### Rebuilding the HTML full text

With Pandoc, Python, Pillow, and pypdf available, run:

```sh
python3 scripts/build_full_text.py
```

This uses the ignored local `paper_fr.tex` and original PDF. The latter
supplies the figure that is missing from `figures/`. The script removes
print-only section overrides and converts equations to MathML. Review the
result against the PDF after manuscript changes. It preserves the published
manuscript date rather than evaluating the source's `\today`.

The public PDF is a separate optimized copy: illustrations are resized and
compressed while text and equations remain searchable. Keep the original
local file for archival submission. Figure licensing also applies to the
copies in `docs/papers/pullback-fisher-rao-atlas/figures/` (CC BY 4.0; see
`figures/LICENSE.md`).
