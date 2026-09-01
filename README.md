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

- HAL: to be added
- arXiv: to be added

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
  author = {Pierre Guilbert},
  title  = {Pullback Fisher--Rao Geometries of Parametric Probability Families:
            An Expository Atlas},
  year   = {2026},
  note   = {Preprint}
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
