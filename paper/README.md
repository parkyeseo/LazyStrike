# Paper build

`main.tex` is the XeLaTeX source for **Beyond FFT: Frequency vs. Variance vs. Sparsity**. The tracked PDF is the release artifact linked from the repository README.

This release uses the official CVPR 2026 camera-ready style, with the upstream style file vendored beside the source for reproducible layout.

The source uses `Batang` and `Malgun Gothic` for Korean text, plus Times New Roman, Arial, and Consolas for Latin text. On another operating system, install equivalent fonts or change the font declarations in `main.tex`.

Build with XeLaTeX (or Tectonic):

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

The release PDF must be rendered page-by-page and visually checked after any content, font, figure, table, or geometry change.

Author policy for this release:

- Tae Joo Lee, Jang Ho Park, and Yeseo Park are all equal contributors.
- The original first-page postal address is omitted.
- The original separate representative-email line is omitted.
- Each author's individual email remains in the author block.
