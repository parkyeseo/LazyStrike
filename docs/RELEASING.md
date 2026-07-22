# Release management

The repository is a research artifact, so a release must keep code, paper, configuration, and reported numbers synchronized.

## Recommended branch model

- Protect `main`; do not perform experiments directly on it.
- Use `experiment/<short-name>` for exploratory training code and server-specific launch scripts.
- Use `agent/research-release` or `release/vX.Y.Z` to assemble a public release.
- Merge through a reviewed pull request after tests and paper rendering pass.
- Tag the merge commit with an annotated tag such as `v1.0.0`.

Avoid committing a dirty GPU-server working tree directly. First preserve exploratory changes on a named experiment branch, then cherry-pick only paper-backed changes into the release branch.

## Versioning

Use semantic versioning for the public artifact:

- Major: changes reported algorithms, tensor contracts, or result interpretation.
- Minor: adds a method, dataset, evaluation protocol, or reproducibility feature without invalidating the paper path.
- Patch: fixes packaging, docs, evaluation initialization, or another behavior that does not change reported results.

Keep these values synchronized:

- `pyproject.toml` project version;
- `CITATION.cff` version and release date;
- Git tag and GitHub Release title;
- release notes.

## Required release checks

1. Start from a clean worktree.
2. Search for hostnames, usernames, absolute paths, tokens, W&B entities, and private dataset locations.
3. Run `pytest tests/ -x -v` and `python scripts/smoke_verify.py`.
4. Run one checkpoint-based classification, PiB, and permutation smoke evaluation without network initialization.
5. Compile `paper/main.tex` with XeLaTeX.
6. Render every PDF page to images and visually inspect title metadata, equations, figures, tables, page numbers, clipping, and Korean glyphs.
7. Verify the README result table against the final paper.
8. Confirm a source-code license; do not infer one from dependencies.
9. Commit only intended files, open a draft PR, and require review from all three equal contributors.
10. After merge, create the annotated tag and a GitHub Release containing the PDF and source archive.

## GitHub Release contents

Recommended assets:

- `Beyond FFT - Frequency vs. Variance vs. Sparsity.pdf`;
- source archive automatically generated from the tag;
- optional checksums for separately hosted checkpoints;
- release notes with environment, dataset split provenance, tested commit, and known limitations.

Large checkpoints should not be committed to Git. Publish them through a durable artifact store, document SHA-256 checksums, and link them from the GitHub Release.

## Paper/code drift rule

The attached final HWP, represented by `paper/main.tex` and the tracked PDF, is the scientific source of truth for `v1.0.0`. Any future code change that alters a paper formula or reported evaluation must trigger either:

- a new paper revision and at least a minor version; or
- an explicit `KNOWN_DIFFERENCES.md` entry if maintained as a separate experimental path.

Do not silently redefine a score under the same config name.
