# Code Walkthrough

This page now acts as a short guide to the focused documentation pages, rather
than duplicating their content.

## Start here

- [Code Structure](code_structure.md)
  Use this for the module-level architecture, execution flow, and the role of
  each major source file.

- [Numerical Method](numerical_method.md)
  Use this for the Numerov formulation, shooting strategy, validation approach,
  and the main numerical accuracy considerations.

- [Reproducing Results](reproducing_results.md)
  Use this for the experiment order, generated outputs, and the organization of
  the `results/` directory.

## When to read the source directly

The documentation above is the compact overview. For file-by-file
implementation details, the source modules themselves are the authoritative
reference: they already contain docstrings, section banners, and inline
comments that explain the role of each top-level function.
