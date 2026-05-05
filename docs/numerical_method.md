# Numerical Method

We solve the one-dimensional time-independent Schrödinger equation

```text
psi''(x) = 2 [V(x) - E] psi(x)
```

for the special energies `E` that satisfy the required boundary behavior. This
makes the problem a boundary-value eigenvalue problem rather than a standard
initial-value problem.

## Symmetry reduction

For the bound-state cases studied here, the main potentials are symmetric:

```text
V(-x) = V(x)
```

The eigenstates can therefore be chosen to have definite parity:

- even states satisfy `psi(-x) = psi(x)` and `psi'(0) = 0`
- odd states satisfy `psi(-x) = -psi(x)` and `psi(0) = 0`

That symmetry allows the solver to integrate only on the half-domain
`[0, x_max]` and reconstruct the negative-`x` side by reflection.

## Workflow

1. Choose a trial energy `E`.
2. Construct `q(x) = 2 [V(x) - E]`.
3. Integrate the wavefunction with the Numerov method.
4. Evaluate a boundary mismatch function.
5. Use bracketing and bisection to locate energies where the mismatch vanishes.
6. Reconstruct and normalize the full wavefunction.

## Shooting formulations

- **Outward shooting**
  Starts at `x = 0` from the exact parity conditions and integrates toward
  `x_max`. It is used for finite-domain or effectively boxed symmetric
  bound-state problems.

- **Inward shooting**
  Starts from a decaying tail near `x_max` and integrates toward the origin. It
  is used for unbounded confining problems such as the harmonic oscillator,
  where outward shooting is numerically less stable.

The outward formulation is the more general one in this project because its
startup comes directly from symmetry at the origin. The inward formulation is
more specialized because it assumes the chosen `x_max` already lies in a
forbidden decaying region.

## Root finding and diagnostics

For a fixed trial energy, the solver computes a mismatch `M(E)`.

- In outward shooting, the mismatch is the boundary leakage at `x = x_max`.
- In inward shooting, the mismatch is the parity condition at the origin.

The code scans the mismatch over an energy range, detects sign changes,
constructs brackets, and refines them with bisection. The outward solver then
applies a short safeguarded secant-style polishing step inside the final
bracket to reduce the final residual.

To make the method transparent, the project also generates:

- global mismatch scans over the full energy interval
- zoomed plots around individual brackets with the bisection history overlaid

## Validation strategy

The solver is validated using:

- the infinite square well, which has an analytic spectrum
- the harmonic oscillator, which has an analytic spectrum
- convergence studies with respect to grid spacing and domain size
- automated tests for normalization, benchmark energies, convergence behavior,
  parity-specific mismatch behavior, and scattering probability conservation

Two implementation details are especially important for the observed accuracy:

- the parity-based startup includes higher-order Taylor terms so the first
  Numerov step does not reduce the overall order
- the inward even-state derivative condition uses a higher-order one-sided
  stencil so the boundary evaluation does not degrade the Numerov solve

## Important numerical checks

- The Numerov derivative stencil is fourth order when enough points are
  available. This matters because even-state inward shooting uses the condition
  `psi'(0) = 0`, so a low-order derivative estimate would degrade the full
  eigenvalue solve.

- Harmonic-oscillator inward shooting starts from a decaying forbidden-region
  tail and integrates toward the origin. This suppresses contamination by the
  unphysical growing exponential mode that can spoil outward shooting on a
  truncated infinite domain.

- The quartic double well can shift its analytic minima to zero. Using the
  analytic minimum instead of the sampled grid minimum keeps the physical
  potential fixed across grid refinements and avoids fake convergence effects.

- Convergence studies are split deliberately:
  grid-refinement studies vary `h` at fixed domain size, while box-size studies
  keep `h` approximately fixed and vary `x_max`. This separates discretization
  error from boundary-truncation error.

- The RK4 comparison uses the same harmonic-oscillator setup and matched grid
  family as the Numerov study, so the comparison isolates the integration
  method rather than changing the physical problem.

- The outward bound-state solver finishes with a short safeguarded polishing
  step inside the final bracket to reduce the reported wall mismatch without
  giving up the robustness of a bracketed solve.

- Normalization is part of the physical result, not just presentation. The code
  uses numerical quadrature so the returned wavefunctions satisfy
  `integral |psi|^2 dx = 1` on the computational grid.
