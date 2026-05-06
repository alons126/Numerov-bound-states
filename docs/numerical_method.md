# Numerical Method

The project uses dimensionless units throughout, with $\hbar=1$ and $m=1$, so the equations and benchmark spectra are written in the same rescaled form used by the code. We solve the one-dimensional time-independent Schrödinger equation:

$$
\psi''(x) = 2 [V(x) - E] \psi(x)
$$

for the special energies $E$ that satisfy the required boundary behavior. This makes the problem a boundary-value eigenvalue problem rather than a standard initial-value problem. The overall numerical pipeline is:

<p align="center">
potential $\rightarrow$ Numerov integration $\rightarrow$ shooting mismatch $\rightarrow$ root finding $\rightarrow$ normalization $\rightarrow$ analysis
</p>

## Symmetry reduction

For the bound-state cases studied here, the main potentials are symmetric:

$$
V(-x) = V(x)
$$

The eigenstates can therefore be chosen to have definite parity:

- Even states satisfy $\psi(-x) = \psi(x)$ and $\psi'(0) = 0$
- Odd states satisfy $\psi(-x) = -\psi(x)$ and $\psi(0) = 0$

That symmetry allows the solver to integrate only on the half-domain $[0, x_{\max}]$ and reconstruct the negative-$x$ side by reflection.

## Workflow

1. Choose a trial energy $E$.
2. Construct $q(x) = 2 [V(x) - E]$.
3. Integrate the wavefunction with the Numerov method.
4. Evaluate a boundary mismatch function.
5. Use bracketing and bisection to locate energies where the mismatch vanishes.
6. Reconstruct and normalize the full wavefunction so that $\int |\psi|^2\,dx = 1$ on the computational grid.

## Normalization

The Numerov and shooting steps determine the shape of the wavefunction, but not its physically meaningful overall amplitude. A bound-state eigenfunction must satisfy

$$
\int |\psi(x)|^2\,dx = 1,
$$

so normalization is part of the physical solution rather than just presentation. The code therefore rescales each solved state after reconstruction on the full grid.

In practice, the normalization integral is evaluated on the sampled mesh with the trapezoid rule in [`numerov.py`](../src/numerov.py#L188). This is a natural choice here because the wavefunction is already known only at discrete grid points, and the quadrature is used only for the final rescaling step rather than to drive the eigenvalue search itself. The implementation also scales the raw wavefunction before squaring it, which helps avoid overflow when a trial shooting solution has grown very large away from an eigenvalue.

## Shooting formulations

- **Outward shooting:** Starts at $x = 0$ from the exact parity conditions and integrates toward $x_{\max}$. It is used for finite-domain or effectively boxed symmetric bound-state problems.

- **Inward shooting:** Starts from a decaying tail near $x_{\max}$ and integrates toward the origin. It is used for unbounded confining problems such as the harmonic oscillator, where outward shooting is numerically less stable.

The outward formulation is the more broadly applicable one in this project because its startup comes directly from symmetry at the origin. The inward formulation is more specialized because it assumes the chosen $x_{\max}$ already lies in a forbidden decaying region. For the harmonic oscillator, this inward setup suppresses contamination by the unphysical growing exponential mode that can spoil outward shooting on a truncated infinite domain.

## Root finding and diagnostics

For a fixed trial energy, the solver computes a mismatch $M(E)$.

- In outward shooting, the mismatch is the boundary leakage at $x = x_{\max}$.
- In inward shooting, the mismatch is the parity condition at the origin.

The code scans the mismatch over an energy range, detects sign changes, constructs brackets, and refines them with bisection. The outward solver then applies [a short safeguarded secant-style polishing step](../src/shooting.py#L614) inside the final bracket to reduce the final residual.

To visualize the eigenvalue search and bracket refinement, the project also generates:

- Global mismatch scans over the full energy interval
- Zoomed plots around individual brackets with the bisection history overlaid

## Validation strategy

The solver is validated using:

- The infinite square well, which has an analytic spectrum
- The harmonic oscillator, which has an analytic spectrum
- Convergence studies with respect to grid spacing and domain size
- Automated tests for normalization, benchmark energies, convergence behavior, parity-specific mismatch behavior, and scattering probability conservation

Two implementation details are especially important for the observed accuracy:

- The parity-based startup includes [higher-order Taylor terms](../src/shooting.py#L125) so the first Numerov step does not reduce the overall order
- The inward even-state derivative condition uses [a higher-order one-sided stencil](../src/numerov.py#L234) so the boundary evaluation does not degrade the Numerov solve. This derivative estimate is fourth order when enough points are available, which matters because even-state inward shooting enforces $\psi'(0) = 0$.

Convergence studies are split deliberately: grid-refinement studies vary $h$ at fixed domain size, while box-size studies keep $h$ approximately fixed and vary $x_{\max}$. This separates discretization error from boundary-truncation error. The quartic double well also [shifts its analytic minima to zero](../src/potentials.py#L168), using the analytic minimum rather than the sampled grid minimum so the physical potential does not drift across grid refinements.

The RK4 comparison uses the same harmonic-oscillator setup and matched grid family as the Numerov study, so the comparison isolates the integration method rather than changing the physical problem.

## Scattering extension

The scattering extension is handled separately from the bound-state workflow: it uses the same Numerov framework on a uniform grid, but for a complex-valued wavefunction with asymptotic scattering boundary conditions instead of bound-state parity conditions, as implemented in [`scattering.py`](../src/scattering.py). The code imposes a unit transmitted outgoing wave on the right, integrates backward through the barrier, and then decomposes the left asymptotic solution into incident and reflected components to extract $T(E)$ and $R(E)$.
