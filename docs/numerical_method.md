}
# Numerical Method

We solve the 1D time-independent Schrödinger equation:

ψ''(x) = 2 (V(x) - E) ψ(x)

## Workflow

1. Choose a trial energy E  
2. Construct q(x) = 2(V(x) - E)  
3. Integrate using the Numerov method  
4. Evaluate boundary mismatch  
5. Use bisection to find eigenvalues  
6. Normalize the wavefunction  

## Shooting methods

- Outward shooting: used for finite-domain problems  
- Inward shooting: used for unbounded potentials (e.g. harmonic oscillator)

## Validation

The solver is validated using:
- infinite square well (analytic solution)
- harmonic oscillator (analytic solution)
- convergence studies with grid refinemen
# Numerical Method

We solve the one-dimensional time-independent Schrödinger equation:

```text
psi''(x) = 2 [V(x) - E] psi(x)
```

The goal is to find the special energies `E` for which the wavefunction satisfies the required boundary behavior. This makes the problem a boundary-value eigenvalue problem.

## Workflow

1. Choose a trial energy `E`.
2. Construct `q(x) = 2 [V(x) - E]`.
3. Integrate the wavefunction using the Numerov method.
4. Evaluate a boundary mismatch function.
5. Use bracketing and bisection to find energies where the mismatch vanishes.
6. Reconstruct and normalize the wavefunction.

## Shooting methods

- **Outward shooting**  
  Used for finite-domain or effectively finite-domain symmetric bound-state problems. It starts from the exact parity conditions at the origin.

- **Inward shooting**  
  Used for unbounded confining problems such as the harmonic oscillator. It starts from a decaying tail at large `x` and enforces the parity condition at the origin.

## Validation

The solver is validated using:

- the infinite square well, which has an analytic spectrum
- the harmonic oscillator, which has an analytic spectrum
- convergence studies with respect to grid spacing and domain size
- automated tests for normalization, benchmark energies, convergence behavior, and scattering probability conservation