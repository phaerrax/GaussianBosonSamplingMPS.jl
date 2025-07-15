# GaussianBosonSamplingMPS.jl

*This is the documentation for the GaussianBosonSamplingMPS.jl package.*

This package implements the tensor-network-based algorithm for simulating
Gaussian boson sampling experiments proposed in [Oh2024](@cite), and of the
algorithm in [Quesada2019](@cite) for computing elements of Gaussian operations
in the Fock number basis.

The functions for computing hafnians and loop hafnians of square matrices are
ported from _The Walrus_ [Gupt2019](@cite).

## Package features

- Simulate a linear optical quantum computer with matrix-product states (MPS)
  through the application of common operations such as one- and two-mode
  squeezing, beam splitters, etc. and also simulate losses through an attenuator
  channel.
- Manipulate MPS (and sample from them) representing mixed states in the
  superboson formalism [Schmutz1978](@cite).
- Find an approximate MPS representation of a Gaussian state in the Fock basis.
- Sample from the outcome of a lossy Gaussian boson sampling experiment with
  the classical MPS-based algorithm described in [Oh2024](@cite).

## Bibliography

```@bibliography
```
