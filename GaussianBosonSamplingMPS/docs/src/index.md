# GaussianBosonSamplingMPS.jl

*This is the documentation for the GaussianBosonSamplingMPS.jl package.*

This package implements the tensor-network-based algorithm for simulating
Gaussian boson sampling experiments proposed in [1], and of the algorithm
in [2] for computing elements of Gaussian operations in the Fock number
basis.

The functions for computing hafnians and loop hafnians of square matrices are
ported from _The Walrus_ [3].

**References**:

1. Changun Oh, Minzhao Liu, Yuri Alexeev, Bill Ferrerman and Liang Jiang.
   ‘Classical algorithm for simulating experimental Gaussian boson sampling’.
   [_Nature Physics_ 20, 1461–1468 (2024)](https://doi.org/10.1038/s41567-024-02535-8)
2. Nicolás Quesada. ‘Franck-Condon factors by counting perfect matchings of
   graphs with loops’.
   [_The Journal of Chemical Physics_ 150.16 (2019)](https://doi.org/10.1063/1.5086387)
3. Brajesh Gupt, Josh Izaac and Nicolás Quesada. ‘The Walrus: a library for the
   calculation of hafnians, Hermite polynomials and Gaussian boson sampling.’
   [_Journal of Open Source Software_, 4(44), 1705 (2019)](https://joss.theoj.org/papers/10.21105/joss.01705)
4. Manfred Schmutz. ‘Real-Time Green’s Functions in Many Body Problems’.
   [_Zeitschrift für Physik B Condensed Matter_, 30.1 (1978)](https://doi.org/10.1007/BF01323673)

## Package features

- Simulate a linear optical quantum computer with matrix-product states (MPS)
  through the application of common operations such as one- and two-mode
  squeezing, beam splitters, etc. and also simulate losses through an attenuator
  channel.
- Manipulate MPS (and sample from them) representing mixed states in the
  superboson formalism [4].
- Find an approximate MPS representation of a Gaussian state in the Fock basis
- Sample from the outcome of a lossy Gaussian boson sampling experiment with
  the classical MPS-based algorithm described in [1].

## Public functions

```@autodocs
Modules = [GaussianBosonSamplingMPS]
Order   = [:function, :type]
```
