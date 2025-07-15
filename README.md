# GaussianBosonSamplingMPS

[![Code Style:
Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

Julia implementation of the tensor-network-based algorithm for simulating
Gaussian boson sampling experiments proposed in [[1]](#1), and of the algorithm
in [[2]](#2) for computing elements of Gaussian operations in the Fock number
basis, plus a set of utilities for a full MPS simulation of quantum optical
operations.

The functions for computing hafnians and loop hafnians of square matrices are
ported from _The Walrus_ [[3]](#3).

## Installation

### From a registry

This package is registered in my
[TensorNetworkSimulations](https://github.com/phaerrax/TensorNetworkSimulations)
registry. By first adding this registry, with

```julia
using Pkg
pkg"registry add https://github.com/phaerrax/TensorNetworkSimulations.git"
```

(this must be done just once per Julia installation) the package can then be
installed as a normal one:

```julia
using Pkg
pkg"add GaussianBosonSamplingMPS"
```

### From GitHub

Alternatively, straight installation from GitHub is also possible:

```julia
using Pkg
pkg "add https://github.com/phaerrax/GaussianBosonSamplingMPS.jl"
```

## References

<a id="1">[1]</a>
Changun Oh, Minzhao Liu, Yuri Alexeev, Bill Ferrerman and Liang Jiang,
‘Classical algorithm for simulating experimental Gaussian boson sampling’.
[_Nature Physics_ 20, 1461–1468 (2024)](https://doi.org/10.1038/s41567-024-02535-8)

<a id="2">[2]</a>
Nicolás Quesada.
‘Franck-Condon factors by counting perfect matchings of graphs with loops’.
[_The Journal of Chemical Physics_ 150.16
(2019)](https://doi.org/10.1063/1.5086387)

<a id="3">[3]</a>
Brajesh Gupt, Josh Izaac and Nicolás Quesada.
‘The Walrus: a library for the calculation of hafnians, Hermite polynomials and
Gaussian boson sampling.’
[_Journal of Open Source Software_, 4(44), 1705
(2019)](https://joss.theoj.org/papers/10.21105/joss.01705)
