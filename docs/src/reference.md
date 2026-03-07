# Reference

This package contains two main parts.

1. The definition of a matrix-product state type to represent mixed states as
   pure states in an enlarged Hilbert space, through a purification technique
   called superboson formalism, described in
   [Superboson matrix-product states](@ref).
   The package offers functions to manage these objects and extract physical
   information from them, as well as methods that implement many Gaussian
   operations on states in this representation (see
   [Gaussian states and operations](@ref)).
2. The implementation of the algorithm in [Oh2024](@cite) that simulates the
   outcome of a Gaussian boson-sampling experiment, given the covariance matrix
   of the output state of the lossy apparatus. The methods pertaining to this
   part are explained in [Gaussian boson-sampling output simulation](@ref).

## Superboson matrix-product states

This package implements the _superboson_ formalism [Schmutz1978](@cite) in a
many-body bosonic Fock space, which is a purification method with which
``N``-mode mixed states can be represented as pure states in a ``2N``-mode
bosonic Fock space.
To each original, _physical_ mode we associate an artificial _ancillary_
mode in the enlarged space, and in the end we recover physical information by
tracing away the ancillary degrees of freedom.

In order to work with this representation, we introduce a new type, called
`SuperBosonMPS`, which is essentially an ordinary `MPS` from ITensors with some
additional decoration and dedicated functions. This type is a subtype of
`AbstractMPS`, so most of the methods already defined by ITensors to work on
MPSs (such as `expect`, `sample`, and so on) can be seamlessly called on
`SuperBosonMPS` objects, too.

### Constructors

```@docs
SuperBosonMPS
```

### Utility functions

To make working with a superboson space easier, this library provides the
following utilities.

```@docs
nmodes
enlargelocaldim
sb_siteinds
sb_outer
```

### Expectation values and sampling

To extract physical information from a `SuperBosonMPS` object, you can use the
`expect`, `correlation_matrix` and `sample` functions, which work exactly as in
the ITensorMPS library.

```@docs
expect
correlation_matrix
sample
```

## Gaussian states and operations

This package works in tandem with the
[GaussianStates](https://github.com/phaerrax/GaussianStates.jl/) package,
extending some of its methods to define them on matrix-product states (both
ordinary and superbosonic ones).

### First and second moments

```@docs
firstmoments
covariancematrix
```

### Gaussian operations

The following methods can be applied to an MPS in order to simulate quantum
optical operations.
The matrix elements of the squeezing operator are taken from
[Marian1992:squeezing_fock_coefficients](@cite).

!!! warning

    The operators are defined through their exact coefficients in the
    infinite-dimensional case, that is, they are the truncated version of the
    “true” physical operators. As such, they are not unitary, and care must be
    taken so that the finite-dimensional Hilbert space is large enough to
    contain the dynamics.

```@docs
attenuate
displace
squeeze
squeeze2
beamsplitter
```

## Gaussian boson-sampling output simulation

Here lies the heart of this package: the following methods implement the
algorithms in [Oh2024](@cite) and [Quesada2019](@cite) in order to be able to
simulate a Gaussian boson sampling experiment with matrix-product states.

The main functions are the following.

- `optimise` uses semi-definite programming, through the
  [JuMP](https://jump.dev/JuMP.jl/stable/) and
  [SCS](https://www.cvxgrp.org/scs/) libraries, to decompose the covariance
  matrix of a Gaussian state into a sum of
    * a pure covariance matrix which "contains" a smaller amount of photons,
    * a positive semi-definite matrix
  following the procedure detailed in [Oh2024](@cite).
- `MPS` computes a matrix-product state that approximates a pure Gaussian state
  following the algorithm presented in [Oh2024](@cite), by using the
  Franck-Condon formula from [Quesada2019](@cite).
- `sample_displaced` samples from a state after the application of a random
  displacement channel.

See [Borealis experiment](@ref) for a tutorial on how to use these methods.

```@docs
normal_mode_decomposition
franckcondon
MPS
optimise
sample_displaced
```
