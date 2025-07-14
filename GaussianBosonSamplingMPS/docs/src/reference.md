# Reference

## Superboson matrix-product states

The package introduces a new type, called `SuperBosonMPS`, which is basically
an ordinary `MPS` from ITensors with some additional decoration.
This MPS represents a generic mixed state in a many-body bosonic Fock space
using the _superboson_ formalism [Schmutz1978](@cite), which translates mixed
states to pure states in an enlarged Hilbert space.

It is a subtype of `AbstractMPS`, therefore you can use most of the methods
already defined by ITensors on the `SuperBosonMPS` type too.

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

```@docs
attenuate
displace
squeeze
beamsplitter
```

## Boson sampling output simulation

Here lies the heart of this package: the following methods implement the
algorithms in [Oh2024](@cite) and [Quesada2019](@cite) in order to be able to
simulate a Gaussian boson sampling experiment with matrix-product states.

The main functions are the following.

- `optimise` uses semi-definite programming, through the JuMP and SCS libraries,
  to decompose the covariance matrix of a Gaussian state into a sum of
    * a pure covariance matrix which "contains" a smaller amount of photons,
    * a positive semi-definite matrix
  following the procedure detailed in [Oh2024](@cite).
- `MPS` computes a matrix-product state that approximates a pure Gaussian state
  following the algorithm presented in [Oh2024](@cite), by using the
  Franck-Condon formula from [Quesada2019](@cite).
- `sample_displaced` samples from a state after the application of a random
  displacement channel.

```@docs
normal_mode_decomposition
franckcondon
MPS
optimise
sample_displaced
```
