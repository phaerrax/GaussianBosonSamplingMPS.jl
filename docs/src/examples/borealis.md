# Borealis experiment

In this page you can learn how to combine the various methods provided by this
package in order to simulate a typical boson-sampling experiment.
We will refer, for the sake of concreteness, to the experiment setup described
in [Madsen2022](@cite), but the simulation routine is easily adaptable: you will
most likely need to change only how the input data is read and how the
covariance matrix of the output state is computed.

## Preparation

### Package requirements

For the simulation we will need the `GaussianStates` and `LinearAlgebra`
packages, and of course `GaussianBosonSamplingMPS`.
We will also show, as an example, how to read the input and write the results
with HDF5 files, so for this reason we also need to import `ITensorMPS` and
`HDF5`, but in general this requirement may vary depending on how the input data
is stored.

### Input data

The covariance matrix of the final state is determined by a vector of squeezing
parameters, that we will call ``r``, and a "transfer matrix" ``T``, as
instructed in the supplementary information of Ref. [Madsen2022](@cite).
In the data provided by the authors of the experiment, we have (for ``n`` modes)
a vector of ``n`` real numbers and an ``n \times n`` complex matrix. We assume
we can find them in HDF5 files whose names are stored in `squeezepar_file` and
`Tmat_file`, respectively.

```julia
r = h5open(squeezepar_file) do hf
    read(hf, "squeeze_parameters")
end
T = h5open(Tmat_file) do hf
    read(hf, "transfer_matrix")
end
```

Let's check that we loaded matrices of compatible sizes.

```julia
if !(length(r) == size(T, 1) == size(T, 2))
    error("sizes of r and T do not match.")
end
```

## Computing the covariance matrix of the final state

First we need to compute the squeezed vacuum state, which can be done with
the `vacuumstate` and `squeeze!` functions from `GaussianStates` as follows:

```julia
n = length(r)
g0 = vacuumstate(n)
squeeze!(g0, r)
```

then we apply the transfer matrix. The matrix from the Borealis experiment
comes as an ``n \times n`` complex matrix, while we need a ``2n \times 2n`` real
matrix that can be multiplied with the covariance matrix. Let's call ``\phi_T``
this new, larger matrix.
With the `xxpp` notation for moments of Gaussian states, ``\phi_T`` would simply
be

```math
\begin{pmatrix}
  \real T & -\imag T\\
  \imag T & \real T
\end{pmatrix}
```

but since the `GaussianStates` package uses the `xpxp` notation we need to
permute its rows and columns: this can be easily done by calling
`GaussianStates.permute_to_xpxp`. In the end, we get our matrix as

```julia
ϕT = GaussianStates.permute_to_xpxp([
    real(T) -imag(T)
    imag(T) real(T)
])
```

The covariance matrix of the final state can then be obtained, from the
covariance matrix ``\sigma_{r}`` of the squeezed vacuum state, by the formula

```math
\sigma = I_{2n} - \phi_T \transpose{\phi_T} + \phi_T \sigma_{r}
\transpose{\phi_T}
```

so we can build our final Gaussian state with

```julia
σ = I - ϕT * ϕT' + ϕT * g0.covariance_matrix * ϕT'
g = GaussianState(Symmetric(σ))
```

which defines a Gaussian state with null first moments and the given covariance
matrix.
We know that `σ` is symmetric, but due to some numerical errors it may not be
so, thus we also call `Symmetric` on it in order to make it _truly_ symmetric:
this way Julia can later choose to use methods that are specific to symmetric
matrices, that can be more efficient.

## Optimisation

Here comes the optimisation part, where we rely on the SCS solver through the
`optimise` function from this package. We can choose its precision by adjusting
the `scs_eps` keyword argument.

```julia
scs_eps = 1e-8
g_opt, W = optimise(g; scs_eps=scs_eps)
```

Now we have `g_opt`, a new Gaussian state which contains (as long as ``T`` is
not unitary) fewer photons than the previous state, and `W`, a positive
semi-definite matrix that will represent a random displacement channel.

## Sampling

The ground-truth probability distribution of the boson-sampling experiment is
the function [Oh2024; Eq. (6)](@cite)

```math
p(m) = \int_{\R^{2n}} p_W(\alpha) p_{g\opt}(m \mid \alpha) \, \dd\alpha,
```

where

```math
p_W(x) = \frac{1}{(2\pi)^n \sqrt{\det W}} \, \exp\bigl(-\tfrac12
\transpose{x} W^{-1} x\bigr)
```

is a multivariate normal probability density with mean zero and covariance
matrix ``W``, and ``p_{g\opt}(m \mid \alpha)`` is the probability of getting a
Fock state with occupation numbers ``m \in \N^{2n}`` from the ``g\opt`` state.
For the purposes of these functions the displacement vector, originally ``\alpha
\in \C^n`` is seen as the vector

```math
(\real\alpha_1, \imag\alpha_1, \real\alpha_2,
\imag\alpha_2, \dotsc, \real\alpha_n, \imag\alpha_n) \in \R^{2n}``.
```

### MPS creation

The first thing we need to do is to approximate `g_opt` with a matrix-product
state: to do this, we call the `MPS` function on the Gaussian state as follows:

```julia
v = MPS(g_opt; maxdim=10, maxnumber=4, purity_atol=10*scs_eps)
```

We need to specify the (maximum) bond dimension of the MPS, `maxdim`, as well as
the dimension of the local Hilbert space by setting the maximum allowed
occupation number `maxnumber` (the dimension will be `maxnumber` plus one).

!!! warning "Keep the local dimension low"
    As the Franck-Condon formula in Ref. [Quesada2019](@cite) involves the
    calculation of factorials of numbers up to the local dimension, it is
    recommended to choose a relatively low `maxnumber`. The authors in Ref.
    [Oh2024](@cite) report that `maxnumber = 4` is enough, at this stage of the
    calculations, to obtain sensible results.

!!! info "Tolerance for intermediate checks"
    The optional `purity_atol` argument can be passed to the function in order
    to adjust the tolerance of some intermediate checks, which assume that the
    input Gaussian state is pure. The optimisation routine however does not
    output a perfectly pure state: mainly, some of its symplectic eigenvalues
    may be less than 1, usually by the same order as `scs_eps`. A positive
    `purity_atol` value can be used to make these checks less strict.

Now is a good time to save the intermediate results, before going on to the
sampling stage. This is an example of how we can bundle the initial data and the
MPS of the optimised Gaussian state in an HDF5 file: if `outputfile` contains
the name of the output file,

```julia
h5open(outputfile, "w") do hf
    write(hf, "squeeze_parameters", r)
    write(hf, "transfer_matrix", T)
    write(hf, "final_state", v)
end
```

### Application of random displacements

Before applying the displacement operators, we first need to increase the local
dimensions of the Hilbert spaces. These operators are local, they consist in
taking a linear combination of the matrices on each site separately, therefore
it is a very efficient operation given the MPS form of the state, regardless of
the local dimension.
However, the displacement may increase the (local) mean number of photons,
therefore we need a larger Hilbert space in order to faithfully represent the
displaced state.
This can be done by using the `enlargelocaldim` function: here we increase the
local dimension, uniformly, to 100.

```julia
v = enlargelocaldim(v, 100)
```

Finally, we use the `sample_displaced` function to sample from the probability
distribution ``p`` defined above:

```julia
samples, displacements = sample_displaced(
    v,
    W;
    nsamples=1000,
    nsamples_per_displacement=10,
    eval_atol=10*scs_eps,
)
```

This function draws a displacement vector from ``p_W``, applies such
displacement to the state then samples from it a certain number of times; then,
it draws a new displacement vector, and so on.
We decide now many samples we want (in total) with `nsamples`, while
`nsamples_per_displacement` controls how many samples are drawn for each
displacement vector.

!!! info
    If `nsamples` is not a multiple of `nsamples_per_displacement` then we will
    actually get `nsamples_per_displacement * floor(nsamples /
    nsamples_per_displacement)` total samples.

Once the sampling routine is over, `samples` is a matrix of natural numbers
whose columns are the samples, corresponding to the multiindices ``m`` in the
equations above; `displacements` is a complex matrix whose columns are the
displacement vectors sampled from ``W``.

!!! info "Tolerance for negative eigenvalues"
    As with the ``g\opt`` Gaussian state, the ``W`` matrix may end up not being
    positive semi-definite due to the finite working precision of the
    optimisation routine: usually it has some positive and negative eigenvalues
    of the order of `scs_eps` which should instead be zero (you can see that
    they get closer and closer to zero as you decrease `scs_eps`). Since the
    normal multivariate distribution cannot be defined if ``W`` is not positive
    semi-definite, the `eval_atol` parameter can be adjusted to replace all
    eigenvalues smaller than this threshold to zero.

Now the simulation is complete: we can append the results to the HDF5 file we
created previously by doing

```julia
h5open(outputfile, "cw") do hf
    write(hf, "displacements", displacements)
    write(hf, "samples", samples)
end
```
