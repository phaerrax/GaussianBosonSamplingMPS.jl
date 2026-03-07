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

```julia-repl
using GaussianStates, LinearAlgebra, GaussianBosonSamplingMPS, HDF5, ITensorMPS
```

### Input data

The covariance matrix of the final state is determined by a vector of squeezing
parameters, that we will call ``r``, and a "transfer matrix" ``T``, as
instructed in the supplementary information of Ref. [Madsen2022](@cite).
In the data provided by the authors of the experiment, we have (for ``n`` modes)
a vector of ``n`` real numbers and an ``n \times n`` complex matrix, which can
be downloaded from [this GitHub
repository](https://github.com/XanaduAI/xanadu-qca-data).
In this tutorial, we will use our package on the `fig2` data, that describes a
16-mode instance, displayed in Fig. 2 of [Madsen2022](@cite) and Figs. S4 and S5
of the relative Supplementary Information.

To download the data, after cloning the XanaduAI repository, we edit
`dowload_data.py` file so that `folders_list = ["fig2/"]`, in order to download
only the data we need.
Once the download is complete, in the `qca-data/fig2` folder we should find,
among other files, the vector of squeezing values `r.npy` and the transfer
matrix `T.npy`. We can read, for example, them with the help of the `npzread`
method, from the `NPZ` Julia package.

```julia-repl
julia> using NPZ

julia> r = npzread("qca-data/fig2/r.npy");

julia> T = npzread("qca-data/fig2/T.npy");
```

Let's check that we loaded matrices of compatible sizes.

```julia-repl
julia> length(r) == size(T, 1) == size(T, 2)
true
```

## Computing the covariance matrix of the final state

First we need to compute the initial squeezed vacuum state. We can do this with
the `vacuumstate` and `squeeze!` functions from `GaussianStates` as follows:

```julia-repl
julia> n = length(r)
16

julia> g0 = vacuumstate(n)
GaussianState([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0 0.0 … 0.0 0.0; 0.0 1.0 … 0.0 0.0; … ; 0.0 0.0 … 1.0 0.0; 0.0 0.0 … 0.0 1.0])

julia> squeeze!(g0, r)
GaussianState([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [5.901118952151876 0.0 … 0.0 0.0; 0.0 0.16945938695835716 … 0.0 0.0; … ; 0.0 0.0 … 5.814360514405247 0.0; 0.0 0.0 … 0.0 0.17198795938478037])
```

Now, we apply the transfer matrix. The matrix from the Borealis experiment comes
as an ``n \times n`` complex matrix, while we need a ``2n \times 2n`` real
matrix that can be combined with the covariance matrix ``\sigma_{r}`` of the
squeezed vacuum state in the formula

```math
\sigma = I_{2n} - \phi_T \transpose{\phi_T} + \phi_T \sigma_{r}
\transpose{\phi_T}
```

to obtain the covariance matrix of the final state.
Let's call ``\phi_T`` this ``2n \times 2n`` matrix obtained from ``T``. With
the `xxpp` notation for moments of Gaussian states, ``\phi_T`` would be given by

```math
\begin{pmatrix}
  \real T & -\imag T\\
  \imag T & \real T
\end{pmatrix}
```

but the `GaussianStates` package uses the `xpxp` notation, therefore we need
first to permute its rows and columns. We can do this by calling
`GaussianStates.permute_to_xpxp`. In the end, we have

```julia-repl
julia> ϕT = GaussianStates.permute_to_xpxp([
           real(T) -imag(T)
           imag(T) real(T)
       ]);
```

so we can build our final Gaussian state with

```julia-repl
julia> σ = I - ϕT * ϕT' + ϕT * covariancematrix(g0) * ϕT';

julia> g = GaussianState(Symmetric(σ));
```

which defines a Gaussian state with (implied) null first moments and the given
covariance matrix.
The `σ` matrix is supposedly symmetric, but due to some numerical errors it may
not be so, thus we also call `Symmetric` on it in order to make it _truly_
symmetric: this way Julia can later choose to use methods that are specific to
symmetric matrices, that can be more efficient.

## Optimisation

Here comes the optimisation part, where we rely on the SCS solver from the
`JuMP` and `SCS` packages.  The ``\sigma = \sigma\pure + W`` decomposition is
handled by the `optimise` method, which takes the initial Gaussian state object
as an input, and returns a tuple with the optimised Gaussian state and the ``W``
matrix.  The precision of the decomposition can be adjusted with the `scs_eps`
keyword argument.
We can check that the optimised state contains a smaller mean number of photons.

```julia-repl
julia> scs_eps = 1e-8;

julia> g_opt, W = optimise(g; scs_eps=scs_eps);

julia> number(g)
5.982489729474189

julia> number(g_opt)
0.5487202626383336
```

!!! warning "Finite precision of the optimisation routine"

    One consequence of the finite precision of the optimisation routine is that
    ``W`` is not actually positive-definite, as you can see by checking its
    eigenvalues.

    ```julia_repl
    julia> evals, _ = eigen(W);
    julia> evals
    32-element Vector{Float64}:
     -3.0409357106623564e-8
     -2.2535396393836932e-8
     -1.8680166847751422e-8
     -1.609116119865202e-8
      ⋮
      1.4965033093249647
      1.5055800742043126
      1.5247063427002268
      1.5293441728736894
    ```

    Some of them are negative: you can expect that they will be of the same
    order of magnitude of `scs_eps`.
    At the same time, `g_opt` will not be exactly pure, but `1 - parity(g_opt)`
    will roughly be of the same order of magnitude of `scs_eps`.

Now we have ``g\opt``, the covariance matrix of a new Gaussian state which
contains (as long as ``T`` is not unitary) fewer photons than the previous
state, and ``W``, a positive semi-definite matrix representing a random
displacement channel.  Note that the first moments of the Gaussian state are
unaffected by the decomposition, thus `firstmoments(g_opt) == firstmoments(g)`
(which in our case are all zero).

## Sampling

The ground-truth probability distribution of the boson-sampling experiment is
the function [Oh2024; Eq. (6)](@cite)

```math
p(m) = \int_{\R^{2n}} p_W(\alpha) p_{g\opt}(m \mid \alpha) \, \dd\alpha,
```

in which

```math
p_W(x) = \frac{1}{(2\pi)^n \sqrt{\det W}} \, \exp\bigl(-\tfrac12
\transpose{x} W^{-1} x\bigr)
```

is a multivariate normal probability density with null mean and covariance
matrix ``W``, and ``p_{g\opt}(m \mid \alpha) = \abs{ \bra{f_m}
\displacement{\alpha} \ket{\psi\pure} }^2`` where ``\ket{f_m}`` is the element
of the Fock basis with occupation numbers ``m = (m_1,m_2,\cdots,m_n) \in \N^n``,
``\displacement{\alpha}`` the operator that displaces by ``\alpha``, and
``\ket{\psi\pure}`` is the pure state associated to the covariance matrix
``g\opt``.

### MPS creation

Once we have the optimised covariance matrix, we need to approximate `g_opt`
with a matrix-product state. To do this, we call the `MPS` function on the
Gaussian state as follows, obtaining a matrix-product state with the specified
dimensions as a result.

```julia-repl
julia> v = MPS(g_opt; maxdim=10, maxnumber=4, purity_atol=10*scs_eps)
Computing partial normal-mode decompositions... 100%|██████████████████| Time: 0:00:01
Computing MPS matrices... 100%|████████████████████████████████████████| Time: 0:00:18
16-element MPS:
 ((dim=5|id=205|"Boson,Site,n=1"), (dim=10|id=190|"Link,l=1"))
 ((dim=5|id=823|"Boson,Site,n=2"), (dim=10|id=190|"Link,l=1"), (dim=10|id=254|"Link,l=2"))
 ((dim=5|id=937|"Boson,Site,n=3"), (dim=10|id=254|"Link,l=2"), (dim=10|id=113|"Link,l=3"))
 ((dim=5|id=403|"Boson,Site,n=4"), (dim=10|id=113|"Link,l=3"), (dim=10|id=358|"Link,l=4"))
 ((dim=5|id=414|"Boson,Site,n=5"), (dim=10|id=358|"Link,l=4"), (dim=10|id=475|"Link,l=5"))
 ((dim=5|id=643|"Boson,Site,n=6"), (dim=10|id=475|"Link,l=5"), (dim=10|id=725|"Link,l=6"))
 ((dim=5|id=5|"Boson,Site,n=7"), (dim=10|id=725|"Link,l=6"), (dim=10|id=50|"Link,l=7"))
 ((dim=5|id=406|"Boson,Site,n=8"), (dim=10|id=50|"Link,l=7"), (dim=10|id=307|"Link,l=8"))
 ((dim=5|id=94|"Boson,Site,n=9"), (dim=10|id=307|"Link,l=8"), (dim=10|id=500|"Link,l=9"))
 ((dim=5|id=725|"Boson,Site,n=10"), (dim=10|id=500|"Link,l=9"), (dim=10|id=935|"Link,l=10"))
 ((dim=5|id=562|"Boson,Site,n=11"), (dim=10|id=935|"Link,l=10"), (dim=10|id=849|"Link,l=11"))
 ((dim=5|id=150|"Boson,Site,n=12"), (dim=10|id=849|"Link,l=11"), (dim=10|id=600|"Link,l=12"))
 ((dim=5|id=790|"Boson,Site,n=13"), (dim=10|id=600|"Link,l=12"), (dim=10|id=950|"Link,l=13"))
 ((dim=5|id=652|"Boson,Site,n=14"), (dim=10|id=950|"Link,l=13"), (dim=10|id=308|"Link,l=14"))
 ((dim=5|id=944|"Boson,Site,n=15"), (dim=10|id=308|"Link,l=14"), (dim=5|id=706|"Link,l=15"))
 ((dim=5|id=534|"Boson,Site,n=16"), (dim=5|id=706|"Link,l=15"))

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
    input Gaussian state is pure. The optimisation routine, as we saw before,
    does not output a perfectly pure state: mainly, some of its symplectic
    eigenvalues may be less than 1, usually by the same order as `scs_eps`. A
    positive `purity_atol` value can be used to make these checks less strict. A
    value equal to ten times `scs_eps` is usually enough for the state to pass
    the test.

Now is a good time to save the intermediate results, before going on to the
sampling stage. Here is an example of how we can bundle the initial data and the
MPS of the optimised Gaussian state in an HDF5 file: if `outputfile` contains
the name of the output file, then we can run

```julia-repl
julia> outputfile = "borealis_gbs_output.h5";

julia> h5open(outputfile, "w") do hf
           write(hf, "squeeze_parameters", r)
           write(hf, "transfer_matrix", T)
           write(hf, "final_state", v)
       end
```

### Application of random displacements

#### Enlargement of the Hilbert space

The displacement operation may increase the (local) mean number of photons,
therefore we need a larger Hilbert space in order to faithfully represent the
resulting state.
Before applying the displacement, we need then to increase the local
dimensions of the Hilbert spaces.
This can be done by using the `enlargelocaldim` function: here we increase the
local dimension, uniformly, to 100, replacing the previous MPS object.

```julia-repl
julia> v = enlargelocaldim(v, 100)
16-element MPS:
 ((dim=10|id=876|"Link,l=1"), (dim=100|id=974|"Boson,Site,n=1"))
 ((dim=10|id=876|"Link,l=1"), (dim=100|id=663|"Boson,Site,n=2"), (dim=10|id=184|"Link,l=2"))
 ((dim=10|id=184|"Link,l=2"), (dim=100|id=566|"Boson,Site,n=3"), (dim=10|id=243|"Link,l=3"))
 ((dim=10|id=243|"Link,l=3"), (dim=100|id=664|"Boson,Site,n=4"), (dim=10|id=22|"Link,l=4"))
 ((dim=10|id=22|"Link,l=4"), (dim=100|id=161|"Boson,Site,n=5"), (dim=10|id=840|"Link,l=5"))
 ((dim=10|id=840|"Link,l=5"), (dim=100|id=362|"Boson,Site,n=6"), (dim=10|id=153|"Link,l=6"))
 ((dim=10|id=153|"Link,l=6"), (dim=100|id=694|"Boson,Site,n=7"), (dim=10|id=545|"Link,l=7"))
 ((dim=10|id=545|"Link,l=7"), (dim=100|id=35|"Boson,Site,n=8"), (dim=10|id=289|"Link,l=8"))
 ((dim=10|id=289|"Link,l=8"), (dim=100|id=348|"Boson,Site,n=9"), (dim=10|id=577|"Link,l=9"))
 ((dim=10|id=577|"Link,l=9"), (dim=100|id=403|"Boson,Site,n=10"), (dim=10|id=396|"Link,l=10"))
 ((dim=10|id=396|"Link,l=10"), (dim=100|id=126|"Boson,Site,n=11"), (dim=10|id=439|"Link,l=11"))
 ((dim=10|id=439|"Link,l=11"), (dim=100|id=23|"Boson,Site,n=12"), (dim=10|id=817|"Link,l=12"))
 ((dim=10|id=817|"Link,l=12"), (dim=100|id=69|"Boson,Site,n=13"), (dim=10|id=117|"Link,l=13"))
 ((dim=10|id=117|"Link,l=13"), (dim=100|id=967|"Boson,Site,n=14"), (dim=10|id=666|"Link,l=14"))
 ((dim=10|id=666|"Link,l=14"), (dim=100|id=951|"Boson,Site,n=15"), (dim=5|id=951|"Link,l=15"))
 ((dim=5|id=951|"Link,l=15"), (dim=100|id=685|"Boson,Site,n=16"))
```

Note how the dimension of the `Site` indices has been increased to 100, as
requested.

The displacements are local operations on each mode, therefore they are a very
efficient operation on the MPS form of the state, regardless of the local
dimension: for this reason, we can be generous when defining the new, larger
dimension.

#### Sampling routine

Finally, we use the `sample_displaced` method to sample from the probability
distribution ``p`` defined above:

```julia-repl
julia> samples, displacements = sample_displaced(
           v,
           W;
           nsamples=1000,
           nsamples_per_displacement=10,
           eval_atol=10*scs_eps,
       )
┌ Warning: sample_displaced: MPS is not normalised, norm=0.9922811919153979. Continuing with a normalised MPS.
└ @ GaussianBosonSamplingMPS /opt/julia/dev/GaussianBosonSamplingMPS/src/sampling.jl:31
Sampling... 100%|████████████████████████████████████████████████████| Time: 0:02:02
```

The MPS sampling algorithm requires the state to be normalised, so if the norm
of the state is not 1 (as in this example, due to the finite numerical precision
of some intermediate steps) the function will print a warning, and then it will
proceed with the normalised state.

The `sample_displaced` method draws a displacement vector from ``p_W``, applies
such displacement to the state then samples from it a certain number of times;
then, it draws a new displacement vector, and so on.

!!! info "Tolerance for negative eigenvalues"

    As with the ``g\opt`` Gaussian state, the ``W`` matrix may end up not being
    positive semi-definite due to the finite working precision of the
    optimisation routine: usually it has some positive and negative eigenvalues
    of the order of `scs_eps` which should instead be zero (you can see that
    they get closer and closer to zero as you decrease `scs_eps`). Since the
    normal multivariate distribution cannot be defined if ``W`` is not positive
    semi-definite, the `eval_atol` parameter can be adjusted to replace all
    eigenvalues smaller than this threshold to zero.

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

```julia-repl
julia> samples
16×1000 Matrix{UInt8}:
 0x00  0x00  0x00  0x01  0x02  0x01  0x00  …  0x00  0x00  0x00  0x01  0x00  0x00  0x00
 0x00  0x00  0x00  0x00  0x02  0x00  0x01     0x00  0x00  0x00  0x00  0x01  0x00  0x00
 0x01  0x00  0x00  0x00  0x00  0x02  0x01     0x00  0x00  0x00  0x00  0x01  0x00  0x00
 0x01  0x02  0x00  0x00  0x00  0x00  0x03     0x00  0x00  0x00  0x00  0x00  0x00  0x00
 0x00  0x01  0x00  0x00  0x00  0x02  0x01     0x00  0x01  0x00  0x00  0x00  0x00  0x00
 0x00  0x00  0x00  0x00  0x00  0x00  0x00  …  0x01  0x01  0x01  0x00  0x00  0x00  0x01
 0x00  0x01  0x00  0x02  0x00  0x01  0x00     0x03  0x00  0x02  0x02  0x01  0x01  0x03
 0x01  0x02  0x00  0x00  0x00  0x01  0x02     0x00  0x00  0x01  0x00  0x00  0x02  0x00
 0x01  0x01  0x03  0x01  0x00  0x02  0x02     0x01  0x00  0x02  0x02  0x01  0x02  0x00
 0x00  0x00  0x00  0x00  0x00  0x00  0x00     0x00  0x00  0x01  0x01  0x01  0x00  0x01
 0x00  0x02  0x02  0x01  0x01  0x02  0x02  …  0x01  0x01  0x00  0x00  0x00  0x00  0x01
 0x00  0x01  0x00  0x00  0x00  0x00  0x00     0x00  0x00  0x00  0x01  0x00  0x01  0x01
 0x00  0x00  0x00  0x01  0x00  0x00  0x00     0x02  0x00  0x00  0x00  0x01  0x03  0x01
 0x01  0x00  0x01  0x02  0x02  0x01  0x00     0x01  0x00  0x00  0x00  0x01  0x00  0x02
 0x01  0x01  0x00  0x01  0x01  0x01  0x00     0x00  0x01  0x00  0x00  0x00  0x01  0x00
 0x00  0x00  0x00  0x00  0x00  0x00  0x00  …  0x01  0x00  0x00  0x00  0x00  0x00  0x00
```

Each column of the `sample` matrix represent a sample from the output of the
experiment.
The samples are stored as 8-bit integers, as we expect to find relatively small
integers. A more human-readable result can be recovered for example by calling
`Int.(samples)`, which converts the matrix elements into integers of the default
integer type.

```julia-repl
julia> displacements
16×100 Matrix{ComplexF64}:
  -0.518011+0.115858im   -0.875232+0.130387im   …   -0.280419-0.0661442im
    0.27752-0.985331im   -0.512708-0.73929im         0.170241+0.606069im
  -0.841006+0.756125im     0.26727+0.567904im        0.441435-0.246802im
  -0.615843+0.283093im   -0.994317-0.100698im         0.20366-0.139411im
  -0.241198-0.850565im   -0.370491+0.0677832im      -0.379553-0.20374im
   0.423471-0.0819535im  -0.914632-0.482326im   …   -0.207274+0.430511im
   0.447169+0.122068im    0.614277+0.287048im        -1.15232+0.0466532im
 -0.0865161+1.24865im    0.0429178+0.261422im        -0.55691-0.352332im
  -0.608897-0.981013im   -0.261221+0.277866im      -0.0702052+0.986501im
 -0.0522963-0.139707im   -0.536141-0.529169im       -0.348973-0.563379im
  -0.264722-1.30377im     0.642328-0.0867902im  …   -0.410557+0.39392im
   0.591988-0.529414im   -0.764109-0.0892786im      -0.134641-0.761003im
    0.76421+0.2773im      0.124114+0.653305im       -0.614632-0.714693im
    1.03728+0.215339im    0.324581-0.494222im       -0.240613+0.811865im
  -0.876219-0.124157im   -0.783585-0.32726im       -0.0554169-0.513018im
   0.339354+0.0148097im  -0.218598-0.175292im   …   -0.660401+0.104063im
```

The columns of `displacements` are the displacement vectors drawn from ``p_W``
during the sampling routine. The first column is associated to the samples (i.e.
the columns of the `samples` matrix) from 1 to `nsamples_per_displacement`, the
second column is associated to `nsamples_per_displacement+1` to
`2*nsamples_per_displacement`, and so on.

The samples follow the ``p`` distribution obtained by combining ``p_W`` and
``p_{g\opt}`` as in the equation above.
Now the simulation is complete: we can append the results to the HDF5 file we
created previously by running

```julia
julia> h5open(outputfile, "cw") do hf
           write(hf, "displacements", displacements)
           write(hf, "samples", samples)
       end
```
