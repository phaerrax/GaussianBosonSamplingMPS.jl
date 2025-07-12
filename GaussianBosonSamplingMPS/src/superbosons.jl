using Random
using ITensors: OneITensor
using ITensors.SiteTypes: SiteTypes, siteind, siteinds, state
using MPSTimeEvolution: LocalOperator, domain

### Type definition

# As far as its inner structure is concerned, a SuperBosonMPS is completely equivalent
# to the "ordinary" MPS type from ITensorMPS. We need a different name so that we can define
# functions such as `expect`, `sample` on this particular kind of MPS without having to use
# a different name.

"""
    SuperBosonMPS

A finite-size matrix-product state type that represents mixed states in the superboson
formalism (Schmutz, 1978).
"""
mutable struct SuperBosonMPS <: AbstractMPS
    data::Vector{ITensor}
    llim::Int
    rlim::Int
end

sb_index(n) = 2n - 1  # mode number --> MPS site
inv_sb_index(n) = div(n + 1, 2)  # MPS site --> mode number
# (inv_sb_index ∘ sb_index).(1:N) == 1:N
# (sb_index ∘ inv_sb_index).(1:N) != 1:N

set_data(SuperBosonMPS::MPS, data::Vector{ITensor}) = SuperBosonMPS(data, A.llim, A.rlim)

### MPS <-> SuperBosonMPS conversion functions

# These functions don't really do anything, since the inner structures of SuperBosonMPSs
# and MPSs does not change, but the `add` function from ITensorMPS explicitly converts
# any AbstractMPS into an MPS before adding them (this happens with MPOs, too), so we need
# to define them.

function Base.convert(::Type{MPS}, M::SuperBosonMPS)
    return MPS(ITensorMPS.data(M); ortho_lims=ortho_lims(M))
end

function Base.convert(::Type{SuperBosonMPS}, M::MPS)
    return SuperBosonMPS(ITensorMPS.data(M); ortho_lims=ortho_lims(M))
end

### Basic constructors

# (carried over from the ones for the MPS type from ITensorMPS without almost any change)
"""
    SuperBosonMPS(v::Vector{<:ITensor})

Construct a SuperBosonMPS from a Vector of ITensors.
"""
function SuperBosonMPS(A::Vector{<:ITensor}; ortho_lims::UnitRange=1:length(A))
    return SuperBosonMPS(A, first(ortho_lims) - 1, last(ortho_lims) + 1)
end

"""
    SuperBosonMPS()

Construct an empty `SuperBosonMPS` with zero sites.
"""
SuperBosonMPS() = SuperBosonMPS(ITensor[], 0, 0)

"""
    SuperBosonMPS(N::Int)

Construct a `SuperBosonMPS` with `N` sites with default-constructed ITensors.
"""
function SuperBosonMPS(N::Int; ortho_lims::UnitRange=1:N)
    if !iseven(length(sites))
        error("SuperBosonMPS: number of sites is not even")
    end

    return SuperBosonMPS(Vector{ITensor}(undef, N); ortho_lims=ortho_lims)
end

"""
    SuperBosonMPS([::Type{ElT} = Float64, ]sites)

Construct a `SuperBosonMPS` filled with Empty ITensors of type `ElT` from a collection of
indices.
"""
function SuperBosonMPS(::Type{T}, sites::Vector{<:Index}) where {T<:Number}
    if !iseven(length(sites))
        error("SuperBosonMPS: number of sites is not even")
    end

    N = length(sites)
    v = Vector{ITensor}(undef, N)

    l = [Index(1, "Link,l=$i") for i in 1:(N - 1)]
    for ii in eachindex(sites)
        s = sites[ii]
        if ii == 1
            v[ii] = ITensor(T, l[ii], s)
        elseif ii == N
            v[ii] = ITensor(T, dag(l[ii - 1]), s)
        else
            v[ii] = ITensor(T, dag(l[ii - 1]), s, l[ii])
        end
    end
    return MPS(v)
end

function SuperBosonMPS(sites::Vector{<:Index}, args...; kwargs...)
    SuperBosonMPS(Float64, sites, args...; kwargs...)
end

"""
    SuperBosonMPS(
        ::Type{T},
        sites::Vector{<:Index},
        states::Union{Vector{String}, Vector{Int}, String, Int}
    )

Construct a product state `SuperBosonMPS` of element type `T`, having site indices `sites`,
and which corresponds to the initial state given by the array `states`. The input `states`
may be an array of strings or an array of ints recognized by the `state` function defined
for the relevant `Index` tag type. In addition, a single string or int can be input to
create a uniform state.

# Examples

```julia
N = 10
sites = sb_siteinds(; nmodes=N, maxnumber=4)
states = [isodd(n) ? "1" : "2" for n in 1:N]
psi = SuperBosonMPS(ComplexF64, sites, states)
phi = SuperBosonMPS(sites, "1")
```
"""
function SuperBosonMPS(eltype::Type{<:Number}, sites::Vector{<:Index}, states_)
    if length(sites) != 2length(states_)
        throw(
            DimensionMismatch(
                "SuperBosonMPS: Number of sites and and initial vals don't match"
            ),
        )
    end
    N = 2length(states_)
    M = SuperBosonMPS(length(states_))

    links = [Index(1; tags="Link,l=$n") for n in 1:N]

    M[1] = state(sites[1], states_[1]) * state(links[1], 1)
    M[2] = state(dag(links[1]), 1) * state(sites[2], states_[1]) * state(links[2], 1)
    for n in 3:2:(N - 2)
        M[n] =
            state(dag(links[n - 1]), 1) *
            state(sites[n], states_[inv_sb_index(n)]) *
            state(links[n], 1)
        M[n + 1] =
            state(dag(links[n]), 1) *
            state(sites[n + 1], states_[inv_sb_index(n)]) *
            state(links[n + 1], 1)
    end
    M[N - 1] =
        state(dag(links[N - 2]), 1) *
        state(sites[N - 1], states_[inv_sb_index(N)]) *
        state(links[N - 1], 1)
    M[N] = state(dag(links[N - 1]), 1) * state(sites[N], states_[inv_sb_index(N)])

    return convert_leaf_eltype(eltype, M)
end

function SuperBosonMPS(
    ::Type{T}, sites::Vector{<:Index}, state::Union{String,Integer}
) where {T<:Number}
    if !iseven(length(sites))
        error("SuperBosonMPS: number of sites is not even")
    end

    return SuperBosonMPS(T, sites, fill(state, div(length(sites), 2)))
end

function SuperBosonMPS(
    ::Type{T}, sites::Vector{<:Index}, states::Function
) where {T<:Number}
    states_vec = [states(n) for n in 1:length(states)]
    return SuperBosonMPS(T, sites, states_vec)
end

"""
    SuperBosonMPS(sites::Vector{<:Index}, states)

Construct a product state `SuperBosonMPS` having site indices `sites`, and which corresponds
to the initial state given by the array `states`. The `states` array may consist of either
an array of integers or strings, as recognized by the `state` function defined for the
relevant Index tag type.

# Examples

```julia
N = 10
sites = sb_siteinds(; nmodes=N, maxnumber=4)
states = [isodd(n) ? "1" : "0" for n in 1:N]
psi = SuperBosonMPS(sites, states)
```
"""
SuperBosonMPS(sites::Vector{<:Index}, states) = SuperBosonMPS(Float64, sites, states)

### Index getting/setting utilities

# Same as the ones for the MPS type. They are needed somewhere in the internal workings
# of the functions for the AbstractMPS types, such as `add`, so we need to define them
# for our new type.

"""
    siteind(M::SuperBosonMPS, j::Int; kwargs...)

Get the first site Index of the SuperBosonMPS. Return `nothing` if none is found.
"""
SiteTypes.siteind(M::SuperBosonMPS, j::Int; kwargs...) = siteind(first, M, j; kwargs...)

"""
    siteind(::typeof(only), M::SuperBosonMPS, j::Int; kwargs...)

Get the only site Index of the SuperBosonMPS. Return `nothing` if none is found.
"""
function SiteTypes.siteind(::typeof(only), M::SuperBosonMPS, j::Int; kwargs...)
    is = siteinds(M, j; kwargs...)
    if isempty(is)
        return nothing
    end
    return only(is)
end

"""
    siteinds(M::SuperBosonMPS)
    siteinds(::typeof(first), M::SuperBosonMPS)

Get a vector of the first site Index found on each tensor of the SuperBosonMPS.

    siteinds(::typeof(only), M::SuperBosonMPS)

Get a vector of the only site Index found on each tensor of the SuperBosonMPS. Errors if
more than one is found.

    siteinds(::typeof(all), M::SuperBosonMPS)

Get a vector of the all site Indices found on each tensor of the SuperBosonMPS. Returns a
Vector of IndexSets.
"""
SiteTypes.siteinds(M::SuperBosonMPS; kwargs...) = siteinds(first, M; kwargs...)

function ITensorMPS.replace_siteinds!(M::SuperBosonMPS, sites)
    for j in eachindex(M)
        sj = only(siteinds(M, j))
        M[j] = replaceinds(M[j], sj => sites[j])
    end
    return M
end

function ITensorMPS.replace_siteinds(M::SuperBosonMPS, sites)
    replace_siteinds!(copy(SuperBosonMPS), sites)
end

### Expectation values

# Here is where the SuperBosonMPS type starts to differ significantly from ordinary MPSs.
# These MPSs represent mixed states, so we want to calculate quantities like tr(Aρ): this
# means applying the "A" operator on the physical sites, leaving the ancillary sites alone,
# then contracting each physical site with its ancillary companion.

function _id_pairs(v::AbstractMPS)
    nmodes = div(length(v), 2)

    maxn = dim(siteind(v, 1)) - 1
    return [
        sum(
            state(siteind(v, n), string(m)) * state(siteind(v, n + 1), string(m)) for
            m in 0:maxn
        ) for n in sb_index.(1:nmodes)
    ]
end

function _id_contractions(v::AbstractMPS)
    nmodes = div(length(v), 2)
    sb_id_blocks = _id_pairs(v)
    return [
        dag(sb_id_blocks[inv_sb_index(n)]) * v[n] * v[n + 1] for n in sb_index.(1:nmodes)
    ]
end

"""
    expect(psi::SuperBosonMPS, op::AbstractString...; kwargs...)
    expect(psi::SuperBosonMPS, op::Matrix{<:Number}...; kwargs...)
    expect(psi::SuperBosonMPS, ops; kwargs...)

Given an SuperBosonMPS `psi` and a single operator name, returns a vector of the expected
value of the operator on each site of the SuperBosonMPS.

If multiple operator names are provided, returns a tuple of expectation value vectors.

If a container of operator names is provided, returns the same type of container with names
replaced by vectors of expectation values.

# Optional keyword arguments

  - `sites = 1:length(psi)`: compute expected values only for modes in the given range

# Examples

```julia
N = 10

s = sb_siteinds("Boson", N)
psi = sb_outer(random_mps(s; linkdims=4))
expect(psi, "N")  # compute for all sites
expect(psi, "N"; sites=2:4)  # compute for sites 2, 3 and 4
expect(psi, "N"; sites=3)  # compute for site 3 only (output will be a scalar)
expect(psi, ["A*Adag", "N"])  # compute A*Adag and N for all sites
expect(psi, [0 0; 0 1])  # same as expect(psi, "N") if maxnumber == 1
```
"""
function ITensorMPS.expect(psi::SuperBosonMPS, ops; sites=1:div(length(psi), 2))
    #psi = copy(psi)
    N = div(length(psi), 2)
    ElT = ITensorMPS.scalartype(psi)
    s = siteinds(psi)

    site_range = (sites isa AbstractRange) ? sites : collect(sites)
    Ns = length(site_range)
    start_site = first(site_range)

    el_types = map(o -> ishermitian(op(o, s[sb_index(start_site)])) ? real(ElT) : ElT, ops)

    sb_id_blocks = _id_pairs(psi)
    ids = _id_contractions(psi)
    tr_psi = scalar(prod(ids))
    iszero(tr_psi) && error("SuperBosonMPS has zero trace in function `expect`")

    ex = map((o, el_t) -> zeros(el_t, Ns), ops, el_types)
    for (entry, j) in enumerate(site_range)
        for (n, opname) in enumerate(ops)
            oⱼ = ITensors.adapt(ITensors.datatype(psi[j]), op(opname, s[sb_index(j)]))
            x = OneITensor()
            for n in 1:N
                if n == j
                    x *=
                        dag(apply(adj(oⱼ), sb_id_blocks[n])) *
                        psi[sb_index(n)] *
                        psi[sb_index(n) + 1]
                else
                    x *= ids[n]
                end
            end
            val = scalar(x) / tr_psi
            ex[n][entry] = (el_types[n] <: Real) ? real(val) : val
        end
    end

    if sites isa Number
        return map(arr -> arr[1], ex)
    end
    return ex
end

function ITensorMPS.expect(psi::SuperBosonMPS, op::AbstractString; kwargs...)
    return first(expect(psi, (op,); kwargs...))
end

function ITensorMPS.expect(psi::SuperBosonMPS, op::Matrix{<:Number}; kwargs...)
    return first(expect(psi, (op,); kwargs...))
end

function ITensorMPS.expect(
    psi::SuperBosonMPS, op1::AbstractString, ops::AbstractString...; kwargs...
)
    return expect(psi, (op1, ops...); kwargs...)
end

function ITensorMPS.expect(
    psi::SuperBosonMPS, op1::Matrix{<:Number}, ops::Matrix{<:Number}...; kwargs...
)
    return expect(psi, (op1, ops...); kwargs...)
end

adj(x) = swapprime(dag(x), 0 => 1)

function measure(v::AbstractMPS, l::LocalOperator)
    nmodes = div(length(v), 2)
    sb_id_blocks = _id_pairs(v)
    ids = _id_contractions(v)

    x = OneITensor()
    for n in sb_index.(1:nmodes)
        if n in domain(l)
            lop = if n + 1 in domain(l)
                # We loop over odd sites only, so we check manually that the next site
                # is in the domain of the operator.
                op(l[n], siteind(v, n)) * op(l[n + 1], siteind(v, n + 1))
            else
                op(l[n], siteind(v, n))
            end
            x *= dag(apply(adj(lop), sb_id_blocks[inv_sb_index(n)])) * v[n] * v[n + 1]
        else
            x *= ids[inv_sb_index(n)]
        end
    end
    return scalar(x)
end

function measure(v::AbstractMPS, ls::Vector{LocalOperator})
    nmodes = div(length(v), 2)

    sb_id_blocks = _id_pairs(v)
    ids = _id_contractions(v)

    results = Vector{ComplexF64}(undef, length(ls))
    for (j, l) in enumerate(ls)
        x = OneITensor()
        for n in sb_index.(1:nmodes)
            if n in domain(l)
                lop = if n + 1 in domain(l)
                    # We loop over odd sites only, so we check manually that the next site
                    # is in the domain of the operator.
                    op(l[n], siteind(v, n)) * op(l[n + 1], siteind(v, n + 1))
                else
                    op(l[n], siteind(v, n))
                end
                x *= dag(apply(adj(lop), sb_id_blocks[inv_sb_index(n)])) * v[n] * v[n + 1]
            else
                x *= ids[inv_sb_index(n)]
            end
        end
        results[j] = scalar(x)
    end
    return results
end

function measure(v::AbstractMPS, ls::Matrix{LocalOperator})
    nmodes = div(length(v), 2)

    sb_id_blocks = _id_pairs(v)
    ids = _id_contractions(v)

    results = Matrix{ComplexF64}(undef, size(ls))
    for (j, l) in enumerate(ls)
        x = OneITensor()
        for n in sb_index.(1:nmodes)
            if n in domain(l)
                lop = if n + 1 in domain(l)
                    # We loop over odd sites only, so we check manually that the next site
                    # is in the domain of the operator.
                    op(l[n], siteind(v, n)) * op(l[n + 1], siteind(v, n + 1))
                else
                    op(l[n], siteind(v, n))
                end
                x *= dag(apply(adj(lop), sb_id_blocks[inv_sb_index(n)])) * v[n] * v[n + 1]
            else
                x *= ids[inv_sb_index(n)]
            end
        end
        results[j] = scalar(x)
    end
    return results
end

function LinearAlgebra.tr(v::SuperBosonMPS)
    return scalar(prod(_id_contractions(v)))
end

function sb_siteinds(; nmodes, maxnumber)
    sites_phy = siteinds("Boson", nmodes; dim=maxnumber+1, addtags="phy")
    sites_anc = siteinds("Boson", nmodes; dim=maxnumber+1, addtags="anc")
    return collect(Iterators.flatten(zip(sites_phy, sites_anc)))
end

"""
    sb_outer(v::MPS)

Compute the projection operator ``|v⟩⟨v| / ‖v‖²``, from the MPS `v` representing a pure
state, expressed as an MPS (of double the size) in the superboson formalism.
"""
function sb_outer(v)
    # 1) We build the MPO representing Pv = |v⟩⟨v| / ‖v‖² starting from the input MPS.
    #
    #    │   │   │   │   │   │       │   │   │   │   │   │
    #    ▒───▒───▒───▒───▒───▒  ──>  ▓───▓───▓───▓───▓───▓
    #                                │   │   │   │   │   │
    Pv = projector(v)
    n = length(v)
    maxnumber = dim(siteind(v, 1)) - 1

    # 2) We decompose each block of Pv in two pieces so that the physical indices end up in
    #    different blocks. In the middle blocks, the two link indices also get separated.
    #
    #     s₁'     s₁'                 sₖ'         sₖ'                  sₙ'       sₙ'
    #     │       │                   │           │                    │         │
    #     ▓─l₁  = ░              lₖ₋₁─▓─lₖ = lₖ₋₁─░               lₙ₋₁─▓  = lₙ₋₁─░
    #     │       │                   │           │                    │         │
    #     s₁      ░─l₁                sₖ          ░─lₖ                 sₙ        ░
    #             │                               │                              │
    #             s₁                              sₖ                             sₙ
    #
    T = ITensor[]

    s₁′ = siteinds(Pv, 1; plev=1)
    s₁ = siteinds(Pv, 1; plev=0)
    l₁ = linkind(Pv, 1)
    U, Σ, V = svd(Pv[1], (s₁′,); righttags="Link,r=1")
    # Let's call the newly created link indices "r=$k", so that we can distinguish them
    # later from the "original" link indices which are tagged as "l=$k".
    append!(T, [U*Σ, V])
    # We could choose to merge Σ with V as well instead of with U, it doesn't really matter
    # where we put the singular values, as long as we obtain two tensors as output.

    for k in 2:(n - 1)
        sₖ′ = siteinds(Pv, k; plev=1)
        sₖ = siteinds(Pv, k; plev=0)
        lₖ₋₁ = linkind(Pv, k-1)
        lₖ = linkind(Pv, k)
        U, Σ, V = svd(Pv[k], (sₖ′, lₖ₋₁); righttags="Link,r=$k")
        append!(T, [U*Σ, V])
    end

    sₙ′ = siteinds(Pv, n; plev=1)
    sₙ = siteinds(Pv, n; plev=0)
    lₙ₋₁ = linkind(Pv, n-1)
    U, Σ, V = svd(Pv[n], (sₙ′, lₙ₋₁); righttags="Link,r=$n")
    append!(T, [U*Σ, V])

    # 3) The resulting tensor network is still one-dimensional: we link everything together
    #    in a proper MPS.
    #
    #     s₁'   s₂'   s₃'                sₙ₋₁'  sₙ'
    #     │     │     │                   │     │
    #     ░  .·─░  .·─░  .·─           .·─░  .·─░       ────────╮
    #     │  ·  │  ·  │  ·     ···     ·  │  ·  │               │
    #     ░─·'  ░─·'  ░─·'           ─·'  ░─·'  ░               │
    #     │     │     │                   │     │               │
    #     s₁    s₂    s₃                 sₙ₋₁   sₙ              │
    #                                                           │
    #                                                           ▼
    #
    #                               s₁' s₁  s₂' s₂  s₃' s₃          sₙ' sₙ
    #                               │   │   │   │   │   │           │   │
    #                               ░───░───░───░───░───░─── ··· ───░───░
    vv = MPS(T)

    # The MPS we just constructed is not in a canonical form, let's fix that.
    orthogonalize!(vv, 1)

    # 4) We need to fix vv's indices, since now siteind(vv, k) == siteind(vv, k + 1)': they
    #    share the same ID and this will surely cause issues when contracting the MPS with
    #    something else. We create a new set of indices and replace them all.
    newsites = sb_siteinds(; nmodes=n, maxnumber=maxnumber)
    replace_siteinds!(vv, newsites)

    # Now the link index situation is something like
    #
    # [1]    (dim=##|id=###|"Link,r=1")
    # [2]    (dim=##|id=###|"Link,l=1")
    # [3]    (dim=##|id=###|"Link,r=2")
    # [4]    (dim=##|id=###|"Link,l=2")
    # [5]    (dim=##|id=###|"Link,r=3")
    # [6]    (dim=##|id=###|"Link,l=3")
    #        ...
    # [n-3]  (dim=##|id=###|"Link,r=n-1")
    # [n-2]  (dim=##|id=###|"Link,l=n-1")
    # [n-1]  (dim=##|id=###|"Link,r=n")
    #
    # while we want a we want a linear sequence like
    #
    # [1]   (dim=##|id=###|"Link,l=1")
    # [2]   (dim=##|id=###|"Link,l=2")
    # [3]   (dim=##|id=###|"Link,l=3")
    #       ...
    # [n-1] (dim=##|id=###|"Link,l=n")

    for k in eachindex(vv)[1:(end - 1)]
        if isodd(k)
            rk = div(k, 2)+1
            replacetags!(vv[k], "r=$rk", "m=$k")
            replacetags!(vv[k + 1], "r=$rk", "m=$k")
            # Unfortunately we already have "l=#" tags in the link indices so we first
            # change both "r=#" and "l=#" into "m=#", then later we'll change all of them
            # to "l=#".
        else
            lk = div(k, 2)
            replacetags!(vv[k], "l=$lk", "m=$k")
            replacetags!(vv[k + 1], "l=$lk", "m=$k")
        end
    end

    for k in eachindex(vv)[1:(end - 1)]
        replacetags!(vv[k], "m=$k", "l=$k")
        replacetags!(vv[k + 1], "m=$k", "l=$k")
    end

    return convert(SuperBosonMPS, vv)
end

"""
    sample(m::SuperBosonMPS)

Given a "superbosonic" MPS `m` with unit trace, compute a `Vector{Int}` of `length(m)/2`
elements corresponding to one sample of the probability distribution defined by the
components of the density matrix that the MPS represents.
"""
function ITensorMPS.sample(m::SuperBosonMPS)
    return sample(Random.default_rng(), m)
end

"""
    randsample(values, weights)

Draw an element from `values`, such that `values[i]` has weight `weights[i]`.
"""
function randsample(values, weights)
    odds = cumsum(weights ./ sum(weights))
    return values[findfirst(odds .≥ rand())]
end

function sample(rng::AbstractRNG, m::SuperBosonMPS)
    # See https://tensornetwork.org/mps/algorithms/sampling for an explanation of this MPS
    # sampling algorithm.
    nmodes = div(length(m), 2)

    trace_m = tr(m)
    if abs(1.0 - trace_m) > 1E-8
        error("sample: MPS is not normalized, trace=$trace")
    end

    ids = _id_contractions(m)
    contractions_from_right = Vector{Union{ITensor,OneITensor}}(undef, nmodes)
    # We want:
    #   contractions_from_right[end] = 1
    #   contractions_from_right[end-1] = ids[end]
    #   contractions_from_right[end-2] = contractions_from_right[end-1] * ids[end-1]
    #   contractions_from_right[end-3] = contractions_from_right[end-2] * ids[end-2]
    # and so on...
    # The first element contractions_from_right[1] will be the MPS `m` contracted on all
    # pairs of sites but the first one, i.e. ids[2] * ids[3] * ... * ids[end].
    contractions_from_right[end] = OneITensor()
    for j in reverse(1:(length(ids) - 1))
        contractions_from_right[j] = ids[j + 1] * contractions_from_right[j + 1]
    end

    # contractions_from_right[k] == prod(ids[j] * v[j] * v[j+1] for j in 2k+1:2:2nmodes)
    #
    # Basically contractions_from_right[k] is the reduced state after the partial trace
    # over all modes > k.

    result = Vector{Int}(undef, nmodes)
    prev = OneITensor()
    for j in 1:nmodes
        ρ = prev * m[sb_index(j)] * m[sb_index(j) + 1] * contractions_from_right[j]
        sl = siteind(m, sb_index(j))
        sr = siteind(m, sb_index(j)+1)
        # `ρ` is a matrix with indices `sl` and `sr`

        @assert dim(sl) == dim(sr)
        d = dim(sl)
        pn = [scalar(ρ * onehot(sl => k) * onehot(sr => k)) for k in 1:d]
        if !isapprox(real(pn), pn)
            throw(error("non-zero imaginary part in sampling probabilities"))
        end
        pn = real(pn)
        n = randsample(1:d, pn)
        result[j] = n

        if j < nmodes
            prev *=
                onehot(sl => n) * m[sb_index(j)] * m[sb_index(j) + 1] * onehot(sr => n) /
                pn[n]
        end
    end
    return result
end
