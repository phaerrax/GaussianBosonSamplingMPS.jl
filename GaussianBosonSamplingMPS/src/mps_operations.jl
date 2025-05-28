# First and second moments of states from their MPS

function ITensors.op(::OpName"x", st::SiteType"Boson", d::Int)
    return (op(OpName("a†"), st, d) + op(OpName("a"), st, d)) / sqrt(2)
end

function ITensors.op(::OpName"p", st::SiteType"Boson", d::Int)
    return (op(OpName("a†"), st, d) - op(OpName("a"), st, d)) * im / sqrt(2)
end

function ITensors.op(::OpName"x²", st::SiteType"Boson", d::Int)
    return op(OpName("x"), st, d)^2
end

function ITensors.op(::OpName"p²", st::SiteType"Boson", d::Int)
    return op(OpName("p"), st, d)^2
end

function ITensors.op(::OpName"xp", st::SiteType"Boson", d::Int)
    return op(OpName("x"), st, d) * op(OpName("p"), st, d)
end

function ITensors.op(::OpName"px", st::SiteType"Boson", d::Int)
    return op(OpName("p"), st, d) * op(OpName("x"), st, d)
end

function firstmoments(v)
    @assert iseven(length(v))
    nmodes = div(length(v), 2)

    X = [LocalOperator(sb_index(j) => "x") for j in 1:nmodes]
    P = [LocalOperator(sb_index(j) => "p") for j in 1:nmodes]
    Ri = collect(Iterators.flatten(zip(X, P)))  # [X[1], P[1], X[2], P[2], ...]
    r = measure(v, Ri)

    if !isapprox(real(r), r)
        @warn "first moments are not real"
    end
    return real(r)
end

function covariancematrix(v)
    @assert iseven(length(v))
    nmodes = div(length(v), 2)

    r = firstmoments(v)

    Rij = Matrix{LocalOperator}(undef, 2nmodes, 2nmodes)
    for i in 1:nmodes
        Rij[2i - 1, 2i - 1] = LocalOperator(sb_index(i) => "x²")
        Rij[2i - 1, 2i] = LocalOperator(sb_index(i) => "xp")
        Rij[2i, 2i - 1] = LocalOperator(sb_index(i) => "px")
        Rij[2i, 2i] = LocalOperator(sb_index(i) => "p²")
    end

    for i in 1:nmodes, j in 1:nmodes
        if i != j
            Rij[2i - 1, 2j - 1] = LocalOperator((sb_index(i) => "x", sb_index(j) => "x"))
            Rij[2i - 1, 2j] = LocalOperator((sb_index(i) => "x", sb_index(j) => "p"))
            Rij[2i, 2j - 1] = LocalOperator((sb_index(i) => "p", sb_index(j) => "x"))
            Rij[2i, 2j] = LocalOperator((sb_index(i) => "p", sb_index(j) => "p"))
        end
    end

    pre_σ = measure(v, Rij)  # this is tr(ρ RᵢRⱼ)
    σ = pre_σ .+ transpose(pre_σ) .- 2r * transpose(r)

    if !isapprox(real(σ), σ)
        @warn "covariance matrix is not real"
    end
    return real(σ)
end

# Attenuator channel

function _attenuatormatrixcoefficient(attenuation, k, n, m)
    # The matrix element ⟨fₙ, Bₖ fₘ⟩ where fₙ is the number eigenbasis element with
    # number `n`.
    return sqrt(binomial(n + k, k)) *
           sqrt(1 - attenuation^2)^k *
           attenuation^n *
           ((n + k == m) ? one(attenuation) : zero(attenuation))
end

function ITensors.op(::OpName"attenuator", ::SiteType"Boson", d1::Int, d2::Int; attenuation)
    @assert d1 == d2
    maxn = d1-1
    A = [
        _attenuatormatrixcoefficient(attenuation, k, n, m) for
        k in 0:maxn, n in 0:maxn, m in 0:maxn
    ]

    return sum(kron(A[k, :, :], conj(A[k, :, :])) for k in axes(A, 1))
end

"""
    attenuate(v::MPS, attenuation, n)

Apply on mode `n` the attenuator channel ``ρ ↦ ∑ₖ Bₖ ρ Bₖ*`` on the MPS `v` representing
the state `ρ` in the superboson formalism.

```math
     +∞  ⎛ n+k ⎞½
  Bₖ =  Σ  ⎜     ⎟  (√1-η²)ᵏ ηᵐ |m⟩⟨m+k|
       ₘ₌₀ ⎝  k  ⎠
```

and ``η`` is the attenuation coefficient, such that ``Σₖ Bₖ ⋅ Bₖ*`` is equal to ``|0⟩⟨0|``
when ``η = 0`` and to the identity when ``η = 1``.
"""
function attenuate(v::MPS, attenuation, n; kwargs...)
    sp, sa = siteind(v, sb_index(n)), siteind(v, sb_index(n)+1)
    return apply(op("attenuator", sp, sa; attenuation=attenuation), v; kwargs...)
end

# Squeezers

function ITensors.op(::OpName"squeezer", st::SiteType"Boson", d::Int; squeeze)
    s = squeeze * op(OpName("a†"), st, d)^2 - conj(squeeze) * op(OpName("a"), st, d)^2
    return exp(s/2)
end

function GaussianStates.squeeze(v::MPS, n, z; kwargs...)
    phy, anc = siteind(v, sb_index(n)), siteind(v, sb_index(n)+1)

    sq_phy = op("squeezer", phy; squeeze=z)
    sq_anc = op("squeezer", anc; squeeze=z)
    v = apply(sq_phy, v; kwargs...)
    return apply(conj(sq_anc), v; kwargs...)
end

function GaussianStates.squeeze(v::MPS, z::AbstractVector; kwargs...)
    @assert length(v) == 2length(z)
    for j in eachindex(z)
        sq_phy = op("squeezer", siteind(v, sb_index(j)); squeeze=z[j])
        sq_anc = op("squeezer", siteind(v, sb_index(j)+1); squeeze=z[j])
        v = apply(sq_phy, v; kwargs...)
        v = apply(conj(sq_anc), v; kwargs...)
    end

    return v
end

function ITensors.op(
    ::OpName"beamsplitter", st::SiteType"Boson", d1::Int, d2::Int; transmittivity
)
    θ = acos(transmittivity)
    b = op(OpName("ab†"), st, d1, d2) - op(OpName("a†b"), st, d1, d2)
    return exp(θ * b)
end

function GaussianStates.beamsplitter(v::MPS, transmittivity, n1, n2; kwargs...)
    phy1, anc1 = siteind(v, sb_index(n1)), siteind(v, sb_index(n1)+1)
    phy2, anc2 = siteind(v, sb_index(n2)), siteind(v, sb_index(n2)+1)

    bs_phy = op("beamsplitter", phy1, phy2; transmittivity=transmittivity)
    bs_anc = op("beamsplitter", anc1, anc2; transmittivity=transmittivity)

    v = apply(bs_phy, v; kwargs...)
    # `apply` already moves the sites so that they are adjacent before actually applying
    # the two-site operator, then it moves them back to their original position.
    return apply(conj(bs_anc), v; kwargs...)
end
