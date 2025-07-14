# First and second moments of states from their MPS

function ITensors.op(::OpName"x", st::SiteType"Boson", d::Int)
    return (op(OpName("a†"), st, d) + op(OpName("a"), st, d)) / sqrt(2)
end

function ITensors.op(::OpName"p", st::SiteType"Boson", d::Int)
    return (op(OpName("a†"), st, d) - op(OpName("a"), st, d)) * im / sqrt(2)
end

function firstmoments(v; warn_atol=1e-14)
    # `expect(v, "x", "p")` returns the expectation values as the following tuple:
    #   ([⟨x[1]⟩, ⟨x[2]⟩, ⟨x[N]⟩], [⟨p[1]⟩, ⟨p[2]⟩, ⟨p[N]⟩])
    r = collect(Iterators.flatten(zip(expect(v, "x", "p")...)))

    if !isapprox(real(r), r) && norm(r) > warn_atol
        # It's not uncommon that the first moments are zero; in this case calling `isapprox`
        # with a zero as argument is inevitably `false`, and the warning is triggered even
        # if it's not necessary; the `norm(r) > warn_atol` cutoff prevents this.
        @warn "first moments are not real"
    end
    return real(r)
end

function covariancematrix(v; warn_atol=1e-14)
    r = firstmoments(v; warn_atol=warn_atol)
    XX = correlation_matrix(v, "x", "x")
    PP = correlation_matrix(v, "p", "p")
    XP = correlation_matrix(v, "x", "p")
    PX = correlation_matrix(v, "p", "x")
    c = [
        XX XP
        PX PP
    ]
    σ = GaussianStates.permute_to_xpxp(c + transpose(c)) - 2kron(r, r')
    if !isapprox(real(σ), σ)
        # σ is never zero so we don't have to worry about using `isapprox` on zero
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

@doc raw"""
    attenuate(v::MPS, attenuation, n)

Apply on mode `n` the attenuator channel ``ρ ↦ \sum_{k=0}^{+∞} B_k ρ \adj{B_k}`` on the MPS
`v` representing the state ``ρ`` in the superboson formalism.

```math
B_k = \sum_{m=0}^{+∞} \binom{m+k}{k}^{\frac12} (1-η^2)^{\frac{k}{2}} η^m |m⟩⟨m+k|
```

and ``η`` is the attenuation coefficient, such that ``\sum_{k=0}^{+\infty} B_k ρ \adj{B_k}``
is equal to ``|0⟩⟨0|`` when ``η = 0`` and to ``ρ`` when ``η = 1``.
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

function GaussianStates.squeeze(v::MPS, z; kwargs...)
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

# Displacement

function ITensors.op(::OpName"displace", st::SiteType"Boson", d::Int; α)
    return exp(α * op(OpName("a†"), st, d) - conj(α) * op(OpName("a"), st, d))
end

"""
    displace_pure(m::MPS, α; kwargs...)

Apply a product of single-mode displacement operators on the pure state represented by the
MPS `m`, with parameter `α[j]` on mode `j`.
"""
function displace_pure(m::MPS, α; kwargs...)
    @assert length(m) == length(α)
    displacement_ops = [op("displace", siteind(m, j); α=α[j]) for j in eachindex(m)]
    return apply(displacement_ops, m; kwargs...)
end

"""
    displace(v::MPS, α; kwargs...)

Apply a product of single-mode displacement operators, with parameter `α[j]` on mode `j`,
on the mixed state represented by the MPS `m` in the superboson formalism.
"""
function GaussianStates.displace(v::MPS, α; kwargs...)
    @assert iseven(length(v))
    nmodes = div(length(v), 2)
    @assert nmodes == length(α)

    displacement_ops = ITensor[]
    for j in 1:nmodes
        append!(
            displacement_ops,
            [
                op("displace", siteind(v, sb_index(j)); α=α[j]),
                conj(op("displace", siteind(v, sb_index(j)+1); α=α[j])),
            ],
        )
    end

    return apply(displacement_ops, v; kwargs...)
end
