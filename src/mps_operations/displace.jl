# Displacement

function _displace_coefficient(z, n′, n, k, l)
    # Matrix element ⟨f(n), D(z) f(m)⟩ of the displacement operator
    #   D(z) = exp(z a* - z̄ a)
    # on the eigenbasis {f(n)}ₙ of the number operator.
    return if n′ - k == n - l
        exp(-0.5abs2(z)) * z^k * (-conj(z))^l / sqrt(factorial(k) * factorial(l)) *
        sqrt(binomial(n′, k) * binomial(n, l))
    else
        zero(z)
    end
end

function ITensors.op(::OpName"displace", ::SiteType"Boson", s::Index; α)
    S = ITensor(ComplexF64, s', s)
    for n′ in eachval(s'), n in eachval(s)
        S[s => n, s' => n′] = sum(
            _displace_coefficient(α, n′-1, n-1, k, l) for k in 0:(n′ - 1), l in 0:(n - 1)
        )
    end

    return S
end

@doc raw"""
    displace(v::Union{MPS,SuperBosonMPS}, α, n; kwargs...)

Apply the displacement operator

```math
D(α) = \exp(α \adj{a} - \conj{α} a)
```

with ``α ∈ ℂ``, to the `n`-th mode of the state represented by `v`.
"""
function GaussianStates.displace(v::SuperBosonMPS, α, n; kwargs...)
    phy, anc = siteind(v, sb_index(n)), siteind(v, sb_index(n)+1)

    sq_phy = op("displace", phy; α=α)
    sq_anc = op("displace", anc; α=α)
    return apply([sq_phy, conj(sq_anc)], v; kwargs...)
end

function GaussianStates.displace(m::MPS, α, n; kwargs...)
    return apply(op("displace", siteind(m, n); α=α), m; kwargs...)
end

@doc raw"""
    displace(v::Union{MPS,SuperBosonMPS}, α; kwargs...)

Apply the displacement operator

```math
⨂_{i=1}^{n} D(α_i),
\quad
D(α) = \exp(α \adj{a} - \conj{α} a)
```

with ``α_i ∈ ℂ``, to all modes of the state represented by `v`.
"""
function GaussianStates.displace(m::MPS, α; kwargs...)
    @assert length(m) == length(α)
    displacement_ops = [op("displace", siteind(m, j); α=α[j]) for j in eachindex(m)]
    return apply(displacement_ops, m; kwargs...)
end

function GaussianStates.displace(v::SuperBosonMPS, α; kwargs...)
    @assert nmodes(v) == length(α)

    displacement_ops = ITensor[]
    for j in 1:nmodes(v)
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
