# Beam splitters

function _beamsplitter_coefficient(z, n′, m′, n, m, k, l)
    # Matrix element ⟨f(n') ⊗ f(m'), B(z) f(n) ⊗ f(m)⟩ of the beam-splitter operator
    #   B(z) = exp(z a* ⊗ a - z̄ a ⊗ a*)
    # on the eigenbasis {f(n)}ₙ of the number operator.
    r, θ = abs(z), angle(z)
    return if (k != n′-m+l) || (n′+m′ != n+m)
        zero(z)
    else
        binomial(n, k) *
        binomial(m, l) *
        sqrt(factorial(k+m-l) * factorial(n-k+l) / (factorial(n) * factorial(m))) *
        cos(r)^(k+l) *
        sin(r)^(n-k+m-l) *
        (-cis(-θ * (n-k))) *
        cis(θ * (m-l))
    end
end

function ITensors.op(::OpName"beamsplitter", ::SiteType"Boson", s1::Index, s2::Index; angle)
    B = ITensor(ComplexF64, s1', s2', s1, s2)
    for n′ in eachval(s1'), m′ in eachval(s2'), n in eachval(s1), m in eachval(s2)
        B[s1' => n′, s2' => m′, s1 => n, s2 => m] = sum(
            _beamsplitter_coefficient(angle, n′-1, m′-1, n-1, m-1, k, l) for
            k in 0:(n - 1), l in 0:(m - 1)
        )
    end
    return B
end

@doc raw"""
    beamsplitter(v::Union{MPS,SuperBosonMPS}, z, n1, n2; kwargs...)

Apply the beam-splitter operator

```math
B(z) = \exp(z \adj{a} ⊗ a - \conj{z} a ⊗ \adj{a})
```

with ``z ∈ ℂ``, to modes `n1` and `n2` of the state represented by `v`.
"""
function GaussianStates.beamsplitter(v::SuperBosonMPS, z, n1, n2; kwargs...)
    phy1, anc1 = siteind(v, sb_index(n1)), siteind(v, sb_index(n1)+1)
    phy2, anc2 = siteind(v, sb_index(n2)), siteind(v, sb_index(n2)+1)

    bs_phy = op("beamsplitter", phy1, phy2; angle=z)
    bs_anc = op("beamsplitter", anc1, anc2; angle=z)

    return apply([bs_phy, conj(bs_anc)], v; kwargs...)
end

function GaussianStates.beamsplitter(m::MPS, z, n1, n2; kwargs...)
    return apply(op("beamsplitter", siteind(m, n1), siteind(m, n2); angle=z), m; kwargs...)
end
