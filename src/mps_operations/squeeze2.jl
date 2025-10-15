function _squeeze2_coefficient(z, n′, m′, n, m, k, l)
    # Matrix element ⟨f(n') ⊗ f(m'), S₂(z) f(n) ⊗ f(m)⟩ of the two-mode squeezing operator
    #   S₂(z) = exp(z a* ⊗ a* - z̄ a ⊗ a)
    # on the eigenbasis {f(n)}ₙ of the number operator.
    return if n′-k == n-l && m′-k==m-l
        r, ψ = abs(z), angle(z)
        ν = sinh(r) * cis(ψ)
        μ = cosh(r)
        sqrt(binomial(n′, k) * binomial(m′, k) * binomial(n, l) * binomial(m, l)) *
        (ν/μ)^k *
        (-conj(ν)/μ)^l *
        μ^(-(m+n-2l+1))
    else
        zero(z)
    end
end

function ITensors.op(::OpName"squeezer2", ::SiteType"Boson", s1::Index, s2::Index; squeeze)
    T = ITensor(ComplexF64, s1', s2', s1, s2)
    for n in eachval(s1), m in eachval(s2), n′ in eachval(s1'), m′ in eachval(s2')
        T[s1 => n, s2 => m, s1' => n′, s2' => m′] = sum(
            _squeeze2_coefficient(squeeze, n′-1, m′-1, n-1, m-1, k, l) for
            k in 0:min(n′ - 1, m′ - 1), l in 0:min(n - 1, m - 1)
        )
    end
    return T
end

@doc raw"""
    squeeze2(v::Union{MPS,SuperBosonMPS}, z, n1, n2; kwargs...)

Apply the two-mode squeezing operator

```math
S₂(z) = \exp(z \adj{a} ⊗ \adj{a} - \conj{z} a ⊗ a)
```

with ``z ∈ ℂ``, on modes `n1` and `n2` of the state represented by `v`.
"""
function GaussianStates.squeeze2(v::SuperBosonMPS, z, n1, n2; kwargs...)
    phy1, anc1 = siteind(v, sb_index(n1)), siteind(v, sb_index(n1)+1)
    phy2, anc2 = siteind(v, sb_index(n2)), siteind(v, sb_index(n2)+1)

    sq_phy = op("squeezer2", phy1, phy2; squeeze=z)
    sq_anc = op("squeezer2", anc1, anc2; squeeze=z)

    return apply([sq_phy, conj(sq_anc)], v; kwargs...)
end

function GaussianStates.squeeze2(m::MPS, z, n1, n2; kwargs...)
    return apply(op("squeezer2", siteind(m, n1), siteind(m, n2); squeeze=z), m; kwargs...)
end
