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

function ITensors.op(
    ::OpName"squeezer2", st::SiteType"Boson", s1::Index, s2::Index; squeeze
)
    T = ITensor(ComplexF64, s1', s2', s1, s2)
    for n in eachval(s1), m in eachval(s2), n′ in eachval(s1'), m′ in eachval(s2')
        T[s1 => n, s2 => m, s1' => n′, s2' => m′] = sum(
            _squeeze2_coefficient(squeeze, n′-1, m′-1, n-1, m-1, k, l) for
            k in 0:min(n′ - 1, m′ - 1), l in 0:min(n - 1, m - 1)
        )
    end
    return T
end

"""
    squeeze2(v::SuperBosonMPS, z, n1, n2; kwargs...)

Apply the two-mode squeezing operator ``exp(z a* ⊗ a* - z̄ a ⊗ a)`` on modes `n1` and `n2`
with squeeze parameter `z` to the state represented by `v`.
"""
function GaussianStates.squeeze2(v::SuperBosonMPS, z, n1, n2; kwargs...)
    phy1, anc1 = siteind(v, sb_index(n1)), siteind(v, sb_index(n1)+1)
    phy2, anc2 = siteind(v, sb_index(n2)), siteind(v, sb_index(n2)+1)

    bs_phy = op("squeeze2", phy1, phy2; squeeze=z)
    bs_anc = op("squeeze2", anc1, anc2; squeeze=z)

    v = apply(bs_phy, v; kwargs...)
    # `apply` already moves the sites so that they are adjacent before actually applying
    # the two-site operator, then it moves them back to their original position.
    return apply(conj(bs_anc), v; kwargs...)
end
