# Attenuator channel

function _attenuator_coefficient(attenuation, k, n, m)
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
        _attenuator_coefficient(attenuation, k, n, m) for
        k in 0:maxn, n in 0:maxn, m in 0:maxn
    ]

    return sum(kron(A[k, :, :], conj(A[k, :, :])) for k in axes(A, 1))
end

@doc raw"""
    attenuate(v::SuperBosonMPS, attenuation, n)

Apply the attenuator channel ``ρ ↦ \sum_{k=0}^{+∞} B_k ρ \adj{B_k}`` to the `n`-th mode of
the state ``ρ`` represented by the `SuperBosonMPS` `v`, where

```math
B_k = \sum_{m=0}^{+∞} \binom{m+k}{k}^{\frac12} (1-η^2)^{\frac{k}{2}} η^m |m⟩⟨m+k|
```

and ``η`` is the attenuation coefficient, such that ``\sum_{k=0}^{+\infty} B_k ρ \adj{B_k}``
is equal to ``|0⟩⟨0|`` when ``η = 0`` and to ``ρ`` when ``η = 1``.
"""
function attenuate(v::SuperBosonMPS, attenuation, n; kwargs...)
    sp, sa = siteind(v, sb_index(n)), siteind(v, sb_index(n)+1)
    return apply(op("attenuator", sp, sa; attenuation=attenuation), v; kwargs...)
end
