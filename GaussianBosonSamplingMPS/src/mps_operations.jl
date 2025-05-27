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
