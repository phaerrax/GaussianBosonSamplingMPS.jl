# Squeezers

function _squeeze_coefficient(z, n, m)
    # Matrix element ⟨f(n), S(z) f(m)⟩ of the squeezing operator
    #   S(z) = exp(z/2 a*² - z̄/2 a²)
    # on the eigenbasis {f(n)}ₙ of the number operator (Marian, 1992).
    r, θ = abs(z), angle(z)
    return if iseven(n) && iseven(m)
        (-1)^div(m, 2) / (factorial(div(m, 2)) * factorial(div(n, 2))) *
        sqrt(factorial(n)*factorial(m)/cosh(r)) *
        cis((n-m)*θ/2) *
        (tanh(r)/2)^div(n+m, 2) *
        pFq((-div(m, 2), -div(n, 2)), (1/2,), -sinh(r)^(-2))
    elseif isodd(n) && isodd(m)
        (-1)^div(m-1, 2) / (factorial(div(m-1, 2)) * factorial(div(n-1, 2))) *
        sqrt(factorial(n)*factorial(m)/(cosh(r)^3)) *
        cis((n-m)*θ/2) *
        (tanh(r)/2)^(div(n+m, 2)-1) *
        pFq((-div(m-1, 2), -div(n-1, 2)), (3/2,), -sinh(r)^(-2))
    else
        zero(z)
    end
end

function ITensors.op(::OpName"squeezer", ::SiteType"Boson", d::Int; squeeze)
    S = zeros(ComplexF64, d, d)
    for n′ in 0:(d - 1), n in 0:(d - 1)
        S[n′ + 1, n + 1] = _squeeze_coefficient(squeeze, n′, n)
    end
    return S
end

@doc raw"""
    squeeze(v::Union{MPS,SuperBosonMPS}, z, n; kwargs...)

Apply the squeezing operator

```math
S(z) = \exp\bigl(\tfrac12 z (\adj{a})^2 - \tfrac12 \conj{z} a^2\bigr)
```

with ``z ∈ ℂ``, to the `n`-th mode of the state represented by `v`.
"""
function GaussianStates.squeeze(v::SuperBosonMPS, z, n; kwargs...)
    phy, anc = siteind(v, sb_index(n)), siteind(v, sb_index(n)+1)

    sq_phy = op("squeezer", phy; squeeze=z)
    sq_anc = op("squeezer", anc; squeeze=z)
    return apply([sq_phy, conj(sq_anc)], v; kwargs...)
end

function GaussianStates.squeeze(m::MPS, z, n; kwargs...)
    return apply(op("squeezer", siteind(m, n); squeeze=z), m; kwargs...)
end

@doc raw"""
    squeeze(v::Union{MPS,SuperBosonMPS}, z; kwargs...)

Apply the squeezing operator

```math
⨂_{i=1}^{n} S(z_i),
\quad
S(z) = \exp\bigl(\tfrac12 z (\adj{a})^2 - \tfrac12 \conj{z} a^2\bigr),
```

with ``z_i ∈ ℂ``, to all modes of the state represented by `v`.
"""
function GaussianStates.squeeze(v::SuperBosonMPS, z; kwargs...)
    @assert nmodes(v) == length(z)

    squeezing_ops = ITensor[]
    for j in 1:nmodes(v)
        append!(
            squeezing_ops,
            [
                op("squeezer", siteind(v, sb_index(j)); squeeze=z[j]),
                conj(op("squeezer", siteind(v, sb_index(j)+1); squeeze=z[j])),
            ],
        )
    end

    return apply(squeezing_ops, v; kwargs...)
end

function GaussianStates.squeeze(m::MPS, z; kwargs...)
    @assert length(m) == length(z)
    squeezing_ops = [op("squeezer", siteind(m, j); squeeze=z[j]) for j in eachindex(m)]
    return apply(squeezing_ops, m; kwargs...)
end
