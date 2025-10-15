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

function ITensors.op(::OpName"squeezer", st::SiteType"Boson", d::Int; squeeze)
    S = zeros(ComplexF64, d, d)
    for n′ in 0:(d - 1), n in 0:(d - 1)
        S[n′ + 1, n + 1] = _squeeze_coefficient(squeeze, n′, n)
    end
    return S
end

"""
    squeeze(v::SuperBosonMPS, n, z; kwargs...)

Apply the squeezing operator with parameter `z` on mode `n` to the state represented by `v`.
"""
function GaussianStates.squeeze(v::SuperBosonMPS, n, z; kwargs...)
    phy, anc = siteind(v, sb_index(n)), siteind(v, sb_index(n)+1)

    sq_phy = op("squeezer", phy; squeeze=z)
    sq_anc = op("squeezer", anc; squeeze=z)
    v = apply(sq_phy, v; kwargs...)
    return apply(conj(sq_anc), v; kwargs...)
end

"""
    squeeze(v::SuperBosonMPS, z; kwargs...)

Apply the squeezing operator with parameter `z_i` on each mode `i` to the state represented
by `v`.
"""
function GaussianStates.squeeze(v::SuperBosonMPS, z; kwargs...)
    @assert length(v) == 2length(z)
    for j in eachindex(z)
        sq_phy = op("squeezer", siteind(v, sb_index(j)); squeeze=z[j])
        sq_anc = op("squeezer", siteind(v, sb_index(j)+1); squeeze=z[j])
        v = apply(sq_phy, v; kwargs...)
        v = apply(conj(sq_anc), v; kwargs...)
    end

    return v
end
