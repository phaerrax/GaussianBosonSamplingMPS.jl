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

"""
    displace(v::Union{MPS, SuperBosonMPS}, α; kwargs...)

Apply a product of single-mode displacement operators on the pure or mixed state `v` , with
parameter `α[j]` on mode `j`.
"""
function GaussianStates.displace(m::MPS, α; kwargs...)
    @assert length(m) == length(α)
    displacement_ops = [op("displace", siteind(m, j); α=α[j]) for j in eachindex(m)]
    return apply(displacement_ops, m; kwargs...)
end

function GaussianStates.displace(v::SuperBosonMPS, α; kwargs...)
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
