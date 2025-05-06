function ITensors.op(::OpName"displ", st::SiteType"Boson", d::Int; p)
    return exp(p * op(OpName("Adag"), st, d) - conj(p) * op(OpName("A"), st, d))
end

"""
    displace(m::MPS, α::AbstractVector)

Apply a product of single-mode displacement operators on the state represented by the MPS
`m`, with parameter `α[j]` on mode `j`.
"""
function GaussianStates.displace!(m::MPS, α::AbstractVector)
    @assert length(m) == length(α)
    for i in eachindex(m)
        orthogonalize!(m, i)
        D = op("displ", siteind(m, i); p=α[i])
        newblock = D * m[i]
        m[i] = noprime(newblock)
    end
end
