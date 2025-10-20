# First and second moments of states from their MPS

function ITensors.op(::OpName"x", st::SiteType"Boson", d::Int)
    return (op(OpName("a†"), st, d) + op(OpName("a"), st, d)) / sqrt(2)
end

function ITensors.op(::OpName"p", st::SiteType"Boson", d::Int)
    return (op(OpName("a†"), st, d) - op(OpName("a"), st, d)) * im / sqrt(2)
end

"""
    firstmoments(v::Union{MPS,SuperBosonMPS}; warn_atol=1e-14)

Compute the first moments of the state represented by `v` (either a pure state, if `v` is an
`MPS`, or a mixed one, if `v` is a `SuperBosonMPS`), displayed in the `xpxp` order, i.e. the
vector ``(⟨x_1⟩, ⟨p_1⟩, ⟨x_2⟩, ⟨p_2⟩, ..., ⟨x_N⟩, ⟨p_N⟩)``.

The `warn_atol` keyword argument can be used to adjust the threshold used by the function
to warn when the computed moments are not real. The imaginary part of the result is always
truncated.
"""
function GaussianStates.firstmoments(v::Union{MPS,SuperBosonMPS}; warn_atol=1e-14)
    # `expect(v, "x", "p")` returns the expectation values as the following tuple:
    #   ([⟨x[1]⟩, ⟨x[2]⟩, ⟨x[N]⟩], [⟨p[1]⟩, ⟨p[2]⟩, ⟨p[N]⟩])
    r = collect(Iterators.flatten(zip(expect(v, "x", "p")...)))

    if !isapprox(real(r), r) && norm(r) > warn_atol
        # It's not uncommon that the first moments are zero; in this case calling `isapprox`
        # with a zero as argument is inevitably `false`, and the warning is triggered even
        # if it's not necessary; the `norm(r) > warn_atol` cutoff prevents this.
        @warn "first moments are not real"
    end
    return real(r)
end

"""
    covariancematrix(v::Union{MPS,SuperBosonMPS}; warn_atol=1e-14)

Compute the covariance matrix of the state represented by `v` (either a pure state, if `v`
is an `MPS`, or a mixed one, if `v` is a `SuperBosonMPS`), displayed in the `xpxp` order.

The `warn_atol` keyword argument can be used to adjust the threshold used by the function
to warn when the computed moments are not real. The imaginary part of the result is always
truncated.
"""
function GaussianStates.covariancematrix(v::Union{MPS,SuperBosonMPS}; warn_atol=1e-14)
    r = firstmoments(v; warn_atol=warn_atol)
    XX = correlation_matrix(v, "x", "x")
    PP = correlation_matrix(v, "p", "p")
    XP = correlation_matrix(v, "x", "p")
    PX = correlation_matrix(v, "p", "x")
    c = [
        XX XP
        PX PP
    ]
    σ = GaussianStates.permute_to_xpxp(c + transpose(c)) - 2kron(r, r')

    if !isapprox(real(σ), σ)
        # σ is never zero so we don't have to worry about using `isapprox` on zero
        @warn "covariance matrix is not real"
    end
    return real(σ)
end
