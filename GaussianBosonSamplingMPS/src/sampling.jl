"""
    sample_displaced(ψ::MPS, W; nsamples, nsamples_per_displacement, eval_atol=0, kwargs...)

Apply random displacements sampled from the positive semi-definite matrix `W` to the pure
state `ψ` and sample `nsamples` elements from the resulting state.

Return a matrix of `UInt8` elements such that each column is a sample drawn from the final
state, such that a new displacement vector is computed each `nsamples_per_displacement`
draws. (The matrix will actually have a number of columns equal to
`nsamples_per_displacement * floor(nsamples / nsamples_per_displacement)`.)

The `eval_atol` keyword argument is used as threshold to decide whether an eigenvalue of `W`
must be considered zero (usually it should be of the same order of the `eps` tolerances of
the `SCS` optimiser).
"""
function sample_displaced(
    ψ::MPS, W; nsamples, nsamples_per_displacement, eval_atol=0, kwargs...
)
    @assert size(W, 1) == size(W, 2) == 2length(ψ)
    n = length(ψ)

    @assert nsamples ≥ nsamples_per_displacement

    ψnorm = norm(ψ)
    if abs(1.0 - ψnorm) > 1E-8
        # This condition is the same used by ITensorMPS in its `sample` method.
        @warn "sample_displaced: MPS is not normalised, norm=$ψnorm. Continuing with " *
            "a normalised MPS."
        ψ = normalize(ψ)
    end

    # The Distributions package defines the multivariate normal distribution as
    #
    #                       1
    #   f(x; μ, Σ)= ────────────────── exp(−1/2 (x−μ)ᵀ Σ⁻¹ (x−μ))
    #               (2π)^(d/2) √det(Σ)
    #
    # where d is the dimension of the vector space, i.e. twice the number of modes.
    # In Serafini's book the classical mixing channel is
    #
    #                 1      ╭
    #   Φ(ρ; Y) = ────────── │    exp(−xᵀ Y⁻¹ x) Dₓ ρ Dₓ* dx
    #             πⁿ √det(Y) ╯ℝ²ⁿ
    #
    # with n = d/2, and has the effect of adding Y to the covariance matrix of the Gaussian
    # state. We already have Y = W, so we need to use Σ = W/2.
    #
    # However, this works only if W is positive. In our case W may be only positive
    # semi-definite, so we need a modified version of the formula above, in which basically
    # the integration is performed only on the subspace of ℝ²ⁿ orthogonal to Y's kernel.
    # Let m = 2n - dim(ker Y):
    #
    #                   1      ╭
    #   Φ(ρ; Y) = ──────────── │   exp(−∑ⱼ λⱼ⁻¹uⱼ²) Dₓ ρ Dₓ * du =
    #             (√π)ᵐ ∏ⱼ √λⱼ ╯ℝᵐ
    #
    #                1     ╭
    #           = ──────── │   exp(−uᵀ Λ⁻¹ u) Dₓ ρ Dₓ * du
    #             √det(πΛ) ╯ℝᵐ
    #
    # where x = Mᵀu, the λⱼ's are the non-zero eigenvalues of Y (collected in the diagonal
    # matrix Λ) and M is the (orthogonal) matrix that diagonalises the restriction of Y
    # to the subspace of ℝ²ⁿ orthogonal to its kernel.
    # Therefore, we actually have to sample a displacement vector u from a multivariate
    # normal distribution with covariance matrix Λ/2 and then displace the state by Mᵀu,
    # instead of by just u.
    W_evals, M = eigen(Symmetric(W); sortby=-)
    # Wrap `W` in a `Symmetric` constructor so that `eigen` knows that it's symmetric: this
    # way we're sure that M is real and orthogonal, and not just approximately so.
    # The `sortby=-` tells `eigen` to return the decomposition with the eigenvalues in
    # decreasing order.
    Λ = Diagonal(filter(x -> abs(x) > eval_atol, W_evals))

    random_displacement_dist = MvNormal(Λ/2)

    nbatches = floor(Int, nsamples / nsamples_per_displacement)

    # Randomly generate displacement vectors from the noise matrix:
    # 1. sample from the normal distribution defined by Λ/2 above (this gives us a
    #    m × nbatches real matrix, where each column is a sample.)
    A = rand(random_displacement_dist, nbatches)
    # 2. pad with zeroes in order to obtain vectors in ℝ²ⁿ (the zeroes correspond to the
    #    result of sampling along the direction where W has a null eigenvalue)
    A = vcat(A, zeros(2n-size(A, 1), size(A, 2)))
    # 3. multiply by Mᵀ to go back to the original basis (note that multiplying the whole
    #    matrix by Mᵀ is the same thing as separately multiplying each column by Mᵀ).
    αs_xpxp = M' * A

    # Since W is (supposed to be) in the xpxp format, so is each αs[:, j], and we need to
    # transform it into a complex vector so that the `displace_pure` function can read it.
    αs = [
        [complex(αs_xpxp[i, j], αs_xpxp[i + 1, j]) for i in 1:2:size(αs_xpxp, 1)] for
        j in axes(αs_xpxp, 2)
    ]

    batchsize = nsamples_per_displacement
    samples = Matrix{UInt8}(undef, n, nbatches * batchsize)
    pbar = Progress(nbatches; desc="Sampling...")
    Threads.@threads for b in 1:nbatches
        ψ_displaced = displace_pure(ψ, αs[b])  # displace the state
        orthogonalize!(ψ_displaced, 1)
        for j in 1:batchsize
            # Sample from the MPS.
            samples[:, (b - 1) * batchsize + j] .= sample(ψ_displaced) .- 1
        end
        next!(pbar)
    end

    return samples
end
