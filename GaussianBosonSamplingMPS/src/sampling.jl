"""
    sample_displaced(ψ::MPS, W; nsamples, nsamples_per_displacement, kwargs...)

Apply random displacements sampled from the positive matrix `W` to the pure state `ψ` and
sample from the resulting state.

The output consists of a total of `nsamples` samples, such that a new random displacement
vector is computed each `nsamples_per_displacement` draws.
"""
function sample_displaced(ψ::MPS, W; nsamples, nsamples_per_displacement, kwargs...)
    @assert size(W, 1) == size(W, 2) == 2length(ψ)
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
    random_displacement_dist = MvNormal(W/2)

    samples = Vector{Vector{Int}}(undef, nsamples)
    nbatches = floor(Int, nsamples / nsamples_per_displacement)
    i = 1

    # Randomly generate displacement vectors from the noise matrix.
    αs_xpxp = rand(random_displacement_dist, nbatches)
    # This gives a 2nmodes × nbatches real matrix, where each column is a sample.
    # Since W is (supposed to be) in the xpxp format, so is each αs[:, j], and we need to
    # transform it into a complex vector so that the `displace_pure` function can read it.
    αs = [
        [complex(αs_xpxp[i, j], αs_xpxp[i + 1, j]) for i in 1:2:size(αs_xpxp, 1)] for
        j in axes(αs_xpxp, 2)
    ]

    @showprogress desc="Sampling..." for b in 1:nbatches
        ψ_displaced = displace_pure(ψ, αs[b])  # displace the state
        orthogonalize!(ψ_displaced, 1)
        for _ in 1:nsamples_per_displacement
            # Sample from the MPS.
            samples[i] = sample(ψ_displaced)
            i += 1
        end
    end

    return samples
end
