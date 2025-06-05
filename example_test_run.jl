using GaussianStates, GaussianBosonSamplingMPS, LinearAlgebra, SCS, JuMP, ITensorMPS
using Distributions: MvNormal
using ProgressMeter

"""
    optimise(g::GaussianState)

Return `gₚ`, `W` where `gₚ` is a new Gaussian state and `W` is a positive semi-definite
matrix such that `W + gₚ.covariance_matrix == g.covariance_matrix` and `gₚ` contains a
smaller number of photons.
"""
function optimise(g::GaussianState)
    @debug "Average photon number in non-optimised state: ", number(g)

    n = nmodes(g)
    # Configure optimisation model
    model = Model(SCS.Optimizer)
    @variable(model, x[1:(2n), 1:(2n)] in PSDCone())
    @objective(model, Min, tr(x))  # minimise photon number
    @constraint(model, g.covariance_matrix ≥ x, PSDCone())
    @constraint(
        model, kron(I(2), x) + kron([[0, 1] [-1, 0]], GaussianStates.Ω(n)) ≥ 0, PSDCone()
    )  # uncertainty relations
    JuMP.optimize!(model)

    sol = JuMP.value(x)

    # Put the solution into a new Gaussian state and show the new photon number
    opt_g = GaussianState(g.first_moments, sol)
    @debug "Average photon number in optimised state: ", number(opt_g)

    ev, _ = eigen(opt_g.covariance_matrix)
    @debug "Eigenvalues of the optimised covariance matrix σₒₚₜ:\n", join(ev, "\n")

    ev, _ = eigen(opt_g.covariance_matrix + im * GaussianStates.Ω(n))
    @debug "Eigenvalues of σₒₚₜ + iΩ:\n", join(ev, "\n")

    return opt_g, g.covariance_matrix - sol
    # Vₚ, W from the paper
end

const _testrun_parameters = Dict(
    :squeeze => [0.05, 0.01+0.02im, 0.032-0.01im, 0.001-0.2im],
    :loss => 0.5,
    :beamsplitters => [
        (1, 2, 0.2),
        (3, 1, 0.5),
        (1, 3, 0.1),
        (2, 4, 0.6),
        (1, 3, 0.2),
        (1, 4, 0.4),
        (2, 3, 0.9),
    ],
)

function finalstate_gaussian(; nmodes, parameters, kwargs...)
    g = vacuumstate(nmodes)
    squeeze!(g, parameters[:squeeze])
    for bs in parameters[:beamsplitters]
        lossybeamsplitter!(g, bs[3], parameters[:loss], bs[1], bs[2])
    end
    return g
end

function finalstate_sb(; nmodes, maxnumber, parameters, kwargs...)
    s = sb_siteinds(; nmodes, maxnumber)
    v0 = MPS(s, "0")

    v = squeeze(v0, parameters[:squeeze]; kwargs...)
    for bs in parameters[:beamsplitters]
        v=beamsplitter(v, bs[3], bs[1], bs[2]; kwargs...)
        v=attenuate(v, parameters[:loss], bs[1]; kwargs...)
        v=attenuate(v, parameters[:loss], bs[2]; kwargs...)
    end

    return v
end

function sb_simulation(; nmodes, maxnumber, parameters, nsamples, kwargs...)
    # Get the final state's MPS.
    v = finalstate_sb(;
        nmodes=nmodes, maxnumber=maxnumber, parameters=parameters, kwargs...
    )

    samples = Vector{Vector{Int}}(undef, nsamples)
    @showprogress desc="Sampling" for n in 1:nsamples
        samples[n] = sb_sample(v)
    end

    return samples
end

function gaussian_simulation(;
    nmodes, maxnumber, parameters, nsamples, nsamples_per_displacement, kwargs...
)
    # Get the final state's covariance matrix.
    g = finalstate_gaussian(; nmodes=nmodes, parameters=parameters, kwargs...)

    # Extract the "pure quantum" part and the noise matrix.
    Vₚ, W = optimise(GaussianState(g.first_moments, Symmetric(g.covariance_matrix)))

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

    # Compute the MPS representing the pure quantum part of the final state.
    m = MPS(Vₚ; maxnumber=maxnumber, kwargs...)

    samples = Vector{Vector{Int}}(undef, nsamples)
    nbatches = floor(Int, nsamples / nsamples_per_displacement)
    i = 1
    αs_xpxp = rand(random_displacement_dist, nbatches)
    # This gives a 2nmodes × nbatches real matrix, where each column is a sample.
    # Since W is (supposed to be) in the xpxp format, so is each αs[:, j], and we need to
    # transform it into a complex vector.
    αs = [
        [complex(αs_xpxp[i, j], αs_xpxp[i + 1, j]) for i in 1:2:size(αs_xpxp, 1)] for
        j in axes(αs_xpxp, 2)
    ]
    @showprogress desc="Sampling" for b in 1:nbatches
        # Randomly generate a displacement vector from the noise matrix.
        # Displace the MPS.
        m_displaced = displace_pure(m, αs[b]; kwargs...)
        orthogonalize!(m_displaced, 1)
        for _ in 1:nsamples_per_displacement
            # Sample from the MPS.
            samples[i] = sample(m_displaced)
            i += 1
        end
    end

    return samples
end
