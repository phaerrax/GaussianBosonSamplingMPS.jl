"""
    optimise(g::GaussianState; verbose=false, scs_eps=nothing)

Return `gₚ, W` where `gₚ` is a new Gaussian state and `W` is a positive semi-definite
matrix such that `W + covariancematrix(gₚ) == covariancematrix(g)` and `gₚ` contains a
smaller number of photons.

Set `verbose = true` to print the information provided by SCS about the optimisation.
"""
function optimise(g::GaussianState; verbose=false, scs_eps=nothing)
    @debug string("Average photon number in non-optimised state: ", number(g))

    n = nmodes(g)
    # Configure optimisation model
    model = Model(SCS.Optimizer)
    if !verbose
        set_silent(model)  # do not print anything
    end

    @variable(model, x[1:(2n), 1:(2n)] in PSDCone())
    @objective(model, Min, tr(x))  # minimise photon number
    @constraint(model, covariancematrix(g) ≥ x, PSDCone())
    @constraint(
        model,
        kron(I(2), x) + kron([[0, 1] [-1, 0]], GaussianStates._symplectic_matrix(n)) ≥ 0,
        PSDCone()
    )  # uncertainty relations

    if !isnothing(scs_eps)
        set_optimizer_attribute(model, "eps_abs", scs_eps)
        set_optimizer_attribute(model, "eps_rel", scs_eps)
    end

    JuMP.optimize!(model)

    sol = JuMP.value(x)

    # Put the solution into a new Gaussian state and show the new photon number
    opt_g = GaussianState(firstmoments(g), sol)
    @debug string("Average photon number in optimised state: ", number(opt_g))

    @debug begin
        ev, _ = eigen(Symmetric(covariancematrix(opt_g)))
        string("Eigenvalues of the optimised covariance matrix σₒₚₜ:\n", join(ev, "\n"))
    end

    @debug begin
        ev, _ = eigen(covariancematrix(opt_g) + im * GaussianStates._symplectic_matrix(n))
        string("Eigenvalues of σₒₚₜ + iΩ:\n", join(ev, "\n"))
    end

    return opt_g, covariancematrix(g) - sol
    # Vₚ, W from the paper
end
