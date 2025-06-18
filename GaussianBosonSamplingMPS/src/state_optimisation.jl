"""
    optimise(g::GaussianState; verbose=false)

Return `gₚ`, `W` where `gₚ` is a new Gaussian state and `W` is a positive semi-definite
matrix such that `W + gₚ.covariance_matrix == g.covariance_matrix` and `gₚ` contains a
smaller number of photons.

Set `verbose = true` to print the information provided by SCS about the optimisation.
"""
function optimise(g::GaussianState; verbose=false)
    @debug "Average photon number in non-optimised state: ", number(g)

    n = nmodes(g)
    # Configure optimisation model
    model = Model(SCS.Optimizer)
    if !verbose
        set_silent(model)  # do not print anything
    end

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

    @debug begin
        ev, _ = eigen(opt_g.covariance_matrix)
        "Eigenvalues of the optimised covariance matrix σₒₚₜ:\n", join(ev, "\n")
    end

    @debug begin
        ev, _ = eigen(opt_g.covariance_matrix + im * GaussianStates.Ω(n))
        "Eigenvalues of σₒₚₜ + iΩ:\n", join(ev, "\n")
    end

    return opt_g, g.covariance_matrix - sol
    # Vₚ, W from the paper
end
