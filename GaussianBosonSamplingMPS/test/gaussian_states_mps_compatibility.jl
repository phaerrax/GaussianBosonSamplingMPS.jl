@testset "First and second moments from MPS" begin
    nmodes = 6
    maxnumber = 4

    # With mixed states
    sites = sb_siteinds(; nmodes=nmodes, maxnumber=maxnumber)
    v = MPS(ComplexF64, sites, "0")
    @test iszero(firstmoments(v))
    @test covariancematrix(v) ≈ I

    # With pure states
    sites = siteinds("Boson", nmodes; dim=maxnumber+1)
    v = MPS(ComplexF64, sites, "0")
    @test iszero(firstmoments(v))
    @test covariancematrix(v) ≈ I
end

const atol = 1e-12
const rtol = 1e-6
@testset "From GaussianState to MPS and back (atol=$atol, rtol=$rtol)" begin
    nmodes = 3
    maxnumber = 12

    cutoff = 1e-12
    r = atanh(cutoff^(1/maxnumber)) * rand()
    g = vacuumstate(nmodes)
    squeeze2!(g, r*cispi(2rand()), 1, 2)
    beamsplitter!(g, rand(), 2, 3)

    sites = sb_siteinds(; nmodes=nmodes, maxnumber=maxnumber)
    v = MPS(g; maxdim=15, maxnumber=maxnumber)

    orthogonalize!(v, nmodes)
    # Re-orthogonalize the MPS so that excess bond dimensions are cut off.

    vv = sb_outer(v)
    #@test firstmoments(vv) ≈ 0 won't work, we can't use `isapprox` with zero
    @test norm(firstmoments(vv)) < atol
    @test covariancematrix(vv) ≈ g.covariance_matrix rtol=rtol
end
