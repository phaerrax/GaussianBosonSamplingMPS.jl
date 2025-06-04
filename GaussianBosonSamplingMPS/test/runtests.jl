using Test
using GaussianStates, ITensors, ITensorMPS, LinearAlgebra
using GaussianBosonSamplingMPS
using GaussianBosonSamplingMPS: hafnian, loophafnian, dirsum
using MPSTimeEvolution: LocalOperator

@testset "Scaling properties of hafnian and loop hafnian" begin
    # See https://hafnian.readthedocs.io/en/latest/loop_hafnian.html
    N = 4
    a = Symmetric(rand(N, N))
    c = rand()

    @test hafnian(c * a) ≈ c^(N / 2) * hafnian(a)

    a_diag = Diagonal(a)
    a_offd = a - Diagonal(a)
    @test loophafnian(sqrt(c) * a_diag .+ c * a_offd) ≈ c^(N / 2) * loophafnian(a)

    b = Symmetric(rand(2, 2))
    @test loophafnian(dirsum(a, b)) ≈ loophafnian(a) * loophafnian(b)
end

function squeezed_state_coeff(x, m)
    if isodd(m)
        return zero(x)
    else
        n = div(m, 2)
        r, θ = abs(x), angle(x)
        return (cis(θ) * tanh(r))^n * sqrt(factorial(2n)) /
               (2^n * factorial(n) * sqrt(cosh(r)))
    end
end

squeezed2_state_coeff(r, θ, n) = (cis(θ) * tanh(r))^n / cosh(r)
squeezed2_state_coeff(x, n) = squeezed2_state_coeff(abs(x), angle(x), n)

@testset verbose = true "Correct construction of MPS of Gaussian states" begin
    @testset "Factorised state has link dims = 1" begin
        # Build an MPS from a Gaussian state that has no entanglement between modes, and
        # check that all its bond dimensions are 1.
        N = 10
        g = vacuumstate(N)
        # Perform some random 1-mode squeezing and displacement.
        squeeze!(g, rand(ComplexF64, N))
        displace!(g, rand(ComplexF64, N))
        s = MPS(g; maxdim=5, maxnumber=5)
        @test all(linkdims(s) .== 1)
    end

    @testset "Squeezed vacuum state on a single mode" begin
        # Build an MPS from a squeezed vacuum state s(r e^(iθ)) and check that
        # its coefficients in the eigenbasis of the number operator are as expected:
        #
        #                    1     +∞                   √(2n)!
        #   s(r e^(iθ)) = ────────  ∑ (e^(iθ) tanh(r))ⁿ ────── |2n⟩
        #                 √cosh(r) n=0                   2ⁿn!

        z = rand(ComplexF64)
        g = vacuumstate(1)
        squeeze!(g, z, 1)
        maxn = 10

        coefficients_expected = [squeezed_state_coeff(z, n) for n in 0:maxn]

        _, S = williamson(g.covariance_matrix)
        Ul, D, Ur = euler(S)
        coefficients_fc = [
            GaussianBosonSamplingMPS.franckcondon([n], Ul, D, Ur, [0]) for n in 0:maxn
        ]

        @test abs.(coefficients_expected) ≈ abs.(coefficients_fc)
    end

    @testset "Squeezed vacuum state on several modes" begin
        # Build an MPS from a product of squeezed vacuum states s(r e^(iθ)) and check that
        # its coefficients in the eigenbasis of the number operator are as expected:
        #
        #                    1     +∞                   √(2n)!
        #   s(r e^(iθ)) = ────────  ∑ (e^(iθ) tanh(r))ⁿ ────── |2n⟩
        #                 √cosh(r) n=0                   2ⁿn!

        N = 2
        maxn = 8
        cutoff = 1e-12
        r = atanh(cutoff^(1/maxn)) .* rand(N)
        θ = 2pi .* rand(N)
        z = r .* cis.(θ)

        g = vacuumstate(N)
        squeeze!(g, z)
        v = MPS(g; maxdim=4, maxnumber=maxn)  # the MPS will have maxdim=1 anyway
        @test norm(v) ≤ 1 || norm(v) ≈ 1

        coefficients_expected = Matrix{ComplexF64}(undef, maxn+1, maxn + 1)
        for n1 in 0:maxn
            for n2 in 0:maxn
                coefficients_expected[n1 + 1, n2 + 1] =
                    squeezed_state_coeff(z[1], n1) * squeezed_state_coeff(z[2], n2)
            end
        end

        coefficients_mps = similar(coefficients_expected)
        for n1 in 0:maxn
            for n2 in 0:maxn
                coefficients_mps[n1 + 1, n2 + 1]=dot(
                    MPS(siteinds(v), [string(n1), string(n2)]), v
                )
            end
        end

        # Another check.
        _, S = williamson(g.covariance_matrix)
        Ul, D, Ur = euler(S)
        coefficients_fc = similar(coefficients_expected)
        for n1 in 0:maxn
            for n2 in 0:maxn
                coefficients_fc[n1 + 1, n2 + 1] = GaussianBosonSamplingMPS.franckcondon(
                    [n1, n2], Ul, D, Ur, [0, 0]
                )
            end
        end

        @test abs.(coefficients_mps) ≈ abs.(coefficients_expected) ≈ abs.(coefficients_fc)
        normalize!(v)
        @test all(isone, linkdims(v))
        @test sum(expect(v, "N")) ≈ number(g)
    end

    @testset "Two-mode squeezed vacuum state" begin
        # Build an MPS from a two-mode squeezed vacuum state and check that
        # its coefficients in the eigenbasis of the number operator are as expected:
        #
        #                  +∞   (e^(iθ) tanh(r))ⁿ
        #   s₂(r e^(iθ)) =  ∑  ─────────────────── f(n)⊗f(n)
        #                  n=0       cosh(r)

        maxn = 10
        maxdim = 10_000
        # The singular values of the reduced state over one of the two modes are
        # (tanh r)^2n / (cosh r)^2, so to get them all in the normal mode decomposition
        # with a cutoff ε = 1e-12 we want an r such that (tanh r)^2n < ε ≤ (cosh r)^2 ε
        # for all n ≤ maxn, i.e. r < artanh(ε^(1/2maxn)).
        cutoff = 1e-12
        r = atanh(cutoff^(1/maxn)) * rand()
        θ = 2pi * rand()
        g = vacuumstate(2)
        squeeze2!(g, r*cis(θ), 1, 2)

        v = MPS(g; maxdim=maxdim, maxnumber=maxn)
        sites = siteinds(v)

        coefficients_expected = Diagonal(squeezed2_state_coeff.(r, θ, 0:maxn))
        # coefficients_expected[n, m] := ⟨f(n)⊗f(m), s₂(r e^(iθ))⟩ =
        #
        #                                 (e^(iθ) tanh(r))ⁿ
        #                              = ─────────────────── δₘₙ
        #                                      cosh(r)
        #

        # Another check.
        _, S = williamson(g.covariance_matrix)
        Ul, D, Ur = euler(S)
        coefficients_fc = Diagonal([
            GaussianBosonSamplingMPS.franckcondon([n, n], Ul, D, Ur, [0, 0]) for n in 0:maxn
        ])

        @test norm(v) ≤ 1 || norm(v) ≈ 1

        coefficients_mps = Matrix{ComplexF64}(undef, size(coefficients_expected))
        for n in 0:maxn
            for m in 0:maxn
                coefficients_mps[n + 1, m + 1] = dot(MPS(sites, [string(n), string(m)]), v)
            end
        end

        # The MPS coefficients here get replaced by zero if their square
        # would be less than the cutoff for the singular values.
        replace!(x -> abs2(x) < cutoff ? zero(x) : x, coefficients_expected)
        replace!(x -> abs2(x) < cutoff ? zero(x) : x, coefficients_fc)
        replace!(x -> abs2(x) < cutoff ? zero(x) : x, coefficients_mps)
        @test abs.(coefficients_mps) ≈ abs.(coefficients_expected) ≈ abs.(coefficients_fc)

        normalize!(v)
        @test isapprox(sum(expect(v, "N")), number(g))
    end

    @testset "Several Gaussian operations on two modes" begin
        # Let's operate on an initial vacuum state with a lot of Gaussian maps, trying out
        # all the operations in GaussianStates, and see if the output MPS makes sense.
        maxn = 12
        maxdim = 10_000

        cutoff = 1e-12
        r = atanh(cutoff^(1/maxn)) * rand()
        θ = 2pi * rand()
        g = vacuumstate(2)
        squeeze2!(g, r * cis(θ), 1, 2)
        beamsplitter!(g, rand(), 1, 2)
        phaseshift!(g, [2pi*rand(), 2pi*rand()])
        squeeze!(g, [rand(ComplexF64) ./ 10, rand(ComplexF64) ./ 10])
        beamsplitter!(g, rand(), 1, 2)
        phaseshift!(g, [0, 2pi*rand()])

        v = MPS(g; maxdim=maxdim, maxnumber=maxn, lowerthreshold=1e-14)
        sites = siteinds(v)

        # Check.
        coefficients_fc = Matrix{ComplexF64}(undef, maxn+1, maxn+1)
        _, S = williamson(Symmetric(g.covariance_matrix))
        for n1 in 0:maxn
            for n2 in 0:maxn
                coefficients_fc[n1 + 1, n2 + 1] = GaussianBosonSamplingMPS.franckcondon(
                    [n1, n2], euler(S)..., [0, 0]
                )
            end
        end

        # The MPS coefficients here get replaced by zero if their square
        # would be less than the cutoff for the singular values.
        replace!(x -> abs2(x) < cutoff ? zero(x) : x, coefficients_fc)

        @test norm(v) ≤ 1 || norm(v) ≈ 1

        coefficients_mps = Matrix{ComplexF64}(undef, maxn+1, maxn+1)
        for n in 0:maxn
            for m in 0:maxn
                coefficients_mps[n + 1, m + 1] = dot(MPS(sites, [string(n), string(m)]), v)
            end
        end
        replace!(x -> abs2(x) < cutoff ? zero(x) : x, coefficients_mps)
        @test abs.(coefficients_mps) ≈ abs.(coefficients_fc)

        normalize!(v)
        @test isapprox(sum(expect(v, "N")), number(g))
    end

    @testset "Two-mode squeezing + beam splitter on three modes" begin
        g = vacuumstate(3)

        cutoff = 1e-12
        maxn = 12
        maxdim = 10_000
        r = atanh(cutoff^(1/maxn)) * rand()
        θ = 2pi * rand()
        ζ = r * cis(θ)  # Squeeze parameter
        squeeze2!(g, ζ, 1, 2)
        s12 = GaussianStates._squeeze2matrix(ζ)

        η = 0.5 * rand()  # BS transmittivity
        beamsplitter!(g, η, 2, 3)
        b23 = GaussianStates._beamsplittermatrix(η)

        d, r = williamson(Symmetric(g.covariance_matrix))
        @test d ≈ I
        d23, r23 = williamson(Symmetric(partialtrace(g, 1).covariance_matrix))

        v = MPS(g; maxdim=maxdim, maxnumber=maxn)

        @test norm(v) < 1 || norm(v) ≈ 1
        @test number(g) ≈ sum(expect(v, "N"))
    end
end

@testset verbose=true "Operations on matrix-product states" begin
    @testset "Attenuator channel" begin
        nmodes = 1
        maxnumber = 10
        maxdim = 10

        attenuator_1 = op(
            OpName("attenuator"), SiteType("Boson"), maxnumber, maxnumber; attenuation=1
        )
        # attenuation = 1 corresponds to no attenuation at all, so we should get the
        # identity.
        @test attenuator_1 == I

        attenuator_0 = op(
            OpName("attenuator"), SiteType("Boson"), maxnumber, maxnumber; attenuation=0
        )
        # attenuation = 0 deletes everything, producing |0⟩⟨0| from any input state.
        # The matrix representing this operator in the superboson basis is not zero only in
        # the first row, so that the output vector is proportional to |0⟩, and the first row
        # is 1s and 0s so that it picks up the elements of the state along the diagonal,
        # this way for any matrix m we have V(m) ↦ tr(m) V(|0⟩⟨0|) where V is the
        # vectorisation map.
        @test attenuator_0[1, :] == reshape(I(maxnumber), maxnumber^2) &&
            all(attenuator_0[k, :] == zeros(maxnumber^2) for k in 2:size(attenuator_0, 2))

        sites = sb_siteinds(; nmodes=nmodes, maxnumber=maxnumber)
        N = LocalOperator(sb_index(1) => "n")

        # If the initial state is an eigenvector of the number operator we can derive
        # an explicit analytic formula for the attenuated expectation value of the number
        # operator.
        n_init = 2
        v = MPS(sites, string(n_init))
        n_pre = real(measure(v, N))
        @test n_pre == n_init

        # Check that the operator with attenuation=1 also doesn't change the MPS.
        # We can't test the strict equality `==` since the IDs of the indices may change,
        # even if the underlying data stays the same.
        @test attenuate(v, 1, 1) ≈ v
        @test attenuate(v, 0, 1) ≈ MPS(siteinds(v), "0")

        attenuationrate = 0.1
        v_post = attenuate(v, attenuationrate, 1)
        n_post = real(measure(v_post, N))
        @test n_post ≈ sum(
            n *
            binomial(n_init, n) *
            (1 - attenuationrate^2)^(n_init - n) *
            attenuationrate^(2n) for n in 0:n_init
        )
    end

    @testset "Squeeze operators" begin
        nmodes = 3
        maxnumber = 10
        maxdim = 10

        sites = sb_siteinds(; nmodes=nmodes, maxnumber=maxnumber)
        N = [
            LocalOperator(sb_index(1) => "n")
            LocalOperator(sb_index(2) => "n")
            LocalOperator(sb_index(3) => "n")
        ]

        v = MPS(sites, "0")
        n_pre = measure(v, N)
        @test iszero(sum(n_pre))

        # As always, we need to keep abs(z) low if we don't want to run into trouble with
        # the truncation of the local Hilbert spaces.
        z = rand(3) ./ 10 .* cispi.(2 .* rand(3))
        w = squeeze(squeeze(v, 1, z[1]), 1, -z[1])
        @test w ≈ v

        w = squeeze(v, z)
        n_post = measure(w, N)
        @test sum(n_post) ≈ sum(@. sinh(abs(z))^2)
    end

    @testset "Beam splitter operators" begin
        nmodes = 3
        maxnumber = 10

        sites = sb_siteinds(; nmodes=nmodes, maxnumber=maxnumber)
        N = [
            LocalOperator(sb_index(1) => "n")
            LocalOperator(sb_index(2) => "n")
            LocalOperator(sb_index(3) => "n")
        ]

        v = MPS(sites, ["1", "1", "0", "0", "0", "0"])
        n_pre = measure(v, N)
        @test sum(n_pre) == 1

        w = beamsplitter(v, cospi(1/4), 1, 3)
        n_post = measure(w, N)
        @test sum(n_post) ≈ sum(n_pre)

        ρ_vacuum = MPS(sites, "0")
        w_expected =
            1/2 * add(
                apply(
                    op("a†", sites, sb_index(1)) * conj(op("a†", sites, sb_index(1)+1)),
                    ρ_vacuum,
                ),
                apply(
                    op("a†", sites, sb_index(3)) * conj(op("a†", sites, sb_index(3)+1)),
                    ρ_vacuum,
                ),
                apply(
                    op("a†", sites, sb_index(3)) * conj(op("a†", sites, sb_index(1)+1)),
                    ρ_vacuum,
                ),
                apply(
                    op("a†", sites, sb_index(1)) * conj(op("a†", sites, sb_index(3)+1)),
                    ρ_vacuum,
                );
                alg="directsum",
            )
        @test w_expected ≈ w
    end

    @testset "Displacement operators" begin
        nmodes = 3
        maxnumber = 10

        # With superboson states
        sites = sb_siteinds(; nmodes=nmodes, maxnumber=maxnumber)
        N = [LocalOperator(sb_index(j) => "n") for j in 1:nmodes]

        v = MPS(sites, "0")
        n_pre = measure(v, N)
        @test sum(n_pre) == 0

        α = rand(ComplexF64, nmodes) ./ 10
        w = displace(v, α)
        n_post = measure(w, N)
        @test sum(n_post) ≈ norm(α)^2

        # With standard pure states
        sites = siteinds("Boson", nmodes; dim=maxnumber+1)
        v = MPS(sites, "0")
        n_pre = expect(v, "n")
        @test sum(n_pre) == 0

        α = rand(ComplexF64, nmodes) ./ 10
        w = displace_pure(v, α)
        n_post = expect(w, "n")
        @test sum(n_post) ≈ norm(α)^2
    end
end

@testset "First and second moments from MPS" begin
    nmodes = 6
    maxnumber = 4
    sites = sb_siteinds(; nmodes=nmodes, maxnumber=maxnumber)
    v = MPS(sites, "0")
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
    @test isapprox(covariancematrix(vv), g.covariance_matrix; rtol=rtol)
end

@testset verbose=true "Superboson matrix-product states" begin
    nmodes = 10
    maxnumber = 5
    s = siteinds("Boson", nmodes; dim=maxnumber+1)

    @testset "Measurements" begin
        v = random_mps(s; linkdims=10)
        vv = sb_outer(v)
        @test GaussianBosonSamplingMPS.sb_trace(vv) ≈ 1
        @test expect(v, "n") ≈ measure(vv, [LocalOperator(sb_index(j) => "n") for j in 1:nmodes])
    end

    @testset "Sampling" begin
        nsamples = 10
        ns = rand(0:maxnumber, nmodes)
        v = MPS(s, string.(ns))
        orthogonalize!(v, 1)  # required for sampling
        vv = sb_outer(v)
        r_std = []
        r_sb = []
        for i in nsamples
            push!(r_std, sample(v))
            push!(r_sb, sb_sample(vv))
        end
        # Since the state is precisely an element of the Fock basis, the sampling procedure
        # should output the state itself each time. We check that:
        # · the sampling is identical between the two MPSs
        @test allequal(r_std) && allequal(r_sb) && all(r_std .== r_sb)
        # · the sampled basis states are all equal to the one we used to build the MPS at
        #   the beginning (note that occupation numbers go from `0` to `maxnumber` while MPS
        #   site indices go from `1` to `maxnumber+1`).
        @test all(first(r_sb) .== (1 .+ ns))
    end
end
