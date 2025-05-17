using Test
using GaussianStates, ITensors, ITensorMPS, LinearAlgebra
using GaussianBosonSamplingMPS
using GaussianBosonSamplingMPS: hafnian, loophafnian, dirsum

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
        s = MPS(g, 5, 5)
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

        @test coefficients_expected ≈ coefficients_fc
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
        v = MPS(g, 4, maxn)  # the MPS will have maxdim=1 anyway
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

        @test coefficients_mps ≈ coefficients_expected ≈ coefficients_fc
        normalize!(v)
        @test all(isone, linkdims(v))
        @test sum(expect(v, "N")) ≈ number(g)
    end

    rtol = 1e-6
    @testset "Two-mode squeezed vacuum state (rtol = $rtol)" begin
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

        v = MPS(g, maxdim, maxn)
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
        normalize!(v)

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
        @test coefficients_mps ≈ coefficients_expected ≈ coefficients_fc
        @test isapprox(sum(expect(v, "N")), number(g); rtol=rtol)
    end

    @testset "Several Gaussian operations on two modes (rtol = $rtol)" begin
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

        v = MPS(g, maxdim, maxn; lowerthreshold=1e-14)
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
        normalize!(v)
        @test isapprox(coefficients_mps, coefficients_fc; rtol=rtol)
        @test isapprox(sum(expect(v, "N")), number(g); rtol=rtol)
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

        v = MPS(g, maxdim, maxn)

        @test norm(v) < 1 || norm(v) ≈ 1
        @test number(g) ≈ sum(expect(v, "N"))
    end
end
