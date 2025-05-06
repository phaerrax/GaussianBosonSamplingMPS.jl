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

    @testset "Number basis coefficients of squeezed vacuum state" begin
        # Build an MPS from a product of squeezed vacuum states s(r e^(iθ)) and check that
        # its coefficients in the eigenbasis of the number operator are as expected:
        #
        #                    1     +∞                    √(2n)!
        #   s(r e^(iθ)) = ────────  ∑ (-e^(iθ) tanh(r))ⁿ ────── |2n⟩
        #                 √cosh(r) n=0                    2ⁿn!

        N = 2
        maxn = 12
        cutoff = 1e-12
        r = atanh(cutoff^(1/maxn)) * rand(N)
        θ = 2pi * rand(N)
        z = r .* cis.(θ)

        g = vacuumstate(N)
        squeeze!(g, z)
        v = MPS(g, 4, maxn)  # the MPS will have maxdim=1 anyway
        normalize!(v)
        @test all(isone, linkdims(v))
        @test sum(expect(v, "N")) ≈ number(g)

        function coeff(x, m)
            if isodd(m)
                return 0
            else
                n = div(m, 2)
                r, θ = abs(x), angle(x)
                return (-cis(θ) * tanh(r))^n * sqrt(factorial(2n)) /
                       (2^n * factorial(n) * sqrt(cosh(r)))
            end
        end

        coefficients_expected = Matrix{ComplexF64}(undef, maxn+1, maxn + 1)
        for n1 in 0:maxn
            for n2 in 0:maxn
                coefficients_expected[n1 + 1, n2 + 1] = coeff(z[1], n1) * coeff(z[2], n2)
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
        coefficients_fc = similar(coefficients_expected)
        for n1 in 0:maxn
            for n2 in 0:maxn
                coefficients_fc[n1 + 1, n2 + 1] = GaussianBosonSamplingMPS.franckcondon(
                    [n1, n2], I(4), S, [0, 0]
                )
            end
        end

        @test coefficients_mps ≈ coefficients_expected ≈ coefficients_fc
    end

    @testset "Number basis coefficients of two-mode squeezed vacuum state" begin
        # Build an MPS from a two-mode squeezed vacuum state and check that
        # its coefficients in the eigenbasis of the number operator are as expected:
        #
        #                  +∞   (-e^(iθ) tanh(r))ⁿ
        #   s₂(r e^(iθ)) =  ∑  ──────────────────── f(n)⊗f(n)
        #                  n=0        cosh(r)

        #                ↙ plus or minus here?? (Franck-Condon function says minus...)
        coeff(r, θ, n) = (-cis(θ) * tanh(r))^n / cosh(r)
        N = 8
        # The singular values of the reduced state over one of the two modes are
        # (tanh r)^2n / (cosh r)^2, so to get them all in the normal mode decomposition
        # with a cutoff ε = 1e-12 we want an r such that (tanh r)^2n < ε ≤ (cosh r)^2 ε
        # for all n ≤ N, i.e. r < artanh(ε^(1/2N)).
        cutoff = 1e-12
        r = atanh(cutoff^(1/N)) * rand()
        θ = 2pi * rand()
        g = vacuumstate(2)
        squeeze2!(g, r*cis(θ), 1, 2)

        v = MPS(g, N^2, N)
        sites = siteinds(v)

        coefficients_expected = Diagonal(coeff.(r, θ, 0:N))
        # coefficients_expected[n, m] := ⟨f(n)⊗f(m), s₂(r e^(iθ))⟩ =
        #
        #                                 (-e^(iθ) tanh(r))ⁿ
        #                              = ──────────────────── δₘₙ
        #                                       cosh(r)
        #

        coefficients_mps = Matrix{ComplexF64}(undef, size(coefficients_expected))
        for n in 0:N
            for m in 0:N
                coefficients_mps[n + 1, m + 1] = dot(MPS(sites, [string(n), string(m)]), v)
            end
        end

        # Another check.
        _, S = williamson(g.covariance_matrix)
        coefficients_fc = Diagonal([
            GaussianBosonSamplingMPS.franckcondon([n, n], I(4), S, [0, 0]) for n in 0:N
        ])

        # The MPS coefficients here get replaced by zero if their square
        # would be less than the cutoff for the singular values.
        replace!(x -> abs2(x) < cutoff ? zero(x) : x, coefficients_mps)
        replace!(x -> abs2(x) < cutoff ? zero(x) : x, coefficients_expected)
        replace!(x -> abs2(x) < cutoff ? zero(x) : x, coefficients_fc)
        @test coefficients_mps ≈ coefficients_expected ≈ coefficients_fc
        @test norm(v) ≈ 1
    end
end
