using Test
using GaussianStates, ITensors, ITensorMPS
using GaussianBosonSamplingMPS

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

        N = 4
        maxn = 10
        ζ = rand(ComplexF64, N)
        g = vacuumstate(N)
        squeeze!(g, ζ)
        v = MPS(g, 2, maxn)  # maxdim=2 is more than enough, the MPS will have maxdim=1
        sites = siteinds(v)

        coefficients_expected = zeros(ComplexF64, N, maxn + 1)
        for j in axes(coefficients_expected, 1)
            for k in axes(coefficients_expected, 2)
                if iseven(k - 1)
                    n = div(k - 1, 2)
                    coefficients_expected[j, k] =
                        1 / sqrt(cosh(abs(ζ[j]))) *
                        (-cis(angle(ζ[j])) * tanh(abs(ζ[j])))^n *
                        sqrt(factorial(2n)) / (2^n * factorial(n))
                end
            end
        end

        coefficients_mps = Matrix{ComplexF64}(undef, N, maxn + 1)
        coefficients_mps[1, :] .= [
            scalar(v[1] * onehot(linkind(v, 1) => 1) * onehot(siteind(v, 1) => n + 1)) for
            n in 0:maxn
        ]
        for j in 2:(N - 1)
            coefficients_mps[j, :] .= [
                scalar(
                    v[j] *
                    onehot(linkind(v, j - 1) => 1) *
                    onehot(linkind(v, j) => 1) *
                    onehot(siteind(v, j) => n + 1),
                ) for n in 0:maxn
            ]
        end
        coefficients_mps[N, :] .= [
            scalar(v[N] * onehot(linkind(v, N - 1) => 1) * onehot(siteind(v, N) => n + 1))
            for n in 0:maxn
        ]
        @test coefficients_mps ≈ coefficients_expected
    end
end
