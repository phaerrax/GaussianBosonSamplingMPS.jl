@testset verbose=true "Superboson matrix-product states" begin
    nmodes = 10
    maxnumber = 5
    s = siteinds("Boson", nmodes; dim=maxnumber+1)

    @testset "Expectation values" begin
        v = random_mps(ComplexF64, s; linkdims=10)
        vv = sb_outer(v)
        @test tr(vv) ≈ 1
        @test expect(v, "n") ≈ expect(vv, "n")
        @test expect(v, "x"; sites=2:4) ≈ expect(vv, "x"; sites=2:4)
    end

    @testset "Correlation matrices" begin
        v = random_mps(ComplexF64, s; linkdims=5)
        vv = sb_outer(v)
        @test correlation_matrix(v, "x", "n") ≈ correlation_matrix(vv, "x", "n")
        @test correlation_matrix(v, "p", "x"; sites=2:4) ≈
            correlation_matrix(vv, "p", "x"; sites=2:4)
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
            push!(r_sb, sample(vv))
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
