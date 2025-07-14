@testset verbose=true "Operations on matrix-product states" begin
    @testset "Attenuator channel" begin
        nmodes = 1
        maxnumber = 5

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

        # If the initial state is an eigenvector of the number operator we can derive
        # an explicit analytic formula for the attenuated expectation value of the number
        # operator.
        n_init = 2
        v = SuperBosonMPS(sites, string(n_init))
        n_pre = real(expect(v, "n"; sites=1))
        @test n_pre == n_init

        # Check that the operator with attenuation=1 also doesn't change the MPS.
        # We can't test the strict equality `==` since the IDs of the indices may change,
        # even if the underlying data stays the same.
        @test attenuate(v, 1, 1) ≈ v
        @test attenuate(v, 0, 1) ≈ SuperBosonMPS(sites, "0")

        attenuationrate = 0.1
        v_post = attenuate(v, attenuationrate, 1)
        n_post = expect(v_post, "n"; sites=1)
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

        sites = sb_siteinds(; nmodes=nmodes, maxnumber=maxnumber)

        v = SuperBosonMPS(sites, "0")
        n_pre = expect(v, "n")
        @test iszero(sum(n_pre))

        # As always, we need to keep abs(z) low if we don't want to run into trouble with
        # the truncation of the local Hilbert spaces.
        z = rand(3) ./ 10 .* cispi.(2 .* rand(3))
        w = squeeze(squeeze(v, 1, z[1]), 1, -z[1])
        @test w ≈ v

        w = squeeze(v, z)
        n_post = expect(w, "n")
        @test sum(n_post) ≈ sum(@. sinh(abs(z))^2)
    end

    @testset "Beam splitter operators" begin
        nmodes = 3
        maxnumber = 3

        sites = sb_siteinds(; nmodes=nmodes, maxnumber=maxnumber)

        v = SuperBosonMPS(sites, ["1", "0", "0"])
        n_pre = expect(v, "n")
        @test sum(n_pre) == 1

        w = beamsplitter(v, cospi(1/4), 1, 3)
        n_post = expect(w, "n")
        @test sum(n_post) ≈ sum(n_pre)

        ρ_vacuum = SuperBosonMPS(sites, "0")
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

        v = SuperBosonMPS(sites, "0")
        n_pre = expect(v, "n")
        @test sum(n_pre) == 0

        α = rand(ComplexF64, nmodes) ./ 10
        w = displace(v, α)
        n_post = expect(w, "n")
        @test sum(n_post) ≈ norm(α)^2

        # With standard pure states
        sites = siteinds("Boson", nmodes; dim=maxnumber+1)
        v = MPS(sites, "0")
        n_pre = expect(v, "n")
        @test sum(n_pre) == 0

        α = rand(ComplexF64, nmodes) ./ 10
        w = displace(v, α)
        n_post = expect(w, "n")
        @test sum(n_post) ≈ norm(α)^2
    end
end
