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
