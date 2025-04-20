using LinearAlgebra, GaussianStates

"""
    normal_mode_eigenvalue(x, m)

Return the result of the function

```math
            2  ⎛ x-1 ⎞ m
  f(x, m) =  ─── ⎜ ────⎟
             x+1 ⎝ x+1 ⎠
```

mapping a symplectic eigenvalue `x` of a Gaussian state to the respective singular
value associated to the `m`-photon state in the normal-mode decomposition (see [1],
Eq. (3.60) at page 52).

# References

[1] Serafini, A. (2017). Quantum Continuous Variables: A Primer of Theoretical Methods
(1st ed.). CRC Press.
[https://doi.org/10.1201/9781315118727](https://doi.org/10.1201/9781315118727)
"""
function normal_mode_eigenvalue(symplectic_eigenvalue, number)
    z = symplectic_eigenvalue
    m = number
    return 2 / (z + 1) * ((z - 1) / (z + 1))^m
end

function largest_normal_mode_eigenvalues(
    symplectic_eigenvalues, N, maxnumber; lowerthreshold=1e-12
)
    # Some symplectic eigenvalues might be slightly less than 1 due to numerical rounding.
    # We cannot allow this, because it can lead to `normal_mode_eigenvalue` being
    # negative, so we replace them by 1. First we check that there isn't any value that is
    # significantly less than 1.
    @assert all(symplectic_eigenvalues .> 1 .|| symplectic_eigenvalues .≈ 1)
    replace!(x -> x ≈ 1 ? one(x) : x, symplectic_eigenvalues)

    num0 = [[n] for n in 0:maxnumber]
    num = deepcopy(num0)
    evs = normal_mode_eigenvalue.(symplectic_eigenvalues[1], 0:maxnumber)

    if length(symplectic_eigenvalues) == 1
        # Special branch for when there's just one symplectic eigenvalue.
        # We cannot just skip the loop otherwise all values in `evs`, even those below
        # `lowerthreshold`, will be kept in the final result. This is fine in principle,
        # it would just result in a lazier approximation in the end, but we'd like to be
        # consistent across all sites of the MPS.

        # In this case we have a single symplectic eigenvalue so it's easy to order the
        # normal-mode eigenvalues: they decrease with the occupation number.
        sort!(evs; rev=true)
        filter!(>(lowerthreshold), evs)
        return evs, first(num0, length(evs))
    else
        for i in 2:length(symplectic_eigenvalues)
            next_sequence = normal_mode_eigenvalue.(symplectic_eigenvalues[i], 0:maxnumber)
            evs_product = Vector{Float64}(undef, length(evs) * length(next_sequence))
            for n in eachindex(evs), m in eachindex(next_sequence)
                evs_product[(n - 1) * length(next_sequence) + m] = evs[n] * next_sequence[m]
            end

            idx_above_threshold = findall(>(lowerthreshold), evs_product)
            filter!(>(lowerthreshold), evs_product)

            largest_evs_idxs = last(sortperm(evs_product), N)
            evs = reverse(evs_product[largest_evs_idxs])

            idx_above_threshold = reverse(idx_above_threshold[largest_evs_idxs])

            num =
                vcat.(
                    num[div.(idx_above_threshold .- 1, maxnumber .+ 1) .+ 1],
                    num0[mod.(idx_above_threshold .- 1, maxnumber .+ 1) .+ 1],
                )
        end

        # Original code. Here we sort and keep the largest `maxdim` values, but this is
        # something that can be easily done outside of this function.
        # idx = last(sortperm(evs), maxdim)
        # idx_sorted = idx[sortperm(evs[idx])]
        # evs = reverse(evs[idx_sorted])
        # num = reverse(num[idx_sorted])

        perm_decreasing = sortperm(evs; rev=true)
        return evs[perm_decreasing], num[perm_decreasing]
    end
end

"""
    normal_mode_decomposition(g::GaussianState, N, maxnumber; kwargs...)

Compute the normal-mode decomposition ([1], Eq. (3.60) at page 52) of the Gaussian state `g`
up to the `N` largest eigenvalues, calculated by considering only the ``k``-particle sectors
for ``k ≤ maxnumber`` on each mode.
Return `vals, nums, S` where `vals` contains the eigenvalues in decreasing order, `nums`
contains the occupation numbers of the Fock basis vector associated to the corresponding
value in `vals`, and `S` is the symplectic matrix from the Williamson decomposition of `g`.

The output satisfies the following identities:

```julia-repl
julia> D, S = williamson(Symmetric(g.covariance_matrix));

julia> d = diag(D)[1:2:end];

julia> normalmode_evs, nums, S' = normal_mode_decomposition(g, N, maxnumber);

julia> all(
    val == prod(normal_mode_eigenvalue(d[j], m[j]) for j in eachindex(m)) for
    (val, m) in zip(normalmode_evs, nums)
)
true

julia> S == S'
true
```

# References

[1] Serafini, A. (2017). Quantum Continuous Variables: A Primer of Theoretical Methods
(1st ed.). CRC Press.
[https://doi.org/10.1201/9781315118727](https://doi.org/10.1201/9781315118727)
"""
function normal_mode_decomposition(g::GaussianState, N, maxnumber; kwargs...)
    D, S = williamson(Symmetric(g.covariance_matrix))
    sp_evals = diag(D)[1:2:end]
    nm_evals, num_idxs = largest_normal_mode_eigenvalues(sp_evals, N, maxnumber; kwargs...)
    return nm_evals, num_idxs, S
end

function twomodesqueezing(ζ, n, k1, k2)
    @assert 1 ≤ k1 ≤ n && 1 ≤ k2 ≤ n
    F = Matrix{Float64}(I, 2n, 2n)

    θ = angle(ζ)
    r = abs(ζ)

    S = [[cos(θ) sin(θ)]; [sin(θ) -cos(θ)]]
    F[(2k1 - 1):(2k1), (2k1 - 1):(2k1)] .= cosh(r) .* I(2)
    F[(2k1 - 1):(2k1), (2k2 - 1):(2k2)] .= -sinh(r) .* S
    F[(2k2 - 1):(2k2), (2k1 - 1):(2k1)] .= -sinh(r) .* S
    F[(2k2 - 1):(2k2), (2k2 - 1):(2k2)] .= cosh(r) .* I(2)
    return F
end

# Eq. (65)-(66) in [1]
# Construct the B(p) matrix with B[unroll(p), unroll(p)]
unroll(p) = reduce(vcat, [repeat([i], p[i]) for i in eachindex(p)])

directsum(A, B) = [A zeros(size(A, 1), size(B, 2)); zeros(size(B, 1), size(A, 2)) B]

"""
    franckcondon(m, α::AbstractVector, Wl::AbstractMatrix, Wr::AbstractMatrix, n)

Compute the matrix element ``⟨m| D(α) U(Wl) U(Wr) |n⟩`` according to the algorithm
presented in [1]. `Wl` and `Wr` are symplectic matrices while `α` is a complex vector.
If ``N`` is the number of modes of the system, then

- `m` and `n` are tuples of ``N`` natural numbers
- `α` is a vector of ``N`` complex numbers
- `Wl` and `Wr` are ``2N × 2N`` symplectic matrices

# References

[1] Nicolás Quesada, ‘Franck-Condon factors by counting perfect matchings of graphs with
loops’. [The Journal of Chemical Physics 150.16 (2019)](https://doi.org/10.1063/1.5086387).
"""
function franckcondon(m, α, Wl, Wr, n)
    # If N is the number of modes of the system:
    # · m and n are tuples of N natural numbers
    # · α is a vector of N complex numbers
    # · Wl and Wr are 2N×2N symplectic matrices
    @assert length(m) == length(n) == length(α)
    L = length(m)
    @assert all(size(Wl) .== 2L .&& size(Wr) .== 2L)

    # U(Wl)* U(Wr) = U(Wl⁻¹ Wr) = U(Ul * S * Ur) = U(Ul) U(S) U(Ur)
    # through the Euler decomposition.
    # Ul and Ur are orthogonal symplectic transformations, S is a diagonal squeezing matrix.
    Ul, S, Ur = euler(inv(Wl) * Wr)

    Ulext = directsum(Ul, I(2L))
    Urext = directsum(Ur, I(2L))
    Sext = directsum(S, I(2L))

    t = @. asinh(sqrt(n))
    sq2m = prod(twomodesqueezing(t[k], 2L, k, L + k) for k in 1:L)
    Vl, K, _ = euler(Ulext * Sext * Urext * sq2m)
    # Here `euler` gives us 4L × 4L real, symplectic matrices; we need complex, unitary
    # 2L × 2L matrices instead. Vl and Vr are orthogonal so there exists an unitary matrix
    # which is equivalent to them, and we extract it as follows (we only need Vl)
    Vl_xxpp = GaussianStates.permute_to_xxpp(Vl)
    uVl = complex.(Vl_xxpp[1:(2L), 1:(2L)], Vl_xxpp[(2L + 1):end, 1:(2L)])
    Λ = log.(diag(GaussianStates.permute_to_xxpp(K))[1:(2L)])
    @assert uVl * uVl' ≈ I

    p = [m; n]
    αext = [α; zeros(L)]

    B = uVl * Diagonal(tanh.(Λ)) * transpose(uVl)
    ζ = αext - B * conj(αext)
    T =
        exp(-1 / 2 * (norm(αext)^2 - dot(αext, B, αext))) /
        sqrt(prod(@. factorial(p) * cosh(Λ)))

    R = prod(@. cosh(t) / (tanh(t)^n))
    p_inds = unroll(p)
    Bp = B[p_inds, p_inds]
    ζp = ζ[p_inds]
    lhf = loophafnian(Bp + Diagonal(ζp .- diag(Bp)))
    return R * T * lhf
end

function franckcondon(m, Wl, Wr, n)
    # If N is the number of modes of the system:
    # · m and n are tuples of N natural numbers
    # · α is a vector of N complex numbers
    # · Wl and Wr are 2N×2N symplectic matrices
    @assert length(m) == length(n)
    L = length(m)
    @assert all(size(Wl) .== 2L .&& size(Wr) .== 2L)

    # U(Wl)* U(Wr) = U(Wl⁻¹ Wr) = U(Ul * S * Ur) = U(Ul) U(S) U(Ur)
    # through the Euler decomposition.
    # Ul and Ur are orthogonal symplectic transformations, S is a diagonal squeezing matrix.
    Ul, S, Ur = euler(inv(Wl) * Wr)

    Ulext = directsum(Ul, I(2L))
    Urext = directsum(Ur, I(2L))
    Sext = directsum(S, I(2L))

    t = @. asinh(sqrt(n))
    sq2m = prod(twomodesqueezing(t[k], 2L, k, L + k) for k in 1:L)
    Vl, K, _ = euler(Ulext * Sext * Urext * sq2m)
    # Here `euler` gives us 4L × 4L real, symplectic matrices; we need complex, unitary
    # 2L × 2L matrices instead. Vl and Vr are orthogonal so there exists an unitary matrix
    # which is equivalent to them, and we extract it as follows (we only need Vl)
    Vl_xxpp = GaussianStates.permute_to_xxpp(Vl)
    uVl = complex.(Vl_xxpp[1:(2L), 1:(2L)], Vl_xxpp[(2L + 1):end, 1:(2L)])
    Λ = log.(diag(GaussianStates.permute_to_xxpp(K))[1:(2L)])
    @assert uVl * uVl' ≈ I

    p = [m; n]

    B = uVl * Diagonal(tanh.(Λ)) * transpose(uVl)
    T = 1 / sqrt(prod(@. factorial(p) * cosh(Λ)))
    R = prod(@. cosh(t) / (tanh(t)^n))

    p_inds = unroll(p)
    Bp = B[p_inds, p_inds]
    hf = hafnian(Bp .- diag(Bp))
    return R * T * hf
end

function hafnian(A)
    if size(A, 1) != size(A, 2)
        error("not a square matrix")
    end
    if isodd(size(A, 1))
        return zero(eltype(A))
    end
    if size(A, 1) == 0
        return one(eltype(A))
    end

    # Ported from the sample GNU Octave function provided by The Walrus at
    # https://github.com/XanaduAI/thewalrus/blob/master/octave/hafnian.m
    n = div(size(A, 1), 2)
    P = kron(I(n), [[0 1]; [1 0]])
    A = A * P
    comb = zeros(eltype(A), 2, n + 1)
    haf = 0
    for m in 1:(2^n - 1)
        sieve = reverse(digits(m; base=2, pad=n)) .== 1
        P = kron(sieve, [1, 1])
        idx = findall(P .== 1)
        B = A[idx, idx]
        B_evals, _ = eigen(B)
        cnt = 1
        comb[1, :] .= zeros(eltype(A), n + 1)
        comb[1, 1] = 1
        for i in 1:n
            factor = sum(B_evals .^ i) / (2i)
            powfactor = 1
            cnt = 3 - cnt
            comb[cnt, :] = comb[3 - cnt, :]
            for j in round.(Int, 1:(n / i))
                powfactor = powfactor * factor / j
                for k in (i * j + 1):(n + 1)
                    comb[cnt, k] += comb[3 - cnt, k - i * j] * powfactor
                end
            end
        end
        if iseven(sum(sieve) - n)
            haf += comb[cnt, n + 1]
        else
            haf -= comb[cnt, n + 1]
        end
    end
    return haf
end

function loophafnian(A)
    if size(A, 1) != size(A, 2)
        error("not a square matrix")
    end
    if isodd(size(A, 1))
        return zero(eltype(A))
    end
    if size(A, 1) == 0
        return one(eltype(A))
    end

    # Ported from the sample GNU Octave function provided by The Walrus at
    # https://github.com/XanaduAI/thewalrus/blob/master/octave/loophafnian.m
    n = div(size(A, 1), 2)
    D = diag(A)
    P = kron(I(n), [[0 1]; [1 0]])
    A = A * P
    C = transpose(P * D)
    comb = zeros(eltype(A), 2, n + 1)
    lhaf = 0
    for m in 1:(2^n - 1)
        sieve = reverse(digits(m; base=2, pad=n)) .== 1
        P = kron(sieve, [1, 1])
        idx = findall(P .== 1)
        B = A[idx, idx]
        C1 = C[idx]'
        D1 = D[idx]
        B_evals, _ = eigen(B)
        cnt = 1
        comb[1, :] .= zeros(eltype(A), n + 1)
        comb[1, 1] = 1
        for i in 1:n
            factor = sum(B_evals .^ i) / (2i) + dot(C1, D1) / 2
            C1 = C1 * B
            powfactor = 1
            cnt = 3 - cnt
            comb[cnt, :] = comb[3 - cnt, :]
            for j in round.(Int, 1:(n / i))
                powfactor = powfactor * factor / j
                for k in (i * j + 1):(n + 1)
                    comb[cnt, k] += comb[3 - cnt, k - i * j] * powfactor
                end
            end
        end
        if iseven(sum(sieve) - n)
            lhaf += comb[cnt, n + 1]
        else
            lhaf -= comb[cnt, n + 1]
        end
    end
    return lhaf
end

function _MPSblock(n_k, num_idxs_left, S_left, num_idxs_right, S_right)
    m = Matrix{ComplexF64}(undef, length(num_idxs_right), length(num_idxs_left))
    for i in axes(m, 1)
        for j in axes(m, 2)
            m[i, j] = franckcondon(
                [n_k; num_idxs_left[j]], dirsum(I(2), S_left), S_right, num_idxs_right[i]
            )
        end
    end
    return m
end

function _MPSblock_end(n_k, num_idxs_right, S_right)
    # Dedicated function for the last site (we can't use _MPSblock) with "empty" `S_left`
    # and `num_idxs_left` otherwise we'd get a 0xN matrix since `length(num_idxs_left) == 0`
    m = Matrix{ComplexF64}(undef, length(num_idxs_right), 1)
    for i in axes(m, 1)
        m[i, 1] = franckcondon([n_k], I(2), S_right, num_idxs_right[i])
    end
    return m
end

"""
    MPS(g::GaussianState, maxdim, maxnumber)

Build an MPS representation of the Gaussian state `g` with bond dimension up to `maxdim`,
truncating the Fock space of each mode at the `maxnumber`-particle sector.
"""
function mps_matrices(g::GaussianState, maxdim, maxnumber; nvals=nmodes(g)^2)
    if !isapprox(purity(g), 1)
        error("the Gaussian state must be pure.")
    end

    N = nmodes(g)
    nm_evals_right, num_idxs_right, S_right = normal_mode_decomposition(g, N, maxnumber)
    # This is the normal-mode decomposition of a pure state, so we should find only one
    # eigenvalue, 1, corresponding to the vacuum.
    @assert length(nm_evals_right) == length(num_idxs_right) == 1
    @assert nm_evals_right[1] ≈ 1
    @assert all(num_idxs_right[1] .== 0)

    A = []  # matrices of the MPS

    for bond_idx in 1:(N - 1)
        gpart = partialtrace(g, 1:bond_idx)
        _, num_idxs_left, S_left = normal_mode_decomposition(gpart, nvals, maxnumber)
        # (We don't need the eigenvalues.)

        # At step k we have the decompositions of [1 ... k] as "right" and
        # [k+1 ... N] as "left".
        num_idxs_left = first(num_idxs_left, maxdim)
        push!(
            A,
            [
                _MPSblock(n_k, num_idxs_left, S_left, num_idxs_right, S_right) for
                n_k in 0:maxnumber
            ],
        )

        num_idxs_right = num_idxs_left
        S_right = S_left
    end
    push!(A, [_MPSblock_end(n_k, num_idxs_right, S_right) for n_k in 0:maxnumber])

    return A
end
