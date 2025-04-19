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
