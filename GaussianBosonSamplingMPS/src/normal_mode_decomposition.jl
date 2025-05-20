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
            evs_product = Vector{eltype(symplectic_eigenvalues)}(
                undef, length(evs) * length(next_sequence)
            )
            for n in eachindex(evs), m in eachindex(next_sequence)
                evs_product[(n - 1) * length(next_sequence) + m] = evs[n] * next_sequence[m]
            end

            idx_above_threshold = findall(>(lowerthreshold), evs_product)
            filter!(>(lowerthreshold), evs_product)

            largest_evs_idxs = last(sortperm(evs_product), N)
            evs = reverse(evs_product[largest_evs_idxs])

            idx_above_threshold = reverse(idx_above_threshold[largest_evs_idxs])

            num = vcat.(
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
