function twomodesqueezing(::Type{T}, ζ, n, k1, k2) where {T<:Number}
    @assert 1 ≤ k1 ≤ n && 1 ≤ k2 ≤ n
    F = Matrix{T}(I, 2n, 2n)

    θ = angle(ζ)
    r = abs(ζ)

    S = [
        cos(θ) sin(θ)
        sin(θ) -cos(θ)
    ]
    F[(2k1 - 1):(2k1), (2k1 - 1):(2k1)] .= cosh(r) .* I(2)
    F[(2k1 - 1):(2k1), (2k2 - 1):(2k2)] .= sinh(r) .* S
    F[(2k2 - 1):(2k2), (2k1 - 1):(2k1)] .= sinh(r) .* S
    F[(2k2 - 1):(2k2), (2k2 - 1):(2k2)] .= cosh(r) .* I(2)
    return F
end

twomodesqueezing(ζ, n, k1, k2) = twomodesqueezing(Float64, ζ, n, k1, k2)

# Eq. (65)-(66) in [1]
# Construct the B(p) matrix with B[unroll(p), unroll(p)]
unroll(p) = reduce(vcat, [repeat([i], p[i]) for i in eachindex(p)])

dirsum(A, B) = [A zeros(size(A, 1), size(B, 2)); zeros(size(B, 1), size(A, 2)) B]

"""
    franckcondon(m, α, Wl, S, Wr, n)
    franckcondon(m, Wl, S, Wr, n)

Compute the matrix element ``⟨m| D(α) U(Wl) U(S) U(Wr) |n⟩`` according to the algorithm
presented in [1]. `Wl`, `S` and `Wr` are symplectic matrices, in particular `Wl` and `Wr`
are also orthogonal, and `α` is a complex vector (which defaults to the zero vector).
If ``N`` is the number of modes of the system, then

- `m` and `n` are tuples of ``N`` natural numbers,
- `α` is a vector of ``N`` complex numbers,
- `Wl`, `S` and `Wr` are ``2N × 2N`` matrices.

# References

[1] Nicolás Quesada, ‘Franck-Condon factors by counting perfect matchings of graphs with
loops’. [The Journal of Chemical Physics 150.16 (2019)](https://doi.org/10.1063/1.5086387).
"""
function franckcondon end

function franckcondon(m, α, Ul, S, Ur, n)
    L = length(m)
    @assert L == length(n) == length(α)
    @assert all(size(Ul) .== 2L .&& size(Ur) .== 2L .&& size(S) .== 2L)

    t = @. asinh(sqrt(n))
    sq2m = prod(twomodesqueezing(t[k], 2L, k, L + k) for k in 1:L)
    Vl, K, _ = euler(dirsum(Ul * S * Ur, I(2L)) * sq2m)
    # Here `euler` gives us 4L × 4L real, symplectic matrices; we need complex, unitary
    # 2L × 2L matrices instead. Vl and Vr are orthogonal so there exists an unitary matrix
    # which is equivalent to them, and we extract it as follows (we only need Vl)
    uVl = GaussianStates.symplectic_to_unitary(GaussianStates.permute_to_xxpp(Vl))
    Λ = log.(diag(K)[1:2:end])
    @assert uVl * uVl' ≈ I

    p = [m; n]
    αext = [α; zeros(L)]

    B = uVl * Diagonal(tanh.(Λ)) * transpose(uVl)
    ζ = αext - B * conj(αext)
    T =
        exp(-1 / 2 * (norm(αext)^2 - dot(αext, B, conj(αext)))) /
        sqrt(prod(@. factorial(p) * cosh(Λ)))

    R = prod(@. cosh(t) / (tanh(t))^n)
    p_inds = unroll(p)
    Bp = B[p_inds, p_inds]
    ζp = ζ[p_inds]
    lhf = loophafnian(Bp + Diagonal(ζp .- diag(Bp)))
    return R * T * lhf
end

function franckcondon(m, Ul, S, Ur, n)
    L = length(m)
    @assert L == length(n)
    @assert all(size(Ul) .== 2L .&& size(Ur) .== 2L .&& size(S) .== 2L)

    t = @. asinh(sqrt(n))
    sq2m = prod(twomodesqueezing(t[k], 2L, k, L + k) for k in 1:L)
    Vl, K, Vr = euler(dirsum(Ul * S * Ur, I(2L)) * sq2m)
    # Here `euler` gives us 4L × 4L real, symplectic matrices; we need complex, unitary
    # 2L × 2L matrices instead. Vl and Vr are orthogonal so there exists an unitary matrix
    # which is equivalent to them, and we extract it as follows (we only need Vl)
    uVl = GaussianStates.symplectic_to_unitary(GaussianStates.permute_to_xxpp(Vl))
    Λ = log.(diag(K)[1:2:end])
    @assert uVl * uVl' ≈ I

    p = [m; n]

    B = uVl * Diagonal(tanh.(Λ)) * transpose(uVl)
    T = 1 / sqrt(prod(@. factorial(p) * cosh(Λ)))
    R = prod(@. cosh(t) / (tanh(t))^n)

    p_inds = unroll(p)
    Bp = B[p_inds, p_inds]
    hf = hafnian(Bp - Diagonal(Bp))
    return R * T * hf
end

function _MPSblock(
    ::Type{T}, n_k, num_idxs_left, U_left, num_idxs_right, U_right
) where {T<:Number}
    m = Matrix{T}(undef, length(num_idxs_right), length(num_idxs_left))
    Ul, S, Ur = euler(dirsum(I(2), inv(U_left)) * U_right)
    for i in axes(m, 1)
        for j in axes(m, 2)
            m[i, j] = franckcondon([n_k; num_idxs_left[j]], Ul, S, Ur, num_idxs_right[i])
        end
    end
    return m
end

function _MPSblock_end(::Type{T}, n_k, num_idxs_right, U_right) where {T<:Number}
    # Dedicated function for the last site (we can't use _MPSblock) with "empty" `S_left`
    # and `num_idxs_left` otherwise we'd get a 0xN matrix since `length(num_idxs_left) == 0`
    m = Matrix{T}(undef, length(num_idxs_right), 1)
    Ul, S, Ur = euler(U_right)
    for i in axes(m, 1)
        m[i, 1] = franckcondon([n_k], Ul, S, Ur, num_idxs_right[i])
    end
    return m
end

function _MPSblock(n_k, num_idxs_left, U_left, num_idxs_right, U_right)
    _MPSblock(ComplexF64, n_k, num_idxs_left, U_left, num_idxs_right, U_right)
end

function _MPSblock_end(n_k, num_idxs_right, U_right)
    _MPSblock_end(ComplexF64, n_k, num_idxs_right, U_right)
end

function _fixed_width_rep(x, nd)
    str = string(round(x; digits=nd))
    len = length(str)
    nspaces = 4
    # 0.###### (nd digits after the zero) plus four spaces
    return string(str, repeat(" ", nspaces + 2 + nd - len))
end

function _inspect_normal_mode_decomposition(evals, num_idxs, bond_idx, N, maxdim)
    nd = 8
    header_str =
        "Decomposition of reduced state on modes $(bond_idx+1) ... $N" *
        "\nFirst $maxdim eigenvalues and corresponding number states:\n"
    table = [
        string(_fixed_width_rep(l[1], nd), "\t", l[2]) for
        l in first(zip(evals, num_idxs), maxdim)
    ]
    sum_evals_str = string("\nSum of discarded eigenvalues: ", 1 - sum(evals))
    return header_str * join(table, "\n") * sum_evals_str
end

function mps_matrices(g::GaussianState, maxdim, maxnumber; nvals=nmodes(g)^2, kwargs...)
    T = complex(eltype(g))

    purity_atol = get(kwargs, :purity_atol, 1e-4)
    g_pur = purity(g)
    if abs(g_pur - 1) > purity_atol
        error("input Gaussian state is not pure: its purity is $g_pur")
    end

    N = nmodes(g)
    num_idxs = Vector{Vector{Vector{Int}}}(undef, N)
    S = Vector{Matrix{Float64}}(undef, N)
    nm_evals, num_idxs[1], S[1] = normal_mode_decomposition(g, N, maxnumber; kwargs...)
    # We don't really need the eigenvalues for the MPS construction, they are needed
    # just to find out the most relevant number states (within the
    # `normal_mode_decomposition` function), so we don't bother saving all of them in an
    # array.
    @debug _inspect_normal_mode_decomposition(nm_evals, num_idxs[1], 0, N, maxdim)
    # This is the normal-mode decomposition of a pure state, so we should find only one
    # eigenvalue, 1, corresponding to the vacuum.
    # It miiiight not be the case if the covariance matrix comes from the convex
    # optimisation routine, where the precision is well below the default settings of
    # `isapprox`; we need to make this function work in that case anyway.
    # Old approach: the following assertions.
    #
    #   @assert length(nm_evals_right) == length(num_idxs_right) == 1
    #   @assert nm_evals_right[1] ≈ 1
    #   @assert all(num_idxs_right[1] .== 0)
    #
    # A likely result of the optimisation routine:
    #
    #  │ 0.99990716    	[0, 0, 0, 0]
    #  │ 7.703e-5      	[1, 0, 0, 0]
    #  │ 1.58e-5       	[0, 1, 0, 0]
    #  │ 1.0e-8        	[2, 0, 0, 0]
    #  │ Sum of discarded eigenvalues: 1.4674150783378082e-9
    #
    # We probably should just check that the largest eigenvalue is sufficiently within 1 and
    # that its corresponding number state is the vacuum, then discard everything else.
    largest_mode1_eval = argmax(first, zip(nm_evals, num_idxs[1]))
    if abs(first(largest_mode1_eval) - 1) > purity_atol
        "state is not 1, but $(first(largest_mode1_eval))"
        throw(error(errmsg))
    end
    if !iszero(last(largest_mode1_eval))
        errmsg="the number state associated to the largest eigenvalue in the normal-mode " *
               "decomposition of the whole state is not the vacuum state"
        throw(error(errmsg))
    end
    num_idxs[1] = [last(largest_mode1_eval)]

    # Precompute all normal-mode decompositions
    Threads.@threads for bond_idx in 1:(N - 1)
        gpart = partialtrace(g, 1:bond_idx)  # Trace away modes from 1 to `bond_idx`
        nm_evals, num_idxs[bond_idx + 1], S[bond_idx + 1] = normal_mode_decomposition(
            gpart, nvals, maxnumber; kwargs...
        )
        @debug _inspect_normal_mode_decomposition(
            nm_evals, num_idxs[bond_idx + 1], bond_idx, N, maxdim
        )
    end

    A = Vector{Array{T}}(undef, N)  # array of MPS matrices

    pbar = Progress(N; desc="Computing MPS matrices...")
    Threads.@threads for bond_idx in 1:N
        if bond_idx < N
            num_idxs_left = first(num_idxs[bond_idx + 1], maxdim)
            num_idxs_right = first(num_idxs[bond_idx], maxdim)
            S_left = S[bond_idx + 1]
            S_right = S[bond_idx]

            # At step k we have the decompositions of [1 ... k] as "right" and
            # [k+1 ... N] as "left".
            t = Array{T}(
                undef, maxnumber + 1, length(num_idxs_right), length(num_idxs_left)
            )
            for n_k in 0:maxnumber
                t[n_k + 1, :, :] .= _MPSblock(
                    n_k, num_idxs_left, S_left, num_idxs_right, S_right
                )
            end
            A[bond_idx] = t
        else  # bond_idx == N
            t = Array{T}(undef, maxnumber + 1, length(num_idxs[end]), 1)
            for n_k in 0:maxnumber
                t[n_k + 1, :, 1] .= _MPSblock_end(T, n_k, num_idxs[end], S[end])
            end
            A[bond_idx] = t
        end
        next!(pbar)
    end

    return A
end

"""
    MPS(g::GaussianState; maxdim, maxnumber, kwargs...)

Build an MPS representation of the Gaussian state `g` with bond dimension up to `maxdim`,
truncating the Fock space of each mode at the `maxnumber`-particle sector.
"""
function ITensorMPS.MPS(g::GaussianState; maxdim, maxnumber, kwargs...)
    N = nmodes(g)
    blocks = mps_matrices(g, maxdim, maxnumber; kwargs...)

    @assert getindex.(size.(blocks), 2)[2:end] == getindex.(size.(blocks), 3)[1:(end - 1)]
    linds_dim = getindex.(size.(blocks), 2)[2:end]

    sites = siteinds("Boson", N; dim=maxnumber + 1)
    v = Vector{ITensor}(undef, N)
    # MPS structure:
    #
    #                s[j]                     s[j+1]
    #                  │                         │
    #                  │                         │
    #                ┌───┐    l[j]             ┌───┐  l[j+1]
    #   ╶╶╶╶─────────│ j │────────── ──────────│j+1│─────────╶╶╶
    #   dag(l[j-1])  └───┘           dag(l[j]) └───┘
    #
    # (Actually the `dag` operation here doesn't really do anything: when called on an
    # Index, if they carry arrows it will reverse these arrows. But our indices do not
    # carry any arrow, it's something that appears only when dealing with quantum numbers,
    # so it's a no-op in our case. We leave the `dag` there anyway.)
    l = [Index(linds_dim[i], "Link,l=$i") for i in 1:(N - 1)]
    v[1] = ITensor(blocks[1], sites[1], l[1])
    for i in 2:(N - 1)
        v[i] = ITensor(blocks[i], sites[i], dag(l[i - 1]), l[i])
    end
    v[N] = ITensor(blocks[N], sites[N], dag(l[N - 1]))
    return MPS(v)
end

"""
    enlargelocaldim(v::MPS, newdim)

Return a new MPS with the local dimension increased to `newdim`.

Note that the new MPS will be defined on a new set of site indices, so it will be
incompatible with the original one.
"""
function enlargelocaldim(v::MPS, newdim)
    # Strategy: let `A[k, iₖ]` be the matrices from `v`. We build a new MPS with matrices
    # `B[k, iₖ]` with the same bond dimensions, but bigger physical dimension (we'll need to
    # create a different set of site indices), then set `B[k, iₖ]` equal to `A[k, iₖ]` if
    # `iₖ ≤ d`, otherwise to zero.
    if !allequal(dim(siteind(v, k)) for k in eachindex(v))
        error("enlargelocaldim with non-homogeneous local dimension is not supported.")
    end

    currentdim = dim(siteind(v, 1))
    n = length(v)
    sind = siteinds("Boson", n; dim=newdim)  # enlarged physical sites
    v_enlarged = MPS(ITensors.scalartype(v), sind; linkdims=linkdims(v))
    lind = linkinds(v_enlarged)  # link indices or the new MPS

    v_enlarged[1][sind[1] => 1:currentdim, lind[1] => :] = array(
        v[1], siteind(v, 1), linkind(v, 1)
    )
    for k in 2:(n - 1)
        v_enlarged[k][sind[k] => 1:currentdim, lind[k - 1] => :, lind[k] => :] = array(
            v[k], siteind(v, k), linkind(v, k-1), linkind(v, k)
        )
    end
    v_enlarged[n][sind[n] => 1:currentdim, lind[n - 1] => :] = array(
        v[n], siteind(v, n), linkind(v, n-1)
    )

    return v_enlarged
end
