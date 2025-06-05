using MPSTimeEvolution: LocalOperator, domain
using ITensors: OneITensor
using Random

sb_index(n) = 2n - 1  # mode number --> MPS site
inv_sb_index(n) = div(n + 1, 2)  # MPS site --> mode number
# (inv_sb_index ∘ sb_index).(1:N) == 1:N
# (sb_index ∘ inv_sb_index).(1:N) != 1:N

function _id_pairs(v)
    @assert iseven(length(v))
    nmodes = div(length(v), 2)

    maxn = dim(siteind(v, 1)) - 1
    return [
        sum(
            state(siteind(v, n), string(m)) * state(siteind(v, n + 1), string(m)) for
            m in 0:maxn
        ) for n in sb_index.(1:nmodes)
    ]
end

function _id_contractions(v)
    @assert iseven(length(v))
    nmodes = div(length(v), 2)
    sb_id_blocks = _id_pairs(v)
    return [
        dag(sb_id_blocks[inv_sb_index(n)]) * v[n] * v[n + 1] for n in sb_index.(1:nmodes)
    ]
end

adj(x) = swapprime(dag(x), 0 => 1)

function measure(v::AbstractMPS, l::LocalOperator)
    @assert iseven(length(v))
    nmodes = div(length(v), 2)
    sb_id_blocks = _id_pairs(v)
    ids = _id_contractions(v)

    x = OneITensor()
    for n in sb_index.(1:nmodes)
        if n in domain(l)
            lop = if n + 1 in domain(l)
                # We loop over odd sites only, so we check manually that the next site
                # is in the domain of the operator.
                op(l[n], siteind(v, n)) * op(l[n + 1], siteind(v, n + 1))
            else
                op(l[n], siteind(v, n))
            end
            x *= dag(apply(adj(lop), sb_id_blocks[inv_sb_index(n)])) * v[n] * v[n + 1]
        else
            x *= ids[inv_sb_index(n)]
        end
    end
    return scalar(x)
end

function measure(v::AbstractMPS, ls::Vector{LocalOperator})
    @assert iseven(length(v))
    nmodes = div(length(v), 2)

    sb_id_blocks = _id_pairs(v)
    ids = _id_contractions(v)

    results = Vector{ComplexF64}(undef, length(ls))
    for (j, l) in enumerate(ls)
        x = OneITensor()
        for n in sb_index.(1:nmodes)
            if n in domain(l)
                lop = if n + 1 in domain(l)
                    # We loop over odd sites only, so we check manually that the next site
                    # is in the domain of the operator.
                    op(l[n], siteind(v, n)) * op(l[n + 1], siteind(v, n + 1))
                else
                    op(l[n], siteind(v, n))
                end
                x *= dag(apply(adj(lop), sb_id_blocks[inv_sb_index(n)])) * v[n] * v[n + 1]
            else
                x *= ids[inv_sb_index(n)]
            end
        end
        results[j] = scalar(x)
    end
    return results
end

function measure(v::AbstractMPS, ls::Matrix{LocalOperator})
    @assert iseven(length(v))
    nmodes = div(length(v), 2)

    sb_id_blocks = _id_pairs(v)
    ids = _id_contractions(v)

    results = Matrix{ComplexF64}(undef, size(ls))
    for (j, l) in enumerate(ls)
        x = OneITensor()
        for n in sb_index.(1:nmodes)
            if n in domain(l)
                lop = if n + 1 in domain(l)
                    # We loop over odd sites only, so we check manually that the next site
                    # is in the domain of the operator.
                    op(l[n], siteind(v, n)) * op(l[n + 1], siteind(v, n + 1))
                else
                    op(l[n], siteind(v, n))
                end
                x *= dag(apply(adj(lop), sb_id_blocks[inv_sb_index(n)])) * v[n] * v[n + 1]
            else
                x *= ids[inv_sb_index(n)]
            end
        end
        results[j] = scalar(x)
    end
    return results
end

function sb_trace(v::MPS)
    @assert iseven(length(v))
    return scalar(prod(_id_contractions(v)))
end

function sb_siteinds(; nmodes, maxnumber)
    sites_phy = siteinds("Boson", nmodes; dim=maxnumber+1, addtags="phy")
    sites_anc = siteinds("Boson", nmodes; dim=maxnumber+1, addtags="anc")
    return collect(Iterators.flatten(zip(sites_phy, sites_anc)))
end

"""
    sb_outer(v::AbstractMPS)

Compute the projection ``|v⟩⟨v| / ‖v‖²``, from the MPS `v` representing a pure state,
expressed as an MPS (of double the size) in the superboson formalism.
"""
function sb_outer(v)
    # 1) We build the MPO representing Pv = |v⟩⟨v| / ‖v‖² starting from the input MPS.
    #
    #    │   │   │   │   │   │       │   │   │   │   │   │
    #    ▒───▒───▒───▒───▒───▒  ──>  ▓───▓───▓───▓───▓───▓
    #                                │   │   │   │   │   │
    Pv = projector(v)
    n = length(v)
    maxnumber = dim(siteind(v, 1)) - 1

    # 2) We decompose each block of Pv in two pieces so that the physical indices end up in
    #    different blocks. In the middle blocks, the two link indices also get separated.
    #
    #     s₁'     s₁'                 sₖ'         sₖ'                  sₙ'       sₙ'
    #     │       │                   │           │                    │         │
    #     ▓─l₁  = ░              lₖ₋₁─▓─lₖ = lₖ₋₁─░               lₙ₋₁─▓  = lₙ₋₁─░
    #     │       │                   │           │                    │         │
    #     s₁      ░─l₁                sₖ          ░─lₖ                 sₙ        ░
    #             │                               │                              │
    #             s₁                              sₖ                             sₙ
    #
    T = ITensor[]

    s₁′ = siteinds(Pv, 1; plev=1)
    s₁ = siteinds(Pv, 1; plev=0)
    l₁ = linkind(Pv, 1)
    U, Σ, V = svd(Pv[1], (s₁′,); righttags="Link,r=1")
    # Let's call the newly created link indices "r=$k", so that we can distinguish them
    # later from the "original" link indices which are tagged as "l=$k".
    append!(T, [U*Σ, V])
    # We could choose to merge Σ with V as well instead of with U, it doesn't really matter
    # where we put the singular values, as long as we obtain two tensors as output.

    for k in 2:(n - 1)
        sₖ′ = siteinds(Pv, k; plev=1)
        sₖ = siteinds(Pv, k; plev=0)
        lₖ₋₁ = linkind(Pv, k-1)
        lₖ = linkind(Pv, k)
        U, Σ, V = svd(Pv[k], (sₖ′, lₖ₋₁); righttags="Link,r=$k")
        append!(T, [U*Σ, V])
    end

    sₙ′ = siteinds(Pv, n; plev=1)
    sₙ = siteinds(Pv, n; plev=0)
    lₙ₋₁ = linkind(Pv, n-1)
    U, Σ, V = svd(Pv[n], (sₙ′, lₙ₋₁); righttags="Link,r=$n")
    append!(T, [U*Σ, V])

    # 3) The resulting tensor network is still one-dimensional: we link everything together
    #    in a proper MPS.
    #
    #     s₁'   s₂'   s₃'                sₙ₋₁'  sₙ'
    #     │     │     │                   │     │
    #     ░  .·─░  .·─░  .·─           .·─░  .·─░       ────────╮
    #     │  ·  │  ·  │  ·     ···     ·  │  ·  │               │
    #     ░─·'  ░─·'  ░─·'           ─·'  ░─·'  ░               │
    #     │     │     │                   │     │               │
    #     s₁    s₂    s₃                 sₙ₋₁   sₙ              │
    #                                                           │
    #                                                           ▼
    #
    #                               s₁' s₁  s₂' s₂  s₃' s₃          sₙ' sₙ
    #                               │   │   │   │   │   │           │   │
    #                               ░───░───░───░───░───░─── ··· ───░───░
    vv = MPS(T)

    # The MPS we just constructed is not in a canonical form, let's fix that.
    orthogonalize!(vv, 1)

    # 4) We need to fix vv's indices, since now siteind(vv, k) == siteind(vv, k + 1)': they
    #    share the same ID and this will surely cause issues when contracting the MPS with
    #    something else. We create a new set of indices and replace them all.
    newsites = sb_siteinds(; nmodes=n, maxnumber=maxnumber)
    replace_siteinds!(vv, newsites)

    # Now the link index situation is something like
    #
    # [1]    (dim=##|id=###|"Link,r=1")
    # [2]    (dim=##|id=###|"Link,l=1")
    # [3]    (dim=##|id=###|"Link,r=2")
    # [4]    (dim=##|id=###|"Link,l=2")
    # [5]    (dim=##|id=###|"Link,r=3")
    # [6]    (dim=##|id=###|"Link,l=3")
    #        ...
    # [n-3]  (dim=##|id=###|"Link,r=n-1")
    # [n-2]  (dim=##|id=###|"Link,l=n-1")
    # [n-1]  (dim=##|id=###|"Link,r=n")
    #
    # while we want a we want a linear sequence like
    #
    # [1]   (dim=##|id=###|"Link,l=1")
    # [2]   (dim=##|id=###|"Link,l=2")
    # [3]   (dim=##|id=###|"Link,l=3")
    #       ...
    # [n-1] (dim=##|id=###|"Link,l=n")

    for k in eachindex(vv)[1:(end - 1)]
        if isodd(k)
            rk = div(k, 2)+1
            replacetags!(vv[k], "r=$rk", "m=$k")
            replacetags!(vv[k + 1], "r=$rk", "m=$k")
            # Unfortunately we already have "l=#" tags in the link indices so we first
            # change both "r=#" and "l=#" into "m=#", then later we'll change all of them
            # to "l=#".
        else
            lk = div(k, 2)
            replacetags!(vv[k], "l=$lk", "m=$k")
            replacetags!(vv[k + 1], "l=$lk", "m=$k")
        end
    end

    for k in eachindex(vv)[1:(end - 1)]
        replacetags!(vv[k], "m=$k", "l=$k")
        replacetags!(vv[k + 1], "m=$k", "l=$k")
    end

    return vv
end

"""
    sb_sample(m::MPS)

Given a "superbosonic" MPS `m` with unit trace, compute a `Vector{Int}` of `length(m)/2`
elements corresponding to one sample of the probability distribution defined by the
components of the density matrix that the MPS represents.
"""
function sb_sample(m::MPS)
    return sb_sample(Random.default_rng(), m)
end

"""
    randsample(values, weights)

Draw an element from `values`, such that `values[i]` has weight `weights[i]`.
"""
function randsample(values, weights)
    odds = cumsum(weights ./ sum(weights))
    return values[findfirst(odds .≥ rand())]
end

function sb_sample(rng::AbstractRNG, m::MPS)
    # See https://tensornetwork.org/mps/algorithms/sampling for an explanation of this MPS
    # sampling algorithm.
    @assert iseven(length(m))
    nmodes = div(length(m), 2)

    trace = sb_trace(m)
    if abs(1.0 - trace) > 1E-8
        error("sample: MPS is not normalized, trace=$trace")
    end

    ids = _id_contractions(m)
    contractions_from_right = Vector{Union{ITensor,OneITensor}}(undef, nmodes)
    # We want:
    #   contractions_from_right[end] = 1
    #   contractions_from_right[end-1] = ids[end]
    #   contractions_from_right[end-2] = contractions_from_right[end-1] * ids[end-1]
    #   contractions_from_right[end-3] = contractions_from_right[end-2] * ids[end-2]
    # and so on...
    # The first element contractions_from_right[1] will be the MPS `m` contracted on all
    # pairs of sites but the first one, i.e. ids[2] * ids[3] * ... * ids[end].
    contractions_from_right[end] = OneITensor()
    for j in reverse(1:(length(ids) - 1))
        contractions_from_right[j] = ids[j + 1] * contractions_from_right[j + 1]
    end

    # contractions_from_right[k] == prod(ids[j] * v[j] * v[j+1] for j in 2k+1:2:2nmodes)
    #
    # Basically contractions_from_right[k] is the reduced state after the partial trace
    # over all modes > k.

    result = Vector{Int}(undef, nmodes)
    prev = OneITensor()
    for j in 1:nmodes
        ρ = prev * m[sb_index(j)] * m[sb_index(j) + 1] * contractions_from_right[j]
        sl = siteind(m, sb_index(j))
        sr = siteind(m, sb_index(j)+1)
        # `ρ` is a matrix with indices `sl` and `sr`

        @assert dim(sl) == dim(sr)
        d = dim(sl)
        pn = [scalar(ρ * onehot(sl => k) * onehot(sr => k)) for k in 1:d]
        if !isapprox(real(pn), pn)
            throw(error("non-zero imaginary part in sampling probabilities"))
        end
        pn = real(pn)
        n = randsample(1:d, pn)
        result[j] = n

        if j < nmodes
            prev *=
                onehot(sl => n) * m[sb_index(j)] * m[sb_index(j) + 1] * onehot(sr => n) /
                pn[n]
        end
    end
    return result
end
