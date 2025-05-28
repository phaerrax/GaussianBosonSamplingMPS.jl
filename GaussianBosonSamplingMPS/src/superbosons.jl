using MPSTimeEvolution: LocalOperator, domain
using ITensors: OneITensor

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

function sb_siteinds(; nmodes, maxnumber)
    sites_phy = siteinds("Boson", nmodes; dim=maxnumber+1, addtags="phy")
    sites_anc = siteinds("Boson", nmodes; dim=maxnumber+1, addtags="anc")
    return collect(Iterators.flatten(zip(sites_phy, sites_anc)))
end
