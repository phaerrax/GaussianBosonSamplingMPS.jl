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
