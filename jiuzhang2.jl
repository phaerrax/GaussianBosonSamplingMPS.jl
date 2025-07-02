using GaussianStates, GaussianBosonSamplingMPS, LinearAlgebra
using ITensorMPS
using DelimitedFiles, HDF5, ArgParse

function parsecommandline()
    s = ArgParseSettings()
    add_arg_table!(
        s,
        ["--squeeze_parameters", "-r"],
        Dict(
            :help => "path to list of squeezing parameters (CSV file)",
            :arg_type => String,
            :required => true,
        ),
        ["--transfer_matrix", "-t"],
        Dict(
            :help => "path to transfer matrix (HDF5 file)",
            :arg_type => String,
            :required => true,
        ),
        ["--maxdim", "-b"],
        Dict(:help => "maximum bond dimension", :arg_type => Int, :required => true),
        ["--maxnumber_mps", "-n"],
        Dict(
            :help => "maximum number of photons in each mode (MPS construction)",
            :arg_type => Int,
            :required => true,
        ),
        ["--maxnumber_displacement", "-d"],
        Dict(
            :help => "maximum number of photons in each mode (before displacement)",
            :arg_type => Int,
            :required => true,
        ),
        ["--nsamples", "-s"],
        Dict(:help => "number of samples", :arg_type => Int, :required => true),
        ["--nsamples_per_displacement", "-i"],
        Dict(
            :help => "number of samples for each random displacement (default: √nsamples)",
            :arg_type => Int,
        ),
        ["--scs_eps"],
        Dict(:help => "SCS working precision", :arg_type => Float64),
        ["--output", "-o"],
        Dict(:help => "basename to output file", :arg_type => String, :required => true),
        ["--verbose", "-v"],
        Dict(:help => "print additional information", :action => :store_true),
    )

    args = Dict()
    for (k, v) in parse_args(s)
        isnothing(v) || push!(args, Symbol(k) => v)
    end

    return args
end

function main()
    args = parsecommandline()

    sqparfile = args[:squeeze_parameters]
    # Should contain 25 real numbers, that define the two-mode squeezing between 25 pairs
    # of modes.
    @info "Reading squeeze parameters from $sqparfile"
    r = vec(readdlm(sqparfile))

    g0 = vacuumstate(2length(r))
    for i in 1:length(r)
        squeeze2!(g0, r[i], 2i-1, 2i)
    end

    Tmatfile = args[:transfer_matrix]
    # A complex matrix which transforms 50 input modes into 144 output modes.
    @info "Reading transfer matrix from $Tmatfile"
    T = h5open(Tmatfile) do f
        read(f, "transfer_matrix")
    end
    T = transpose(T)

    # From fhe readme file: «The rows of the matrix are arranged according to {1H, 1V, 2H,
    # 2V, ..., 25H, 25V}. It needs to be rearranged to {1H, 2H, ..., 25H, 1V, 2V, ..., 25V}
    # when calculating the output probability.»
    # The matrix though is not square so `GaussianStates.permute_to_xpxp` cannot be used
    # (it assumes a square matrix, otherwise it silently truncates it).
    ϕT_xxpp = [
        real(T) -imag(T)
        imag(T) real(T)
    ]
    nrows, ncols = size(ϕT_xxpp)
    ϕT = ϕT_xxpp[invperm([1:2:nrows; 2:2:nrows]), invperm([1:2:ncols; 2:2:ncols])]

    σ = I - ϕT * ϕT' + ϕT * g0.covariance_matrix * ϕT'
    g = GaussianState(Symmetric(σ))

    scs_eps = get(args, :scs_eps, 1e-4)  # default value used by SCS
    @info "Optimising final state (with scs_eps = $scs_eps)"
    g_opt, W = optimise(g; :verbose => args[:verbose], scs_eps=scs_eps)

    @info "Computing MPS of final state"
    v = MPS(
        g_opt; maxdim=args[:maxdim], maxnumber=args[:maxnumber_mps], purity_atol=10*scs_eps
    )

    v = enlargelocaldim(v, args[:maxnumber_displacement]+1)

    outputfile = args[:output] * ".h5"
    @info "Writing final MPS on $outputfile"
    h5open(outputfile, "w") do hf
        write(hf, "squeeze_parameters", r)
        write(hf, "transfer_matrix", T)
        write(hf, "final_state", v)
    end

    samples, displacements = sample_displaced(
        v,
        W;
        :nsamples => args[:nsamples],
        :nsamples_per_displacement =>
            get(args, :nsamples_per_displacement, isqrt(args[:nsamples])),
        :eval_atol => 10*scs_eps,
    )

    @info "Writing samples on $outputfile"
    h5open(outputfile, "cw") do hf
        write(hf, "displacements", displacements)
        write(hf, "samples", samples)
    end

    return nothing
end

main()
