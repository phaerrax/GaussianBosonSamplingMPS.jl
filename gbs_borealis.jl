using GaussianStates, GaussianBosonSamplingMPS, LinearAlgebra
using ITensorMPS
using NPZ, HDF5, ArgParse

function parsecommandline()
    s = ArgParseSettings()
    add_arg_table!(
        s,
        ["--squeeze_parameters", "-r"],
        Dict(
            :help => "path to vector of squeezing parameters (NumPy vector)",
            :arg_type => String,
            :required => true,
        ),
        ["--transfer_matrix", "-t"],
        Dict(
            :help => "path to transfer matrix (NumPy matrix)",
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
    @info "Reading squeeze parameters from $sqparfile"
    r = npzread(sqparfile)

    Tmatfile = args[:transfer_matrix]
    @info "Reading transfer matrix from $Tmatfile"
    T = npzread(Tmatfile)

    if !(length(r) == size(T, 1) == size(T, 2))
        error("sizes of r and T do not match.")
    end

    N = length(r)

    g0 = vacuumstate(N)
    squeeze!(g0, r)

    ϕT = GaussianStates.permute_to_xpxp([
        real(T) -imag(T)
        imag(T) real(T)
    ])
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

    samples = sample_displaced(
        v,
        W;
        :nsamples => args[:nsamples],
        :nsamples_per_displacement =>
            get(args, :nsamples_per_displacement, isqrt(args[:nsamples])),
        :eval_atol => 10*scs_eps,
    )

    @info "Writing samples on $outputfile"
    h5open(outputfile, "cw") do hf
        write(hf, "samples", samples)
    end

    return nothing
end

main()
