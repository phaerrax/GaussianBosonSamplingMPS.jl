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

function generate_mps()
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

    @info "Optimising final state"
    optimise_kwargs = Dict{Symbol,Any}(:verbose => args[:verbose])
    haskey(args, :scs_eps) && push!(optimise_kwargs, :scs_eps => args[:scs_eps])
    g_opt, W = optimise(g; optimise_kwargs...)

    @info "Computing MPS of final state"
    v = MPS(g_opt; maxdim=args[:maxdim], maxnumber=args[:maxnumber_mps], purity_atol=1e-3)

    v = enlargelocaldim(v, args[:maxnumber_displacement]+1)

    outputfile = args[:output] * ".h5"
    @info "Writing final MPS on $outputfile"
    h5open(outputfile, "w") do hf
        write(hf, "squeeze_parameters", r)
        write(hf, "transfer_matrix", T)
        write(hf, "final_state", v)
    end

    # TODO sampling...

    return nothing
end
