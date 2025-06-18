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
        ["--maxdim_pre", "-b"],
        Dict(
            :help => "maximum bond dimension (before displacement)",
            :arg_type => Int,
            :required => true,
        ),
        ["--maxdim_post", "-d"],
        Dict(
            :help => "maximum bond dimension (after displacement)",
            :arg_type => Int,
            :required => true,
        ),
        ["--maxnumber_pre", "-n"],
        Dict(
            :help => "maximum number of photons in each mode (before displacement)",
            :arg_type => Int,
            :required => true,
        ),
        ["--maxnumber_post", "-p"],
        Dict(
            :help => "maximum number of photons in each mode (after displacement)",
            :arg_type => Int,
            :required => true,
        ),
        ["--output", "-o"],
        Dict(:help => "basename to output file", :arg_type => String, :required => true),
        ["--verbose", "-v"],
        Dict(:help => "print additional information", :action => :store_true),
    )

    return Dict(Symbol(k) => v for (k, v) in parse_args(s))
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
    g_opt, W = optimise(g; verbose=args[:verbose])

    @info "Computing MPS of the final state"
    v = MPS(
        g_opt; maxdim=args[:maxdim_pre], maxnumber=args[:maxnumber_pre], purity_atol=1e-3
    )

    #TODO enlarge the MPS
    #TODO add sampling, maybe in another function
    
    outputfile = args[:output] * ".h5"
    @info "Writing final MPS on $outputfile"
    h5open(outputfile, "w") do hf
        write(hf, "final_state", v)
    end

    return nothing
end

generate_mps()
