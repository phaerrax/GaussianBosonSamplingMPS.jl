using Documenter, DocumenterCitations
using ITensors, ITensorMPS, GaussianStates
using GaussianBosonSamplingMPS

push!(LOAD_PATH, "../src/")
# This line is needed because the GaussianBosonSamplingMPS is not accessible through Julia's
# LOAD_PATH.

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

makedocs(;
    modules=[GaussianBosonSamplingMPS],
    sitename="GaussianBosonSamplingMPS",
    repo=Remotes.GitHub("phaerrax", "GaussianBosonSamplingMPS.jl"),
    checkdocs=:exported,
    authors="Davide Ferracin",
    pages=[
        "Home" => "index.md",
        "Reference" => "reference.md",
        "Mathematical background" => "gbs_simulation_algorithm.md",
        "Tutorial" => "examples/borealis.md",
    ],
    plugins=[bib],
    format=Documenter.HTML(;
        mathengine=Documenter.MathJax3(
            Dict(
                :tex => Dict(
                    :inlineMath => [["\$", "\$"], [raw"\(", raw"\)"]],
                    :tags => "ams",
                    :packages => ["base", "ams", "autoload", "configmacros"],
                    :macros => Dict(
                        :C => [raw"\mathbb{C}"],
                        :N => [raw"\mathbb{N}"],
                        :Num => [raw"\mathscr{N}"],
                        :Og => [raw"\mathrm{O}(#1)", 1],
                        :R => [raw"\mathbb{R}"],
                        :Sp => [raw"\mathrm{Sp}(#1)", 1],
                        :abs => [raw"\lvert #1 \rvert", 1],
                        :adj => [raw"#1^\dagger", 1],
                        :bra => [raw"\langle #1 \rvert", 1],
                        :conj => [raw"\bar{#1}", 1],
                        :covmat => [raw"\sigma"],
                        :dd => [raw"\mathrm{d}"],
                        :defeq => [raw"\mathrel{\mathop:}="],
                        :det => [raw"\operatorname{det}"],
                        :diag => [raw"\operatorname{diag}"],
                        :displacement => [raw"\mathscr{D}_{#1}", 1],
                        :fcomm => [raw"\{#1\}", 1],
                        :fmom => [raw"r"],
                        :fockb => [raw"\mathscr{F}"],
                        :gauss => [raw"\gamma_{#1}", 1],
                        :id => [raw"1"],
                        :imag => [raw"\operatorname{Im}"],
                        :imat => [raw"I_{#1}", 1, ""],
                        :innp => [raw"\langle #1, #2\rangle", 2],
                        :iso => [raw"\cong"],
                        :iu => [raw"\mathrm{i}"],
                        :ket => [raw"\lvert #1 \rangle", 1],
                        :lseq => [raw"\ell_2"],
                        :modepart => [raw"^{[#1]}", 1],
                        :multi => [raw"\mathbf{#1}", 1],
                        :norm => [raw"\lVert #1 \rVert", 1],
                        :ns => [raw"f_{#1}", 1],
                        :opt => [raw"_{\mathrm{opt}}"],
                        :outp => [raw"\lvert #1\rangle \langle #2\rvert", 2],
                        :pure => [raw"_{\mathrm{p}}"],
                        :real => [raw"\operatorname{Re}"],
                        :sb => [raw"_{#1}", 1],
                        :spunitary => [raw"\spunitarysymbol(#1)", 1],
                        :spunitarymp => [raw"\spunitarysymbol^{[#1]}(#2)", 2, ""],
                        :spunitarysymbol => [raw"\mathscr{U}"],
                        :sympmat => [raw"\varOmega"],
                        :tr => [raw"\operatorname{tr}"],
                        :transpose => [raw"#1^{\mathrm{T}}", 1],
                        :tsp => [raw"^{\otimes #1}", 1],
                        :vacuum => [raw"\varOmega"],
                        :xpvec => ["R"],
                    ),
                ),
            ),
        ),
    ),
)

# Automatically deploy documentation to gh-pages.
deploydocs(; repo="github.com/phaerrax/GaussianBosonSamplingMPS.jl.git")
