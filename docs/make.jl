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
        "Examples" => ["examples/borealis.md"],
    ],
    plugins=[bib],
    format=Documenter.HTML(;
        mathengine=Documenter.MathJax(
            Dict(
                :TeX => Dict(
                    :Macros => Dict(
                        :ket => [raw"\lvert #1 \rangle", 1],
                        :bra => [raw"\langle #1 \rvert", 1],
                        :transpose => [raw"#1^{\mathrm{T}}", 1],
                        :conj => [raw"\bar{#1}", 1],
                        :adj => [raw"#1^\dagger", 1],
                        :real => [raw"\operatorname{Re}"],
                        :imag => [raw"\operatorname{Im}"],
                        :N => [raw"\mathbb{N}"],
                        :C => [raw"\mathbb{C}"],
                        :R => [raw"\mathbb{R}"],
                        :dd => [raw"\mathrm{d}"],
                        :det => [raw"\operatorname{det}"],
                        :opt => [raw"_{\mathrm{opt}}"],
                        :tr => [raw"\operatorname{tr}"],
                    ),
                ),
            ),
        ),
    ),
)

# Automatically deploy documentation to gh-pages.
deploydocs(; repo="github.com/phaerrax/GaussianBosonSamplingMPS.jl.git")
