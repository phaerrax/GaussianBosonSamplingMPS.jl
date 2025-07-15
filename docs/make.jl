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
    checkdocs=:exported,
    authors="Davide Ferracin",
    pages=["Home" => "index.md", "Reference" => "reference.md"],
    plugins=[bib],
    format=Documenter.HTML(;
        mathengine=Documenter.MathJax(
            Dict(
                :TeX => Dict(
                    :Macros => Dict(
                        :ket => [raw"\lvert #1 \rangle", 1],
                        :bra => [raw"\langle #1 \rvert", 1],
                        :adj => [raw"#1^\dagger", 1],
                    ),
                ),
            ),
        ),
    ),
)

# Automatically deploy documentation to gh-pages.
deploydocs(; repo="github.com/phaerrax/gaussian_boson_sampling.git")
