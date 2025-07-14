using Documenter
using ITensors, ITensorMPS, GaussianStates
using GaussianBosonSamplingMPS

push!(LOAD_PATH, "../src/")
# This line is needed because the GaussianBosonSamplingMPS is not accessible through Julia's
# LOAD_PATH.

makedocs(;
    sitename="GaussianBosonSamplingMPS",
    modules=[GaussianBosonSamplingMPS],
    remotes=nothing,
    checkdocs=:exported,
    pages=["Home" => "index.md", "Reference" => "reference.md"],
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

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
