using Armon
using Documenter

DocMeta.setdocmeta!(Armon, :DocTestSetup, :(using Armon); recursive=true)

makedocs(;
    modules=[Armon],
    authors="Luc Briand <luc.briand35@gmail.com> and contributors",
    repo=Remotes.GitHub("Keluaa", "Armon.jl"),
    sitename="Armon.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
