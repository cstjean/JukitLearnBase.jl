ScikitLearnBase.jl
------------

This package exposes the interface to ScikitLearn.jl. If you are interested in
integrating your machine learning package with the ScikitLearn.jl framework,
read on. If you just want to fit a model to some data, check out
[ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl)

---

You can inherit from BaseEstimator, but it only gives you a default
`fit_transform!` implementation (that calls `fit!`, then `transform`).