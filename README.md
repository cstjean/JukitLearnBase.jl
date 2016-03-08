ScikitLearnBase.jl
------------

This package exposes the interface to ScikitLearn.jl. If you are interested in
implementing the ScikitLearn.jl interface for your machine learning algorithm,
read on. If you just want to fit a model to some data, check out
[ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl)

Overview
-----

In contrast with [scikit-learn](scikit-learn.org), the ScikitLearn.jl library
does not contain any model (beyond a few special cases). It's up to Julia
library writers to implement the scikit-learn API. The API is defined
[here](docs/API.md). If your library is registered in METADATA, please let us
know by [filing an issue](https://github.com/cstjean/ScikitLearn.jl/issues). We
can add it to the [list of
models](http://scikitlearnjl.readthedocs.org/en/latest/models/).