ScikitLearnBase.jl
------------

This package exposes the scikit-learn interface. Libraries that implement this
interface can be used in conjunction with [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl) (pipelines, cross-validation, hyperparameter tuning, ...)

This is an intentionally slim package (~100 LOC, no dependencies). That way,
ML libraries can `import ScikitLearnBase` without dragging along all of
`ScikitLearn`'s dependencies.

Overview
-----

There's an overview of the API
[here](http://scikitlearnjl.readthedocs.org/en/latest/api/) and a more detailed
description [here](docs/API.md). For models with simple hyperparameters, it
boils down to this:

```julia
import ScikitLearnBase

type NaiveBayes
    # The model hyperparameters (not learned from data)
    bias::Float64

    # The parameters learned from data
    counts::Matrix{Int}
    
    # A constructor that accepts the hyperparameters as keyword arguments
    # with sensible defaults
    NaiveBayes(; bias=0.0f0) = new(bias)
end

# This will define `clone`, `set_params!` and `get_params` for the model
ScikitLearnBase.@declare_hyperparameters(NaiveBayes, [:bias])

# NaiveBayes is a classifier
ScikitLearnBase.is_classifier(::NaiveBayes) = true   # not required for transformers

function ScikitLearnBase.fit!(model::NaiveBayes, X, y)
    .... # modify model.counts here
    return model
end

function ScikitLearnBase.predict(model::NaiveBayes, X)
    .... # returns a vector of predicted classes here
end
```

You can try it out with `ScikitLearn.CrossValidation.cross_val_score`. Models
with more complex hyperparameter specifications should implement `clone`,
`get_params` and `set_params!` explicitly instead of using
`@declare_hyperparameters`.

The interface has been implemented for a few libraries, see PRs:
[GaussianMixtures.jl](https://github.com/davidavdav/GaussianMixtures.jl/pull/18), [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl/pull/17)

Notes:

- If the model performs unsupervised learning, implement `transform` instead of
`predict`.
- If the model is already coded up and the type does not contain
hyperparameters (eg. if it follows the StatsBase interface, or if you're
implementing the interface for someone else's library), you can create a new
type that contains the old one:

```julia
type SkNaiveBayes  # prefix the name with Sk
    bias::Float64

    nb::NaiveBayes
    NaiveBayes(; bias=0.0f0) = new(bias)
end
```

Once your library implements the API, let us know by [filing an
issue](https://github.com/cstjean/ScikitLearn.jl/issues). It will be added to
the [list of models](http://scikitlearnjl.readthedocs.io/en/latest/models/#julia-models).
