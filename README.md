ScikitLearnBase.jl
------------

This package exposes the scikit-learn interface. Libraries that implement this
interface can be used in conjunction with [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl) (pipelines, cross-validation, hyperparameter tuning, ...)

This is an intentionally slim package (~50 LOC, no dependencies).

Overview
-----

There's a detailed description of the API [here](docs/API.md). For most
algorithms, it boils down to these functions:

```julia
import ScikitLearnBase

type NaiveBayes
    # The model hyperparameters (not learned from data)
    bias::Float64

    # The parameters learned from data
    counts::Vector{Float64}
    
    # A constructor that accepts the hyperparameters as keyword arguments
    # with sensible defaults
    NaiveBayes(; bias=0.0f0) = new(bias)
end

# This will define `clone`, `set_params!` and `get_params` for you
declare_hyperparameters(NaiveBayes, [:bias])

# NaiveBayes is a classifier
is_classifier(::NaiveBayes) = true   # not required for transformers

function ScikitLearnBase.fit!(model::NaiveBayes, X, y)
    .... # modify model.counts here
end

function ScikitLearnBase.predict(model::NaiveBayes, X)
    .... # returns a vector of predicted classes here
end
```

You can now try it out with `ScikitLearn.CrossValidation.cross_val_score`

Notes:

- If your model performs unsupervised learning, implement `transform` instead of
`predict`.
- If your model is already coded up and the type does not contain
hyperparameters (eg. if it follows the StatsBase interface), you can create a
new type that contains the old one:

```julia
type SkNaiveBayes  # prefix the name with Sk
    bias::Float64

    nb::NaiveBayes
    NaiveBayes(; bias=0.0f0) = new(bias)
end
```

Once your library implements the API, let us know by [filing an
issue](https://github.com/cstjean/ScikitLearn.jl/issues). It will be added to
the [list of models](http://scikitlearnjl.readthedocs.org/en/latest/models/).
