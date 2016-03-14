module ScikitLearnBase

# These are the functions that can be implemented by estimators/transformers
api = [:fit!, :partial_fit!, :transform, :fit_transform!,
       :predict, :predict_proba, :predict_log_proba,
       :score_samples, :sample,
       :score, :decision_function, :clone, :set_params!, :get_params,
       :is_classifier, :is_pairwise, :get_feature_names, :get_classes,
       :inverse_transform]

for f in api
    eval(Expr(:function, f))  # forward declaration
    # Export all the API. Not sure if we should do that...
    @eval(export $f)
end

export BaseEstimator, declare_hyperparameters

abstract BaseEstimator


################################################################################
# These functions are useful for defining estimators that do not themselves
# contain other estimators

function simple_get_params(estimator, param_names::Vector{Symbol})
    Dict([name => getfield(estimator, name)
          for name in param_names])
end

function simple_set_params!{T}(estimator::T, params; param_names=nothing)
    for (k, v) in params
        if param_names !== nothing && !(k in param_names)
            throw(ArgumentError("An estimator of type $T was passed the invalid hyper-parameter $k. Valid hyper-parameters: $param_names"))
        end
        setfield!(estimator, k, v)
    end
    estimator
end

simple_clone{T}(estimator::T) = T(; get_params(estimator)...)

"""
    function declare_hyperparameters{T}(model_type::Type{T}, params::Vector{Symbol};
                                        define_fit_transform=true)

This function helps to implement the scikit-learn protocol for simple
estimators (those that do not contain other estimators). It will define
`set_params!`, `get_params`, `clone` and `fit_transform!` for `::model_type`.
It is called at the top-level. Example:

    declare_hyperparameters(GaussianProcess, [:regularization_strength])

Each parameter should be a field of `model_type`.

Most models should call this function. The only exception are models that
contain other models. They should implement `get_params` and `set_params!`
manually. """
function declare_hyperparameters{T}(model_type::Type{T}, params::Vector{Symbol};
                                    define_fit_transform=true)
    @eval begin
        ScikitLearnBase.get_params(estimator::$(model_type); deep=true) =
            simple_get_params(estimator, $params)
        ScikitLearnBase.set_params!(estimator::$(model_type);
                                    new_params...) =
            simple_set_params!(estimator, new_params; param_names=$params)
        ScikitLearnBase.clone(estimator::$(model_type)) =
            simple_clone(estimator)
    end
    if define_fit_transform
        @eval ScikitLearnBase.fit_transform!(estimator::$model_type, X, y=nothing; fit_kwargs...) = transform(fit!(estimator, X, y; fit_kwargs...), X)
    end
end


################################################################################
# Defaults

fit_transform!(estimator::BaseEstimator, X, y=nothing; fit_kwargs...) =
    transform(fit!(estimator, X, y; fit_kwargs...), X)


end
