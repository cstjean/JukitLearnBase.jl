VERSION >= v"0.4.0" && __precompile__()

module ScikitLearnBase

macro declare_api(api_functions...)
    # GaussianMixtures supports Julia 0.3, so we need to account for it here.
    # ScikitLearn.jl won't support it, so it's mostly a matter of not
    # triggering any error.
    esc(:(begin
        $([VERSION >= v"0.4.0" ? Expr(:function, f) : :(function $f() end)
           for f in api_functions]...)
        # Expr(:export, f) necessary in Julia 0.3
        $([Expr(:export, f) for f in api_functions]...)
        const api = [$([Expr(:quote, x) for x in api_functions]...)]
    end))
end
# These are the functions that can be implemented by estimators/transformers.
# See http://scikitlearnjl.readthedocs.org/en/latest/api/
@declare_api(fit!, partial_fit!, transform, fit_transform!, fit_predict!,
             predict, predict_proba, predict_log_proba,
             score_samples, sample,
             score, decision_function, clone, set_params!,
             get_params, is_classifier, is_pairwise,
             get_feature_names, get_classes, get_components,
             inverse_transform)

export BaseEstimator, BaseClassifier, BaseRegressor, declare_hyperparameters

# Ideally, all scikit-learn estimators would inherit from BaseEstimator, but
# it's hard to ask library writers to do that given single-inheritance, so the
# API doesn't rely on it.
abstract BaseEstimator
abstract BaseClassifier <: BaseEstimator
abstract BaseRegressor <: BaseEstimator

is_classifier(::BaseClassifier) = true
is_classifier(::BaseRegressor) = false

# This hasn't been used so far, but it seems like it should be useful at some
# point, and it doesn't cost much.
implements_scikitlearn_api(estimator) = false   # global default
implements_scikitlearn_api(estimator::BaseEstimator) = true

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
    function declare_hyperparameters{T}(estimator_type::Type{T}, params::Vector{Symbol})

This function helps to implement the scikit-learn protocol for simple
estimators (those that do not contain other estimators). It will define
`set_params!`, `get_params` and `clone` for `::estimator_type`.
It is called at the top-level. Example:

    declare_hyperparameters(GaussianProcess, [:regularization_strength])

Each parameter should be a field of `estimator_type`.

Most models should call this function. The only exception are models that
contain other models. They should implement `get_params` and `set_params!`
manually. """
function declare_hyperparameters{T}(estimator_type::Type{T},
                                    params::Vector{Symbol})
    @eval begin
        ScikitLearnBase.get_params(estimator::$(estimator_type); deep=true) =
            simple_get_params(estimator, $params)
        ScikitLearnBase.set_params!(estimator::$(estimator_type);
                                    new_params...) =
            simple_set_params!(estimator, new_params; param_names=$params)
        ScikitLearnBase.clone(estimator::$(estimator_type)) =
            simple_clone(estimator)
    end
end

################################################################################
# Standard scoring functions (those are good defaults)

# Helper
function weighted_sum(sample_score, sample_weight; normalize=false)
    if sample_weight === nothing
        return normalize ? mean(sample_score) : sum(sample_score)
    else
        s = dot(sample_score, sample_weight)
        return normalize ? (s / sum(sample_weight)) : s
    end
end

# scikit-learn's version is fancier, but I would rather KISS for now
function classifier_accuracy_score(y_true::Vector, y_pred::Vector;
                                   normalize=true, sample_weight=nothing)
    weighted_sum(y_true.==y_pred, sample_weight, normalize=normalize)
end

function mean_squared_error(y_true::Vector, y_pred::Vector;
                            sample_weight=nothing)
    weighted_sum((y_true - y_pred) .^ 2, sample_weight; normalize=true)
end
mse_score(y_true, y_pred; sample_weight=nothing) =
    -mean_squared_error(y_true, y_pred; sample_weight=sample_weight)

score(clf::BaseClassifier, X, y_true; sample_weight=nothing) =
    classifier_accuracy_score(y_true, predict(clf, X);
                              sample_weight=sample_weight)
score(reg::BaseRegressor, X, y_true; sample_weight=nothing) =
    mse_score(y_true, predict(reg, X); sample_weight=sample_weight)


################################################################################
# Defaults

fit_transform!(estimator::BaseEstimator, X, y=nothing; fit_kwargs...) =
    transform(fit!(estimator, X, y; fit_kwargs...), X)
fit_predict!(estimator::BaseEstimator, X, y=nothing; fit_kwargs...) =
    predict(fit!(estimator, X, y; fit_kwargs...), X)

end
