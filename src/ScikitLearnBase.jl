module ScikitLearnBase

using MacroTools: @capture   # TODO: get rid of this dependency

# These are the functions that should be implemented by estimators/transformers
api = [:fit!, :transform, :fit_transform!,
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

export @simple_estimator_constructor, BaseEstimator

abstract BaseEstimator


# I don't think this is necessary anymore.
macro import_api()
    # I wish `importall ..` worked
    esc(:(begin $([Expr(:import, :., :., f) for f in api]...) end))
end

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
    @simple_estimator_constructor function SomeEstimator(; param_1=..., param_2=..., ...)
        ...
        SomeEstimator(param_1, param_2, ...)
    end

This macro helps to implement the scikit-learn protocol for simple estimators
(those that do not contain other estimators). It is used in front of the
definition of an outer constructor. In addition to defining the constructor
normally, it also defines the `set_params!`, `get_params` and `clone` methods
for this estimator.

The arguments accepted by the constructor should all be keyword arguments with
a default value. In other words, a user should be able to instantiate an
estimator without passing any arguments to it. The arguments should all
correspond to hyperparameters describing the model or the optimisation problem
the estimator tries to solve. These initial arguments (or parameters) are
always remembered by the estimator. Also note that they should not be
documented under the “Attributes” section, but rather under the “Parameters”
section for that estimator.

In addition, every keyword argument accepted by the constructor should
correspond to a field on the instance. The macro relies on this to find
the relevant fields to set on an estimator when doing model selection.

It is used in front of an outer constructor. The macro assumes that this
constructor's keyword arguments are all the estimator's hyperparameters (all
parameters set by the user, that do not depend on the training data). It will
define the `get_params`, `set_params!` and `clone` methods accordingly.

There should be no logic, not even input validation, and the parameters should
not be changed. The corresponding logic should be put where the parameters are
used, typically in fit! """
macro simple_estimator_constructor(function_definition)
    type_name, args, kwargs, body =
        parse_function_definition(function_definition)
    param_names = Symbol[parse_kwarg(kw)[1] for kw in kwargs]
    @assert isempty(args) "A @model_init function should accept only keyword arguments representing the model's hyper-parameters"
    :(begin
        ScikitLearnBase.get_params(estimator::$(esc(type_name)); deep=true) =
            simple_get_params(estimator, $param_names)
        ScikitLearnBase.set_params!(estimator::$(esc(type_name)); params...) =
            simple_set_params!(estimator, params; param_names=$param_names)
        ScikitLearnBase.clone(estimator::$(esc(type_name))) =
            simple_clone(estimator)
        $(esc(:(function $type_name(; $(kwargs...))
            $(body...)
        end)))
    end)
end


################################################################################
# Defaults

fit_transform!(estimator::BaseEstimator, X, y=nothing; fit_kwargs...) =
    transform(fit!(estimator, X, y; fit_kwargs...), X)

function set_params!(estimator::BaseEstimator; params...) # from base.py
    # Simple optimisation to gain speed (inspect is slow)
    if isempty(params) return estimator end

    valid_params = get_params(estimator, deep=true)
    for (key, value) in params
        sp = split(string(key), "__"; limit=2)
        if length(sp) > 1
            name, sub_name = sp
            if !haskey(valid_params, name::AbstractString)
                throw(ArgumentError("Invalid parameter $name for estimator $estimator"))
            end
            sub_object = valid_params[name]
            set_params!(sub_object; Dict(Symbol(sub_name)=>value)...)
        else
            TODO() # should be straight-forward
        end
    end
    estimator
end


################################################################################

""" `parse_function_definition(fdef)`

Macro helper: parses the given function definition and returns
`(fname::Symbol, args::Vector{Any}, kwargs::Vector{Any}, body::Vector{Any})`

One can rebuild the function definition with
`esc(:(function &fname (&(args...); &(kwargs...)) &(body...) end))`

(replace the & with dollar sign - the docstring system didn't allow us to write
it correctly) """
function parse_function_definition(fdef)
    if @capture(fdef, function fname_(args__; kwargs__) body__ end)
        (fname, args, kwargs, body)
    elseif @capture(fdef, function fname_(args__) body__ end)
        (fname, args, Any[], body)
    elseif @capture(fdef, (fname_(args__; kwargs__) = bexpr_))
        (fname, args, kwargs, Any[bexpr])
    elseif @capture(fdef, fname_(args__) = bexpr_)
        (fname, args, Any[], Any[bexpr])
    else
        error("Not a function definition: $fdef")
    end
end


""" `macro_keyword_args(kwarg)`

Macro helper: if a function/macro definition contains x=y in an argument list,
this will return (x, y) """ # TODO: add example
function parse_kwarg(kwarg)
    @assert kwarg.head == :kw
    return (kwarg.args[1], kwarg.args[2])
end

end
