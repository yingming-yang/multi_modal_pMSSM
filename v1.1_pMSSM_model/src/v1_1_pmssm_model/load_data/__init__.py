from .load_data import load_data

_cache = None
_VAR_NAMES = ['parameters', 'mo_omega', 'parameters_mean', 'parameters_std',
              'parameters_standardized', 'theory_param', 'theory_param_nonconst']

def __getattr__(name):
    if name in _VAR_NAMES:
        global _cache
        if _cache is None:
            _cache = load_data()
        return _cache[_VAR_NAMES.index(name)]
    raise AttributeError(f"module 'load_data' has no attribute {name!r}")
