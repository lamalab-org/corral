"""Pickle-based state persistence between sandbox executions."""

RESTORE_TEMPLATE = """\
import pickle as __pkl__
from pathlib import Path as __Path__
__state_path__ = __Path__({state_file!r})
if __state_path__.exists():
    with __state_path__.open('rb') as __f__:
        __saved__ = __pkl__.load(__f__)
    locals().update(__saved__)
    del __saved__
del __pkl__, __Path__, __state_path__
try:
    del __f__
except NameError:
    pass
"""

SAVE_TEMPLATE = """\
import pickle as __pkl__
import types as __types__
__save__ = {{}}
__excluded__ = {{
    '__builtins__', '__name__', '__doc__', '__package__', '__loader__',
    '__spec__', '__annotations__', '__cached__', '__file__',
    '__pkl__', '__types__', '__f__', '__save__', '__excluded__', '__state_path__',
    '__saved__', '__Path__',
    'result_', 'is_json_serializable', 'safe_convert_to_serializable',
    'preferred_vars', 'captured', 'excluded', 'local_vars',
    'json', 'types', 'sys', 'Iterable',
    'var_name', 'var_value', 'converted_value', 'suitable_vars',
}}
for __k__, __v__ in dict(locals()).items():
    if (not __k__.startswith('__') and
        __k__ not in __excluded__ and
        not isinstance(__v__, __types__.ModuleType) and
        not callable(__v__)):
        try:
            __pkl__.dumps(__v__)
            __save__[__k__] = __v__
        except Exception:
            pass
with open({state_file!r}, 'wb') as __f__:
    __pkl__.dump(__save__, __f__)
del __pkl__, __types__, __save__, __excluded__, __k__, __v__, __f__
"""


def generate_state_restore_code(state_file: str) -> str:
    """Generate Python code that restores variables from a previous execution."""
    return RESTORE_TEMPLATE.format(state_file=state_file)


def generate_state_save_code(state_file: str) -> str:
    """Generate Python code that saves user-defined variables for the next execution."""
    return SAVE_TEMPLATE.format(state_file=state_file)
