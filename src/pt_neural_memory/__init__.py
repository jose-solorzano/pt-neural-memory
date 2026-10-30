"""Fast associative memory built from addressed recursive-least-squares cells.

The package has three layers:

* :class:`RLSMemCell` -- one linear map fitted by the exact recursive-least-
  squares update, which is the Kalman filter for ``value = W key + noise``,
  with its posterior error covariance.
* :class:`AddressedMemory` -- a bank of independent Kalman filters selected by
  the sign bits of a random projection, so many local maps can be stored and
  read.
* :class:`AddressedRegressor` -- a random feature map composed with the table,
  which turns the memory into a nonlinear regressor fitted in closed form.

A store is an indexed, exact posterior update and a read is a table lookup and
a matrix-vector product. Stores are no-grad by default;
``store(..., grad_enabled=True)`` keeps the graph through a write. The feature
map remains random and untrained, and address selection is discrete.
"""

from importlib.metadata import PackageNotFoundError, version

from pt_neural_memory.addressed_memory import (
    AddressedMemory,
    CellRead,
    VerifiedMemory,
)
from pt_neural_memory.addressed_regressor import AddressedRegressor, RandomFeatureMap
from pt_neural_memory.forgetting_data import Task, regression_stream, target_function
from pt_neural_memory.mem_rls import RLSCellBank, RLSMemCell
from pt_neural_memory.memory_sizing import (
    ADMISSIBLE_WRITERS_2PCT,
    ADMISSIBLE_WRITERS_5PCT,
    MemorySizing,
    balanced_layout,
    bits_for_capacity,
    capacity,
    pairs_per_cell,
    size_for_budget,
    state_bytes,
    state_parts,
    writers_per_cell,
)

try:
    __version__ = version("pt-neural-memory")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.1.0"

__all__ = [
    "ADMISSIBLE_WRITERS_2PCT",
    "ADMISSIBLE_WRITERS_5PCT",
    "AddressedMemory",
    "AddressedRegressor",
    "CellRead",
    "MemorySizing",
    "RLSCellBank",
    "RLSMemCell",
    "RandomFeatureMap",
    "Task",
    "VerifiedMemory",
    "__version__",
    "balanced_layout",
    "bits_for_capacity",
    "capacity",
    "pairs_per_cell",
    "regression_stream",
    "size_for_budget",
    "state_bytes",
    "state_parts",
    "target_function",
    "writers_per_cell",
]
