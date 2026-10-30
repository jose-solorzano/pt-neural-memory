"""Reproducible benchmarks and the reference continual-regression protocol.

The modules are runnable with ``python -m``:

* ``memory_bench`` -- one RLS cell: capacity, retention, correlations, noise,
  drift and write cost.
* ``addressed_bench`` -- the addressed table: load, cue noise, read strategy,
  ridge and cost.
* ``scaling_bench`` -- occupancy and block-count sweeps.
* ``balance_bench`` -- error law and the address-bits/blocks budget split.
* ``capacity_bench`` -- one cell versus a table at equal exactness.
* ``forgetting_bench`` -- ten-part regression stream, MLP baselines,
  replay and the addressed memory.
* ``training_cost`` -- wall-clock fit time for every method, plus a from-scratch
  MLP budget curve.
* ``export_forgetting`` -- rebuild the published Hugging Face CSV dataset
  (requires the ``export``/``hub`` extras).
"""

__all__: list[str] = []
