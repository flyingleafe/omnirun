# v2 implementation follow-ups (integration-time)

Small items owed after the parallel waves; owner = integrator.

1. `engine/engine.py` `_enact_fail` docstring: stale — the model now HAS a
   queued→failed edge (`failQueued`) and the exporter emits `fail`.
   Update wording; `Fail` enactment should emit the `fail` event via
   `transition` (action token now validated), not only the diagnostic.
2. `tests/conftest.py` trace gate: remove the α-assert exemption for
   stores containing a `fail` event (obsolete since the model extension).
3. `state/`: `resources` PK forbids re-minting a released deterministic
   key (second placement arc of the same job on the same provider).
   P4b worked around it in the supervisor (StoreError caught, event still
   committed). Proper fix: `mint_resource` may revive a row whose
   `released_at` is set (clear released_at, bump minted_at, keep history
   in data) — still refusing duplicates of an UNRELEASED key (I7).
4. `providertypes.observe_terminal`: still used by the cancel grace
   window only; revisit whether cancel can ride `observe_batch` at P6.
