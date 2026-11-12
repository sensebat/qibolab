# CP1-CP5 references

One folder per laboneq major.minor (`26.4`, `26.10`), one `cpN.json` per checkpoint,
written by `test_references.py`. Each file holds, per case and per
`ZiController.play` call, what the driver hands to LabOne Q in qibolab terms
(channels, operations, frequencies, sweeps, acquisition loop) and the qibolab results
in emulation. Uids, section names, nesting and version-specific bookkeeping are left
out; see the module doc of `test_references.py`.

Regenerate when a driver change is meant to change what LabOne Q gets, or when a
laboneq version is added:

    pytest -p tests.instruments.zi.conformance.conftest_references \
        tests/instruments/zi/conformance/test_references.py --update-references -k driver

(`-p` loads the option from `conftest_references.py`). Read the JSON diff like code before committing it. Never
regenerate only to turn a failing test green: a diff nobody expected is a driver
change to explain.
