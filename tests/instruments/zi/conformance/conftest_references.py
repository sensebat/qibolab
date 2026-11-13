"""The ``--update-references`` option of test_references.py.

To be loaded once, either by ``conftest.py`` (``from .conftest_references import
pytest_addoption``) or on the command line::

    pytest -p tests.instruments.zi.conformance.conftest_references \
        tests/instruments/zi/conformance/test_references.py --update-references

Without it, the option is off and the references are only compared.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--update-references",
        action="store_true",
        default=False,
        help="Rewrite the CP1-CP5 reference files of the installed laboneq version.",
    )
