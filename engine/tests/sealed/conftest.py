"""Shared fixtures for the sealed challenge suite (slices A and C).

Written double-blind from CONTRACT-slice-A-simmember.md / CONTRACT-slice-C-env.md.
Imports of the implementation happen inside fixtures so collection never depends
on the implementation existing.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "category(name): sealed-suite category label for the referee"
    )


@pytest.fixture(scope="session")
def cohort0():
    from daypart.simmember.cohort import generate

    return generate(0)


@pytest.fixture(scope="session")
def cohort0_again():
    from daypart.simmember.cohort import generate

    return generate(0)


@pytest.fixture(scope="session")
def cohort1():
    from daypart.simmember.cohort import generate

    return generate(1)
