import pytest
from acadian_arc.data import generate_synthetic_panel
from acadian_arc.signals import design_matrix


@pytest.fixture(scope="session")
def panel():
    return generate_synthetic_panel(months=22, seed=42)


@pytest.fixture(scope="session")
def design(panel):
    return design_matrix(panel)
