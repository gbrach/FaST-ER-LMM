"""Keep tiny CPU scan fixtures from oversubscribing BLAS threads."""

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
