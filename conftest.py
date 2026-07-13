import pytest

def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'integration: requires a real simulator env and optional checkpoint',
    )
