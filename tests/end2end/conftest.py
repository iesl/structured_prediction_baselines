from .common import marks


def pytest_configure(config) -> None:  # type: ignore
    for mark in marks:
        config.addinivalue_line("markers", mark)
