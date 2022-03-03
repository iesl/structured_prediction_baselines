from .common import marks
import nltk


try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

def pytest_configure(config) -> None:  # type: ignore
    for mark in marks:
        config.addinivalue_line("markers", mark)
