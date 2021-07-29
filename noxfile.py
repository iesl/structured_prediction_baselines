"""Nox sessions."""
import shutil
import sys
from pathlib import Path
from textwrap import dedent

import nox
from nox import Session, session

package = "python_research_project"
python_versions = [
    "3.8",
]
nox.needs_version = ">= 2021.6.6"
nox.options.sessions = (
    "pre-commit",
    "mypy",
    "tests",
)


def activate_virtualenv_in_precommit_hooks(session: Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    Args:
        session: The Session object.
    """

    if session.bin is None:
        return

    virtualenv = session.env.get("VIRTUAL_ENV")

    if virtualenv is None:
        return

    hookdir = Path(".git") / "hooks"

    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        text = hook.read_text()
        bindir = repr(session.bin)[1:-1]  # strip quotes

        if not (
            Path("A") == Path("a")
            and bindir.lower() in text.lower()
            or bindir in text
        ):
            continue

        lines = text.splitlines()

        if not (lines[0].startswith("#!") and "python" in lines[0].lower()):
            continue

        header = dedent(
            f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """
        )

        lines.insert(1, header)
        hook.write_text("\n".join(lines))


@session(name="pre-commit", python="3.8")
def precommit(session: Session) -> None:
    """Lint using pre-commit."""
    args = session.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    session.install("-r", "lint_requirements.txt")
    session.run("pre-commit", *args)

    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@session(python="3.8")
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or ["structured_prediction_baselines", "tests"]
    # read the mypy version from lint_requirements.txt
    with open("lint_requirements.txt") as f:
        for line in f:
            if "mypy" in line:
                mypy_version = line.strip()
    session.install(mypy_version)
    session.run("mypy", *args)

    if not session.posargs:
        session.run(
            "mypy", f"--python-executable={sys.executable}", "noxfile.py"
        )


@session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    session.install("-r", "core_requirements.txt")
    session.install(".")
    session.install("-r", "test_requirements.txt")
    try:
        session.run(
            "coverage", "run", "--parallel", "-m", "pytest", *session.posargs
        )
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@session
def coverage(session: Session) -> None:
    """Produce the coverage report."""
    args = session.posargs or ["report"]

    session.install("coverage[toml]")

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args)
