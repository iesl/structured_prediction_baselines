#!python3

"""
Print all the external variables in a jsonnet config.

It looks for 'std.extVar(*)' in the file.
"""

import argparse
from pathlib import Path
import logging
import re

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "external_variables",
        description="Print the external variables in a jsonnet config",
    )
    parser.add_argument("input_file", type=Path)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    with open(args.input_file) as f:
        content = f.read()

        for match in re.finditer(
            r"std\.extVar\('(.+?)'\)", content, re.MULTILINE
        ):
            print(match.group(1))
