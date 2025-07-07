# dependencies/general/mkdir_if_not_exists.py
import logging
import os

logger = logging.getLogger(__name__)


def mkdir_if_not_exists(directory: str) -> None:
    if not os.path.isdir(directory):
        os.makedirs(directory)
        logger.info("Created new directory: %s", directory)
    else:
        logger.info("Directory exists, skipping creation\n%s", directory)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a directory if it does not exist."
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory path to create if not present.",
    )
    args = parser.parse_args()
    mkdir_if_not_exists(args.directory)
