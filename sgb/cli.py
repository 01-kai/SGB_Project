from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from sgb.config import (
    BASE_CONFIG_PATH,
    ConfigurationError,
    STUDY_CONFIG_DIRECTORY,
    load_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sgb",
        description=(
            "Synthetic Governance Benchmark command-line interface."
        ),
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    config_parser = commands.add_parser(
        "config",
        help="Validate and inspect SGB configuration files.",
    )

    config_commands = config_parser.add_subparsers(
        dest="config_command",
        required=True,
    )

    validate_parser = config_commands.add_parser(
        "validate",
        help="Validate one configuration file.",
    )
    validate_parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the YAML configuration.",
    )

    show_parser = config_commands.add_parser(
        "show",
        help="Validate and print one configuration file.",
    )
    show_parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the YAML configuration.",
    )

    config_commands.add_parser(
        "paths",
        help="Display standard configuration paths.",
    )

    return parser


def handle_config_command(
    arguments: argparse.Namespace,
) -> int:
    if arguments.config_command == "validate":
        config = load_config(arguments.file)

        print(
            f"Valid {config['config_kind']} configuration: "
            f"{config['_metadata']['source_path']}"
        )
        return 0

    if arguments.config_command == "show":
        config = load_config(arguments.file)

        printable_config = {
            key: value
            for key, value in config.items()
            if key != "_metadata"
        }

        print(
            json.dumps(
                printable_config,
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if arguments.config_command == "paths":
        print(
            json.dumps(
                {
                    "base": str(BASE_CONFIG_PATH),
                    "studies": str(STUDY_CONFIG_DIRECTORY),
                },
                indent=2,
            )
        )
        return 0

    raise ConfigurationError(
        f"Unsupported config command: "
        f"{arguments.config_command!r}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "config":
            return handle_config_command(arguments)

        parser.error(
            f"Unsupported command: {arguments.command}"
        )

    except ConfigurationError as error:
        print(
            f"Configuration error: {error}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())