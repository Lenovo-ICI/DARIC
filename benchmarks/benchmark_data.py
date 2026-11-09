#!/usr/bin/env python3
"""Prepare and validate data for the vLLM serving benchmarks."""

import json
import sys


def convert(source: str, destination: str, count: int) -> None:
    with open(source, encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or len(records) < count:
        raise ValueError(f"{source} must contain at least {count} records")

    with open(destination, "w", encoding="utf-8") as file:
        for index, record in enumerate(records[:count]):
            if not isinstance(record, dict) or not isinstance(record.get("prompt"), str):
                raise ValueError(f"{source}: record {index} must have a string prompt")
            output_len = record.get("expected_output_len")
            if isinstance(output_len, bool) or not isinstance(output_len, int) or output_len < 1:
                raise ValueError(f"{source}: record {index} must have a positive expected_output_len")
            json.dump(
                {"prompt": record["prompt"], "output_tokens": output_len},
                file,
                ensure_ascii=False,
            )
            file.write("\n")


def check(path: str, expected: int) -> None:
    with open(path, encoding="utf-8") as file:
        result = json.load(file)
    if result.get("completed") != expected or result.get("failed") != 0:
        raise RuntimeError(
            f"{path}: expected {expected} successful requests, "
            f"got completed={result.get('completed')}, failed={result.get('failed')}"
        )


def main() -> None:
    command = sys.argv[1]
    if command == "convert":
        convert(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif command == "check":
        check(sys.argv[2], int(sys.argv[3]))
    elif command == "config":
        print(json.dumps({
            "kv_connector": "DARICConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"config_file": sys.argv[2]},
        }))
    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
