#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def convert_inline_math(line: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(line):
        if line[i] != "$" or (i > 0 and line[i - 1] == "\\"):
            out.append(line[i])
            i += 1
            continue

        if i + 1 < len(line) and line[i + 1] == "$":
            out.append("$$")
            i += 2
            continue

        j = i + 1
        while j < len(line):
            if line[j] == "$" and line[j - 1] != "\\":
                break
            j += 1

        if j >= len(line):
            out.append(line[i])
            i += 1
            continue

        out.append(r"\(")
        out.append(line[i + 1 : j])
        out.append(r"\)")
        i = j + 1

    return "".join(out)


def prepare_markdown(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_code = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue

        if in_code:
            out.append(line)
            i += 1
            continue

        if stripped == "$$":
            inner: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                if lines[i].strip():
                    inner.append(lines[i].strip())
                i += 1

            out.append("$$")
            out.append(" ".join(inner))
            out.append("$$")
            if i < len(lines):
                i += 1
            continue

        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            out.append("$$")
            out.append(stripped[2:-2].strip())
            out.append("$$")
            i += 1
            continue

        out.append(convert_inline_math(line))
        i += 1

    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    if args.destination.exists():
        shutil.rmtree(args.destination)
    shutil.copytree(args.source, args.destination)

    for path in args.destination.rglob("*.md"):
        path.write_text(prepare_markdown(path.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
