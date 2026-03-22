#!/usr/bin/env python3
"""
galc — GAL to C++23 transpiler

Usage:
    galc.py <input.gal> [-o output.cpp] [-I dir] ...
"""

import sys
import os
import argparse
from lexer   import lex
from parser  import parse
from codegen import generate


def main() -> None:
    ap = argparse.ArgumentParser(prog="galc", description="GAL → C++23 transpiler")
    ap.add_argument("input",          help="source .gal file")
    ap.add_argument("-o", "--output", help="output .cpp file (default: stdout)")
    ap.add_argument("-I", "--include", action="append", default=[],
                    metavar="DIR",    help="add include-search directory")
    args = ap.parse_args()

    try:
        src = open(args.input).read()
    except OSError as e:
        print(f"galc: error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        tokens = lex(src, args.input)
        tu     = parse(tokens, args.input)
        code   = generate(tu, args.input, args.include)
    except Exception as e:
        print(f"galc: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        try:
            open(args.output, "w").write(code)
        except OSError as e:
            print(f"galc: error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(code)


if __name__ == "__main__":
    main()
