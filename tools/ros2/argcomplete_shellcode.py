"""Print argcomplete's bash completion code for the given executables."""

import sys

import argcomplete

if __name__ == "__main__":
    sys.stdout.write(argcomplete.shellcode(sys.argv[1:]))
