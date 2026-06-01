from __future__ import annotations

import sys

from . import product_terminal


def gui_main(argv: list[str] | None = None) -> int:
    try:
        from .product_gui import main as run_gui
    except ModuleNotFoundError as exc:
        if exc.name != "tkinter":
            raise
        print(
            "IDS Sentinel GUI requires tkinter, but it is not installed for this Python.",
            file=sys.stderr,
        )
        print(
            "Install it on Ubuntu/Debian with: sudo apt install python3-tk",
            file=sys.stderr,
        )
        print(
            "If your system uses a version-specific package, try: sudo apt install python3.14-tk",
            file=sys.stderr,
        )
        print("The terminal CLI still works: ids-sentinel status", file=sys.stderr)
        return 1
    return run_gui(argv or [])


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0].lower() == "gui":
        return gui_main(args[1:])
    return product_terminal.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
