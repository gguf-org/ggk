#!/usr/bin/env python3
"""ggk command line entry point.

    python -m ggk / ggk    launch the unified GUI (3 panels)
    ggk server [...]       the LLM server panel on its own
    ggk diffuser [...]     the image generation panel on its own
    ggk editor [...]       the GGUF editor panel on its own

Each panel subcommand accepts exactly what its standalone package did, so
    ggk server engine -- --model model.gguf
    ggk diffuser engine -- --help
    ggk editor quantize -m in.gguf -o out.gguf --type q4_k
run the bundled engines directly.
"""

from __future__ import annotations

import argparse
import importlib
import signal
import sys
import threading
import webbrowser

from . import __version__

PANELS = ("server", "diffuser", "editor")


def _cmd_gui(args) -> int:
    from . import gui

    httpd = gui.serve(host=args.host, port=args.port)

    # Ctrl-C arrives as KeyboardInterrupt below; SIGTERM (kill, a closing
    # terminal, a service manager) would otherwise skip the cleanup and
    # orphan an engine child. shutdown() must run off the serving thread,
    # hence the helper thread.
    def _terminate(_signum, _frame):
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    try:
        signal.signal(signal.SIGTERM, _terminate)
    except (ValueError, OSError, AttributeError):
        pass  # not the main thread, or no SIGTERM on this platform

    host, port = httpd.server_address[0], httpd.server_address[1]
    url = f"http://{host}:{port}/"
    print(f"ggk {__version__} — serving on {url}")
    for name in PANELS:
        print(f"  {name:<9} {url}{name}/")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        gui.shutdown_engines()
        httpd.server_close()
    return 0


def main(argv=None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)

    # panel subcommands delegate to the panel's own CLI, unchanged from the
    # standalone gguf-server / gguf-diffusion / gguf-editor packages
    if argv_list and argv_list[0] in PANELS:
        panel = argv_list[0]
        module = importlib.import_module(f".{panel}.__main__", package=__package__)
        return module.main(argv_list[1:])

    parser = argparse.ArgumentParser(
        prog="ggk",
        description="Unified GUI for GGUF models: LLM server, image generation "
                    "and file editing in one place.",
        epilog="panel subcommands: ggk server|diffuser|editor [...] "
               "(each accepts its standalone CLI, e.g. `ggk server engine -- --help`)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8640,
                        help="port the unified GUI listens on (0 = auto; default 8640)")
    parser.add_argument("--no-browser", action="store_true",
                        help="don't open the web browser automatically")
    args = parser.parse_args(argv_list)
    return _cmd_gui(args)


if __name__ == "__main__":
    sys.exit(main())
