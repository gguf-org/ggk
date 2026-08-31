#!/usr/bin/env python3
"""ggk diffuser panel entry point.

    ggk diffuser              launch the diffusion GUI in the browser
    ggk diffuser engine ...   run the bundled diffusion CLI directly
"""

from __future__ import annotations

import argparse
import sys
import webbrowser

from . import __version__, engine


def _cmd_serve(args) -> int:
    from .server import serve

    httpd = serve(host=args.host, port=args.port)
    host, port = httpd.server_address[0], httpd.server_address[1]
    url = f"http://{host}:{port}/"
    print(f"ggk diffuser {__version__} — serving on {url}")
    if not engine.is_available():
        print(f"note: diffusion engine unavailable ({engine.load_error()})")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
    return 0


def _cmd_engine(args) -> int:
    argv = list(args.engine_args)
    if argv and argv[0] == "--":
        argv = argv[1:]
    return engine.run_cli(argv)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ggk diffuser",
        description="Image generation GUI for GGUF diffusion models.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="launch the diffusion GUI (default)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8643,
                         help="port to listen on (0 = auto; default 8643)")
    p_serve.add_argument("--no-browser", action="store_true",
                         help="don't open the web browser automatically")
    p_serve.set_defaults(func=_cmd_serve)

    p_engine = sub.add_parser(
        "engine", help="run the bundled diffusion.cpp CLI with raw arguments")
    p_engine.add_argument("engine_args", nargs=argparse.REMAINDER,
                          help="arguments passed to the diffusion binary verbatim")
    p_engine.set_defaults(func=_cmd_engine)

    argv_list = list(sys.argv[1:] if argv is None else argv)
    # bare invocation (`ggk diffuser [--port ...]`) defaults to serve
    if not argv_list or argv_list[0] not in ("serve", "engine",
                                             "-h", "--help", "--version"):
        argv_list.insert(0, "serve")
    args = parser.parse_args(argv_list)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
