#!/usr/bin/env python3
"""ggk server panel entry point.

    ggk server              launch the server GUI in the browser
    ggk server engine ...   run the bundled gguf-server engine directly
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import webbrowser

from . import __version__, engine


def _cmd_serve(args) -> int:
    from .server import serve, shutdown_engine

    httpd = serve(host=args.host, port=args.port)

    # Ctrl-C arrives as KeyboardInterrupt below; SIGTERM (kill, a closing
    # terminal, a service manager) would otherwise skip the cleanup and
    # orphan the engine child. shutdown() must run off the serving
    # thread, hence the helper thread.
    def _terminate(_signum, _frame):
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    try:
        signal.signal(signal.SIGTERM, _terminate)
    except (ValueError, OSError, AttributeError):
        pass  # not the main thread, or no SIGTERM on this platform

    host, port = httpd.server_address[0], httpd.server_address[1]
    url = f"http://{host}:{port}/"
    print(f"ggk server {__version__} — serving on {url}")
    if not engine.is_available():
        print(f"note: engine unavailable ({engine.load_error()})")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        shutdown_engine()
        httpd.server_close()
    return 0


def _cmd_engine(args) -> int:
    argv = list(args.engine_args)
    if argv and argv[0] == "--":
        argv = argv[1:]
    return engine.run_cli(argv)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ggk server",
        description="Local OpenAI-compatible LLM server GUI for GGUF models.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="launch the server GUI (default)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8642,
                         help="port the GUI listens on (0 = auto; default 8642). "
                              "The LLM server itself defaults to 8888.")
    p_serve.add_argument("--no-browser", action="store_true",
                         help="don't open the web browser automatically")
    p_serve.set_defaults(func=_cmd_serve)

    p_engine = sub.add_parser(
        "engine", help="run the bundled gguf-server engine with raw arguments")
    p_engine.add_argument("engine_args", nargs=argparse.REMAINDER,
                          help="arguments passed to the engine verbatim")
    p_engine.set_defaults(func=_cmd_engine)

    argv_list = list(sys.argv[1:] if argv is None else argv)
    # bare invocation (`ggk server [--port ...]`) defaults to serve
    if not argv_list or argv_list[0] not in ("serve", "engine",
                                             "-h", "--help", "--version"):
        argv_list.insert(0, "serve")
    args = parser.parse_args(argv_list)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
