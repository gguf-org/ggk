#!/usr/bin/env python3
"""ggk editor panel entry point.

    ggk editor [model.gguf]   launch the editor GUI in the browser
    ggk editor quantize ...   run the bundled quantizer (CLI)
    ggk editor devices        list quantizer devices in this build
"""

from __future__ import annotations

import argparse
import sys
import webbrowser

from . import __version__, quantizer


def _cmd_serve(args) -> int:
    from .server import serve

    httpd = serve(host=args.host, port=args.port, initial_file=args.model)
    host, port = httpd.server_address[0], httpd.server_address[1]
    url = f"http://{host}:{port}/"
    print(f"ggk editor {__version__} — serving on {url}")
    if not quantizer.is_available():
        print(f"note: quantizer engine unavailable ({quantizer.load_error()})")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
    return 0


def _cmd_quantize(args) -> int:
    def on_log(level, message):
        prefix = {quantizer.LOG_WARN: "warning:", quantizer.LOG_ERROR: "error:"}.get(level, "")
        print(f"{prefix} {message}".strip(), file=sys.stderr)

    try:
        quantizer.quantize(
            args.model, args.output,
            default_type=args.type,
            tensor_type_rules=args.tensor_type_rules,
            n_threads=args.threads,
            device=args.device,
            log_callback=on_log,
        )
    except quantizer.QuantizeCancelled:
        return 2
    except quantizer.QuantizeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_devices(_args) -> int:
    if not quantizer.is_available():
        print(f"error: {quantizer.load_error()}", file=sys.stderr)
        return 1
    print("available devices:")
    for dev in quantizer.list_devices():
        print(f"  {dev['name']:<16} {dev['description']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ggk editor",
        description="GGUF metadata/tensor editor with a built-in quantizer.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="launch the editor GUI (default)")
    p_serve.add_argument("model", nargs="?", help="GGUF file to open on start")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8644,
                         help="port to listen on (0 = auto; default 8644)")
    p_serve.add_argument("--no-browser", action="store_true",
                         help="don't open the web browser automatically")
    p_serve.set_defaults(func=_cmd_serve)

    p_quant = sub.add_parser("quantize", help="quantize a model (no GUI)")
    p_quant.add_argument("-m", "--model", required=True,
                         help="input model file (GGUF or safetensors)")
    p_quant.add_argument("-o", "--output", required=True, help="output GGUF file")
    p_quant.add_argument("--type", help="default weight type (e.g. q4_k)")
    p_quant.add_argument("--tensor-type-rules",
                         help='comma-separated "<regex>=<type>" overrides')
    # p_quant.add_argument("--device", default="cpu",
    p_quant.add_argument("--device", default="auto",
                         help="cpu (default), auto, or a device from `devices`")
    p_quant.add_argument("-t", "--threads", type=int, default=0,
                         help="number of threads (0 = auto)")
    p_quant.set_defaults(func=_cmd_quantize)

    p_dev = sub.add_parser("devices", help="list quantizer devices")
    p_dev.set_defaults(func=_cmd_devices)

    argv_list = list(sys.argv[1:] if argv is None else argv)
    # bare invocation (`ggk editor [model.gguf] [--port ...]`)
    # defaults to the `serve` subcommand
    if not argv_list or argv_list[0] not in ("serve", "quantize", "devices",
                                             "-h", "--help", "--version"):
        argv_list.insert(0, "serve")
    args = parser.parse_args(argv_list)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
