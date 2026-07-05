"""Serve the project dashboard with explicit UTF-8 headers.

Python's default ``http.server`` often serves Markdown as ``text/markdown``
without a charset. Some browsers then guess a legacy encoding and Chinese text
appears garbled. This tiny server keeps the same static-file behavior but adds
``charset=utf-8`` for text-like files.
"""

from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Utf8StaticHandler(SimpleHTTPRequestHandler):
    UTF8_SUFFIXES = {
        ".css",
        ".csv",
        ".html",
        ".js",
        ".json",
        ".md",
        ".svg",
        ".txt",
        ".yaml",
        ".yml",
    }

    def guess_type(self, path: str) -> str:
        content_type = super().guess_type(path)
        suffix = Path(path).suffix.lower()
        if suffix in self.UTF8_SUFFIXES and "charset=" not in content_type.lower():
            return f"{content_type}; charset=utf-8"
        return content_type

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the MoE platform with UTF-8 text headers.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--directory", default=Path.cwd(), type=Path)
    args = parser.parse_args()

    handler = functools.partial(Utf8StaticHandler, directory=str(args.directory.resolve()))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {args.directory.resolve()} at http://{args.host}:{args.port}/")
    print("Text files are served with charset=utf-8.")
    server.serve_forever()


if __name__ == "__main__":
    main()
