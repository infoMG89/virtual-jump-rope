#!/usr/bin/env python3
"""
Reverse proxy for Svihej game-app development.
Serves local frame.html and gameset.html at the correct path,
proxies everything else to form-data.cz so all files share the same origin.

Run: python3 serve.py
Open: http://localhost:8080/
"""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error
import urllib.parse
import os

BASE_URL  = "https://www.form-data.cz"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
PORT      = 8080

APP_BASE  = "/app/svihej-camapp/v2025-july-mux"
VAL_DIR   = ("/Users/martingren/Library/CloudStorage/"
             "GoogleDrive-gren.martin89@gmail.com/Shared drives/"
             "VÝVOJ PRODUKTŮ/Počítadlo/Validační videa")

# Local files served at their proxied path
OVERRIDES = {
    "/debug.html": "debug.html",
    APP_BASE + "/frame.html":   "frame.html",
    APP_BASE + "/gameset.html": "gameset.html",
    "/index.html":              "index.html",
    "/":                        "index.html",
    "/poc.html":                "poc.html",
    "/game_rhythm.html":        "game_rhythm.html",
    "/game_rope.html":          "game_rope.html",
    "/mix_237plus_skoku.mp4":   "mix_237plus_skoku.mp4",
    "/test-video.mp4":          "/Users/martingren/Library/CloudStorage/GoogleDrive-gren.martin89@gmail.com/Shared drives/VÝVOJ PRODUKTŮ/Počítadlo/Videa testerů/IMG_9717.MP4",
}


def _fetch_url(path):
    url = BASE_URL + path
    req = urllib.request.Request(url, headers={
        "User-Agent":      "Mozilla/5.0",
        "Accept":          "*/*",
        "Accept-Encoding": "identity",
    })
    with urllib.request.urlopen(req, timeout=20) as resp:
        ct   = resp.headers.get("Content-Type", "application/octet-stream")
        body = resp.read()
    return ct, body


class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path_only = self.path.split("?")[0]
        if path_only in OVERRIDES:
            self._serve_local(OVERRIDES[path_only])
        elif path_only.startswith("/val/"):
            fname = urllib.parse.unquote(path_only[5:])
            self._serve_local(os.path.join(VAL_DIR, fname))
        else:
            self._proxy(self.path)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_local(self, filename):
        local_path = filename if os.path.isabs(filename) else os.path.join(LOCAL_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()
        ct_map = {".html": "text/html; charset=utf-8", ".js": "application/javascript",
                  ".css": "text/css", ".json": "application/json",
                  ".mp4": "video/mp4"}
        ct = ct_map.get(ext, "application/octet-stream")
        try:
            if ext == ".mp4":
                self._serve_video(local_path, ct)
            else:
                with open(local_path, "rb") as f:
                    body = f.read()
                self._send(200, ct, body)
        except FileNotFoundError:
            self._send(404, "text/plain", b"Not found")

    def _serve_video(self, path, ct):
        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")
        if range_header:
            # Parse "bytes=start-end"
            byte_range = range_header.strip().split("=")[1]
            start_str, _, end_str = byte_range.partition("-")
            start = int(start_str) if start_str else 0
            end   = int(end_str)   if end_str   else file_size - 1
            end   = min(end, file_size - 1)
            length = end - start + 1
            with open(path, "rb") as f:
                f.seek(start)
                body = f.read(length)
            self.send_response(206)
            self.send_header("Content-Type",                ct)
            self.send_header("Content-Length",              str(length))
            self.send_header("Content-Range",               f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges",               "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Type",                ct)
            self.send_header("Content-Length",              str(file_size))
            self.send_header("Accept-Ranges",               "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    self.wfile.write(chunk)

    def _proxy(self, path):
        try:
            ct, body = _fetch_url(path)
            self._send(200, ct, body)
        except urllib.error.HTTPError as e:
            self._send(e.code, "text/plain", str(e).encode())
        except Exception as e:
            self._send(502, "text/plain", str(e).encode())

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type",                content_type)
        self.send_header("Content-Length",              str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"  {args[0]} {args[1]}", flush=True)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("localhost", PORT), ProxyHandler)
    print(f"Svihej game-app proxy → http://localhost:{PORT}/", flush=True)
    server.serve_forever()
