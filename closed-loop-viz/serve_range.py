"""Static file server with HTTP Range support, so <video> can seek (python -m http.server has none).

  python closed-loop-viz/serve_range.py 8780 /path/to/gallery     # binds 127.0.0.1; reach it via ssh -L 8780:localhost:8780
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            self._left = None
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        size = os.path.getsize(path)
        a = int(m.group(1)) if m.group(1) else 0
        b = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
        f = open(path, "rb")  # noqa: SIM115 -- http.server hands the open file to copyfile and closes it
        f.seek(a)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        self._left = b - a + 1
        return f

    def copyfile(self, src, dst):
        n = getattr(self, "_left", None)
        try:
            if n is None:
                return super().copyfile(src, dst)
            while n > 0:
                buf = src.read(min(1 << 16, n))
                if not buf:
                    break
                dst.write(buf)
                n -= len(buf)
        except (BrokenPipeError, ConnectionResetError):
            pass        # the browser drops a range request whenever the video seeks; nothing to report

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port, root = int(sys.argv[1]), sys.argv[2]
    os.chdir(root)
    print(f"serving {root} at http://127.0.0.1:{port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
