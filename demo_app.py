import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from outfit_inference import (
    DEFAULT_CHECKPOINT,
    image_bytes_to_pil,
    image_to_data_url,
    load_checkpoint,
    predict_image,
    random_test_image,
)


HOST = "127.0.0.1"
PORT = 8000
MAX_PORT_TRIES = 20

MODEL, CHECKPOINT, DEVICE = load_checkpoint(DEFAULT_CHECKPOINT)


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Outfit Classifier Demo</title>
  <style>
    :root { color-scheme: light; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f5f6f8; color: #1f2933; }
    main { max-width: 1050px; margin: 0 auto; padding: 28px 18px 40px; }
    h1 { margin: 0 0 6px; font-size: 30px; letter-spacing: 0; }
    .meta { margin: 0 0 22px; color: #52606d; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 18px; }
    button, label.file { border: 1px solid #9aa5b1; background: #fff; color: #1f2933; border-radius: 6px; padding: 10px 14px; font-size: 15px; cursor: pointer; }
    button.primary { background: #116466; border-color: #116466; color: #fff; }
    button:disabled { opacity: .5; cursor: wait; }
    input[type=file] { display: none; }
    .threshold { display: inline-flex; gap: 8px; align-items: center; padding: 8px 10px; background: #fff; border: 1px solid #d1d7de; border-radius: 6px; }
    .threshold input { width: 78px; }
    .layout { display: grid; grid-template-columns: minmax(280px, 430px) 1fr; gap: 18px; align-items: start; }
    .panel { background: #fff; border: 1px solid #d9dee5; border-radius: 8px; padding: 14px; }
    .preview { width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #eef1f4; border-radius: 6px; border: 1px solid #d9dee5; }
    .empty { display: grid; place-items: center; aspect-ratio: 1 / 1; color: #7b8794; background: #eef1f4; border-radius: 6px; border: 1px dashed #9aa5b1; text-align: center; padding: 12px; }
    .truth { margin-top: 12px; color: #3e4c59; font-size: 14px; }
    .status { margin: 0 0 12px; color: #52606d; min-height: 22px; }
    .result { display: grid; gap: 8px; }
    .row { display: grid; grid-template-columns: 150px 1fr 58px; gap: 10px; align-items: center; font-size: 14px; }
    .bar { height: 10px; background: #e4e7eb; border-radius: 999px; overflow: hidden; }
    .fill { height: 100%; background: #116466; }
    .predicted { font-weight: 700; }
    @media (max-width: 820px) { .layout { grid-template-columns: 1fr; } h1 { font-size: 25px; } }
  </style>
</head>
<body>
<main>
  <h1>Outfit Classifier Demo</h1>
  <p class="meta" id="modelMeta"></p>
  <div class="toolbar">
    <label class="file">Choose image<input id="fileInput" type="file" accept="image/*"></label>
    <button class="primary" id="predictBtn">Predict</button>
    <button id="sampleBtn">Random test image</button>
    <button id="clearBtn">Clear</button>
    <span class="threshold">Threshold <input id="thresholdInput" type="number" min="0" max="1" step="0.05" value="0.5"></span>
  </div>
  <section class="layout">
    <div class="panel">
      <div id="imageBox" class="empty">Choose an image or load a random test image.</div>
      <div class="truth" id="truthBox"></div>
    </div>
    <div class="panel">
      <p class="status" id="status"></p>
      <div class="result" id="results"></div>
    </div>
  </section>
</main>
<script>
const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const sampleBtn = document.getElementById('sampleBtn');
const clearBtn = document.getElementById('clearBtn');
const thresholdInput = document.getElementById('thresholdInput');
const imageBox = document.getElementById('imageBox');
const truthBox = document.getElementById('truthBox');
const statusBox = document.getElementById('status');
const resultsBox = document.getElementById('results');
const modelMeta = document.getElementById('modelMeta');
let currentFile = null;

async function loadMeta() {
  const response = await fetch('/meta');
  const data = await response.json();
  modelMeta.textContent = `${data.model_name} | ${data.image_size}px | ${data.num_classes} labels | device: ${data.device}`;
  thresholdInput.value = data.threshold;
}

function showImage(src) {
  imageBox.className = '';
  imageBox.innerHTML = `<img class="preview" src="${src}" alt="selected outfit image">`;
}

function renderPredictions(data) {
  const top = data.predictions.slice(0, 10);
  statusBox.textContent = `${data.model_name} | threshold ${data.threshold.toFixed(2)} | ${data.latency_ms.toFixed(1)} ms`;
  resultsBox.innerHTML = top.map(item => {
    const width = Math.max(2, Math.round(item.probability * 100));
    const cls = item.predicted ? 'predicted' : '';
    return `<div class="row ${cls}">
      <span>${item.label}</span>
      <span class="bar"><span class="fill" style="width:${width}%"></span></span>
      <span>${(item.probability * 100).toFixed(1)}%</span>
    </div>`;
  }).join('');
}

async function predictBlob(blob) {
  predictBtn.disabled = true;
  sampleBtn.disabled = true;
  statusBox.textContent = 'Running inference...';
  const threshold = encodeURIComponent(thresholdInput.value || '0.5');
  const response = await fetch(`/predict?threshold=${threshold}`, {
    method: 'POST',
    headers: {'Content-Type': blob.type || 'application/octet-stream'},
    body: blob
  });
  const data = await response.json();
  predictBtn.disabled = false;
  sampleBtn.disabled = false;
  if (!response.ok) {
    statusBox.textContent = data.error || 'Prediction failed.';
    return;
  }
  renderPredictions(data);
}

fileInput.addEventListener('change', () => {
  currentFile = fileInput.files[0] || null;
  truthBox.textContent = '';
  resultsBox.innerHTML = '';
  statusBox.textContent = '';
  if (currentFile) showImage(URL.createObjectURL(currentFile));
});

predictBtn.addEventListener('click', () => {
  if (!currentFile) {
    statusBox.textContent = 'Choose an image first.';
    return;
  }
  predictBlob(currentFile);
});

sampleBtn.addEventListener('click', async () => {
  predictBtn.disabled = true;
  sampleBtn.disabled = true;
  statusBox.textContent = 'Loading sample...';
  const threshold = encodeURIComponent(thresholdInput.value || '0.5');
  const response = await fetch(`/sample?threshold=${threshold}`);
  const data = await response.json();
  predictBtn.disabled = false;
  sampleBtn.disabled = false;
  if (!response.ok) {
    statusBox.textContent = data.error || 'Sample failed.';
    return;
  }
  currentFile = null;
  showImage(data.image_data_url);
  truthBox.textContent = `True labels: ${data.true_labels.join(', ') || 'none'}`;
  renderPredictions(data.result);
});

clearBtn.addEventListener('click', () => {
  currentFile = null;
  fileInput.value = '';
  imageBox.className = 'empty';
  imageBox.textContent = 'Choose an image or load a random test image.';
  truthBox.textContent = '';
  statusBox.textContent = '';
  resultsBox.innerHTML = '';
});

loadMeta();
</script>
</body>
</html>
"""


class DemoHandler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _threshold(self):
        params = parse_qs(urlparse(self.path).query)
        value = params.get("threshold", [CHECKPOINT.get("threshold", 0.5)])[0]
        return max(0.0, min(1.0, float(value)))

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif parsed.path == "/meta":
                self._send(200, {
                    "model_name": CHECKPOINT.get("model_name", "model"),
                    "image_size": int(CHECKPOINT["image_size"]),
                    "threshold": float(CHECKPOINT.get("threshold", 0.5)),
                    "num_classes": len(CHECKPOINT["class_names"]),
                    "device": str(DEVICE),
                })
            elif parsed.path == "/sample":
                threshold = self._threshold()
                image_path, true_labels = random_test_image(CHECKPOINT)
                result = predict_image(MODEL, CHECKPOINT, image_bytes_to_pil(Path(image_path).read_bytes()), DEVICE, threshold)
                self._send(200, {
                    "filename": Path(image_path).name,
                    "image_data_url": image_to_data_url(image_path),
                    "true_labels": true_labels,
                    "result": result,
                })
            else:
                self._send(404, {"error": "Not found"})
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            self._send(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(length)
            image = image_bytes_to_pil(data)
            result = predict_image(MODEL, CHECKPOINT, image, DEVICE, self._threshold())
            self._send(200, result)
        except Exception as exc:
            self._send(400, {"error": str(exc)})

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")


def create_server(host=HOST, start_port=PORT, max_tries=MAX_PORT_TRIES):
    for port in range(start_port, start_port + max_tries):
        try:
            return ThreadingHTTPServer((host, port), DemoHandler), port
        except OSError as exc:
            if exc.errno not in {98, 48}:
                raise
            print(f"Port {port} is already in use; trying {port + 1}.")
    end_port = start_port + max_tries - 1
    raise OSError(f"No free port found from {start_port} to {end_port}.")


if __name__ == "__main__":
    server, port = create_server()
    print(f"Demo running at http://{HOST}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()
