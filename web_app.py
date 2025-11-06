import io
import os
from typing import List

import numpy as np
import torch
from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template_string,
    send_from_directory,
)
from PIL import Image
from torchvision import transforms

from model import SupConMobileNet


# Configuration
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Device selection with safe CUDA fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    try:
        major, minor = torch.cuda.get_device_capability(0)
        if major * 10 + minor < 50:
            device = torch.device("cpu")
    except Exception:
        device = torch.device("cpu")

# Labels (alphabetical)
LABELS: List[str] = [
    "anthracnose",
    "healthy",
    "powdery",
    "rust",
    "sooty_mold",
    "spot",
    "yellow",
]

# Model setup (load once)
NUM_CLASSES = 7
MODEL_WEIGHTS = "/home/dangmanh/study/fpt_uni/dbm302m/plant_disease_resnet18.pth"

model = SupConMobileNet(num_classes=NUM_CLASSES).to(device)
state = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=False)
model.load_state_dict(state)
model.eval()

# Preprocess (match infer.py)
resize_transform = transforms.Resize((256, 256))


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    pil_img = pil_img.convert("RGB")
    pil_img = resize_transform(pil_img)
    np_img = np.array(pil_img).astype(np.float32)
    np_img = np_img / 255.0
    np_img = np.transpose(np_img, (2, 0, 1))
    tensor = torch.tensor(np_img, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)  # (1, C, H, W)
    return tensor.to(device)


def predict(pil_img: Image.Image):
    tensor = preprocess_image(pil_img)
    with torch.no_grad():
        logits = model(tensor, return_features=False)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probabilities))
        pred_label = LABELS[pred_idx]
        return pred_label, probabilities


# Flask app
app = Flask(__name__)


def find_logo() -> str | None:
    """Return a static logo filename if it exists, else None."""
    candidates = [
        "logo_fpt.png",
        "logo_fpt.jpg",
        "logo_fpt.jpeg",
        "logo_fpt.webp",
        "logo.png",
        "logo.jpg",
        "logo.jpeg",
        "logo.webp",
    ]
    static_dir = app.static_folder or os.path.join(os.path.dirname(__file__), "static")
    for name in candidates:
        if os.path.exists(os.path.join(static_dir, name)):
            return name
    return None


def list_ai_images(max_items: int = 5) -> list[str]:
    """Return up to max_items image paths under static suitable for the AI panel."""
    static_dir = app.static_folder or os.path.join(os.path.dirname(__file__), "static")
    ai_dir = os.path.join(static_dir, "ai")
    exts = {".gif", ".webp", ".png", ".jpg", ".jpeg"}
    results: list[str] = []
    if os.path.isdir(ai_dir):
        for name in sorted(os.listdir(ai_dir)):
            _, ext = os.path.splitext(name.lower())
            if ext in exts:
                results.append(f"ai/{name}")
                if len(results) >= max_items:
                    break
    return results


def left_gif() -> str | None:
    static_dir = app.static_folder or os.path.join(os.path.dirname(__file__), "static")
    path = os.path.join(static_dir, "ai", "094a7727804cdbeda42b8102fa80693c.gif")
    if os.path.exists(path):
        return "ai/094a7727804cdbeda42b8102fa80693c.gif"
    return None


def right_gif() -> str | None:
    static_dir = app.static_folder or os.path.join(os.path.dirname(__file__), "static")
    path = os.path.join(static_dir, "ai", "231fb5027639114dd7cf3f8f3ef9cb86.gif")
    if os.path.exists(path):
        return "ai/231fb5027639114dd7cf3f8f3ef9cb86.gif"
    return None


TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Plant Disease Classifier</title>
    <style>
      :root {
        --bg1: #ff7a7a;
        --bg2: #ffb86c;
        --card: #ffffff;
        --text: #0f172a;
        --muted: #64748b;
        --border: #e5e7eb;
        --brand: #ff4d6d;
        --brand-2: #7b2ff7;
        --brand-3: #00c2ff;
        --brand-4: #22c55e;
      }
      @media (prefers-color-scheme: dark) {
        :root {
          --card: #0b1220;
          --text: #e2e8f0;
          --muted: #94a3b8;
          --border: #1f2937;
        }
      }

      body {
        margin: 0; padding: 0; color: var(--text);
        background: linear-gradient(125deg, rgba(255,122,122,.22), rgba(255,184,108,.22), rgba(0,194,255,.22), rgba(123,47,247,.22));
        background-size: 300% 300%;
        animation: bgShift 14s ease infinite;
        min-height: 100vh;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      }
      @keyframes bgShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
      .container { max-width: 1080px; margin: 0 auto; padding: 28px; }

      .navbar {
        display:flex; align-items:center; justify-content:center;
        padding: 16px 18px; border: 1px solid var(--border);
        border-radius: 14px; backdrop-filter: blur(8px);
        background: linear-gradient(180deg, rgba(255,255,255,.65), rgba(255,255,255,.45));
        box-shadow: 0 10px 30px rgba(2,6,23,.08);
      }
      @media (prefers-color-scheme: dark) {
        .navbar { background: linear-gradient(180deg, rgba(2,6,23,.6), rgba(2,6,23,.4)); }
      }
      .brand { display:flex; align-items:center; gap: 12px; font-weight: 900; letter-spacing: .2px; }
      .logo { width: 34px; height: 34px; display:grid; place-items:center; border-radius: 10px; color:white; background: conic-gradient(from 180deg, var(--brand), var(--brand-2), var(--brand-3), var(--brand-4), var(--brand)); box-shadow: 0 8px 18px rgba(255,77,109,.28); }
      .brand-title { font-size: 24px; }
      .device { display:none; }

      .grid { display: grid; grid-template-columns: 1.1fr .9fr; gap: 24px; margin-top: 22px; }
      @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }

      .card {
        background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 18px 18px 20px;
        box-shadow: 0 10px 25px rgba(2,6,23,.06);
        transition: transform .18s ease, box-shadow .18s ease;
      }
      .card:hover { transform: translateY(-2px); box-shadow: 0 16px 38px rgba(2,6,23,.10); }

      .card-title { display:flex; align-items:center; justify-content:space-between; margin: 6px 2px 14px; }
      .pill { background: linear-gradient(135deg, #ffd1dc, #ffe1a8, #c2f5ff); color: #1f2937; padding: 6px 12px; border-radius: 9999px; font-weight: 800; font-size: 13px; }
      @media (prefers-color-scheme: dark) { .pill { background: linear-gradient(135deg, #5b21b6, #1d4ed8, #06b6d4); color: #f8fafc; } }

      .btn {
        display:inline-block; padding: 12px 18px; border: none; border-radius: 12px; color:white; cursor:pointer; font-weight:700;
        background: linear-gradient(135deg, var(--brand), var(--brand-2), var(--brand-3));
        box-shadow: 0 10px 22px rgba(255,77,109,.35);
        transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
      }
      .btn:hover { transform: translateY(-1px); box-shadow: 0 16px 36px rgba(255,77,109,.45); filter: brightness(1.06); }
      .btn:active { transform: translateY(0); }
      .muted { color: var(--muted); }

      .dropzone {
        display:flex; gap: 12px; align-items:center; justify-content:space-between;
        padding: 14px; border: 1.5px dashed var(--border); border-radius: 14px; background: rgba(148,163,184,.08);
      }
      input[type=file] { background: transparent; border: none; color: var(--text); }

      table { width: 100%; border-collapse: collapse; }
      th, td { text-align: left; padding: 10px 10px; border-bottom: 1px solid var(--border); }
      th { background: rgba(148,163,184,.10); }

      .bar { height: 12px; background: #e2e8f0; border-radius: 9999px; overflow: hidden; }
      .bar > span { display:block; height:100%; background: repeating-linear-gradient(45deg, rgba(255,255,255,.35) 0 8px, rgba(255,255,255,.1) 8px 16px), linear-gradient(90deg, var(--brand), var(--brand-2), var(--brand-3)); box-shadow: inset 0 0 6px rgba(0,0,0,.12); background-size: 200% 100%, 100% 100%; animation: shimmer 2.4s linear infinite; }
      @keyframes shimmer { 0% { background-position: 0 0, 0 0; } 100% { background-position: 200% 0, 0 0; } }
      @media (prefers-color-scheme: dark) { .bar { background: #1f2937; } }

      /* Accent gradients per row for extra vibrance */
      .prob-row.r1 .bar > span { background: linear-gradient(90deg, #ff4d6d, #ff9f1c); }
      .prob-row.r2 .bar > span { background: linear-gradient(90deg, #10b981, #84cc16); }
      .prob-row.r3 .bar > span { background: linear-gradient(90deg, #06b6d4, #3b82f6); }
      .prob-row.r4 .bar > span { background: linear-gradient(90deg, #a78bfa, #f472b6); }
      .prob-row.r5 .bar > span { background: linear-gradient(90deg, #22c55e, #14b8a6); }
      .prob-row.r6 .bar > span { background: linear-gradient(90deg, #f59e0b, #ef4444); }
      .prob-row.r7 .bar > span { background: linear-gradient(90deg, #9333ea, #2563eb); }

      img { max-width: 100%; height: auto; border-radius: 12px; border: 1px solid var(--border); }
      .footer { margin-top: 18px; color: var(--muted); font-size: 14px; }

      /* Corner logo */
      .corner-logo { position: fixed; top: 14px; right: 14px; width: 108px; height: auto; z-index: 50; filter: drop-shadow(0 6px 14px rgba(0,0,0,.15)); opacity: .95; transition: transform .2s ease; }
      .corner-logo:hover { transform: scale(1.03) rotate(-1deg); }

      /* Fireworks canvas */
      #fx { position: fixed; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 40; }

      /* Right-side AI animated panel */
      .ai-panel { position: fixed; right: 14px; top: 50%; transform: translateY(-50%); width: 240px; height: 360px; z-index: 45; border-radius: 16px; border: none; overflow: hidden; background: transparent; box-shadow: none; backdrop-filter: none; }
      @media (prefers-color-scheme: dark) { .ai-panel { background: transparent; } }
      .ai-panel .slides { position: relative; width: 100%; height: 100%; }
      .ai-panel img.slide { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0; animation: slideFade 12s linear infinite; border: none !important; box-shadow: none !important; }
      /* delay per slide */
      .ai-panel img.slide.d1 { animation-delay: 0s; }
      .ai-panel img.slide.d2 { animation-delay: 4s; }
      .ai-panel img.slide.d3 { animation-delay: 8s; }
      .ai-panel img.slide.d4 { animation-delay: 12s; }
      .ai-panel img.slide.d5 { animation-delay: 16s; }
      @keyframes slideFade { 0%{opacity:0} 5%{opacity:1} 30%{opacity:1} 35%{opacity:0} 100%{opacity:0} }
      /* caption + hearts in AI panel */
      .ai-caption { position:absolute; left:10px; bottom:12px; color:#111111; font-weight:900; text-shadow: 0 1px 2px rgba(255,255,255,.6); letter-spacing:.3px; }
      .ai-hearts { position:absolute; right:10px; bottom:10px; display:flex; gap:6px; pointer-events:none; }
      .ai-hearts .heart { filter: drop-shadow(0 2px 6px rgba(0,0,0,.35)); animation: floatUp 2.8s ease-in-out infinite; }
      .ai-hearts .h1 { font-size:18px; color:#ff4d6d; animation-delay: .0s; }
      .ai-hearts .h2 { font-size:22px; color:#ff758f; animation-delay: .3s; }
      .ai-hearts .h3 { font-size:16px; color:#ff8fab; animation-delay: .6s; }
      .ai-hearts .h4 { font-size:20px; color:#ff4d6d; animation-delay: .9s; }
      .ai-hearts .h5 { font-size:14px; color:#ffb3c1; animation-delay: 1.2s; }
      @keyframes floatUp { 0% { transform: translateY(0) scale(1); opacity: .95; } 70% { opacity:.9; } 100% { transform: translateY(-24px) scale(1.12); opacity: .0; } }

      /* Left-side fixed panel */
      .ai-panel-left { position: fixed; left: 14px; top: 50%; transform: translateY(-50%); width: 200px; height: 300px; z-index: 45; border-radius: 16px; border: none; overflow: hidden; background: transparent; box-shadow: none; backdrop-filter: none; }
      .ai-panel-left img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; border: none; }
      .ai-left-caption { position:absolute; left:10px; bottom:10px; right:10px; color:#111111; font-weight:900; text-shadow: 0 1px 2px rgba(255,255,255,.6); letter-spacing:.3px; }
      .ai-left-hearts { position:absolute; left:10px; top:10px; display:flex; gap:6px; pointer-events:none; }
      .ai-left-hearts .heart { filter: drop-shadow(0 2px 6px rgba(0,0,0,.25)); animation: crackFloat 2.6s ease-in-out infinite; }
      .ai-left-hearts .h1 { font-size:16px; color:#ef4444; animation-delay: .0s; }
      .ai-left-hearts .h2 { font-size:20px; color:#f87171; animation-delay: .3s; }
      .ai-left-hearts .h3 { font-size:14px; color:#fb7185; animation-delay: .6s; }
      @keyframes crackFloat { 0% { transform: translateY(0) scale(1); opacity:.95; } 60% { opacity:.9; } 100% { transform: translateY(-18px) scale(1.08) rotate(-6deg); opacity:0; } }

      /* Hide side GIF panels */
      .ai-panel, .ai-panel-left { display: none !important; }
    </style>
  </head>
  <body>
    {% if logo_filename %}
    <img class="corner-logo" src="{{ url_for('static', filename=logo_filename) }}" alt="Logo" />
    {% endif %}
    <canvas id="fx"></canvas>
    <div class="container">
      <div class="navbar">
        <div class="brand">
          <div class="logo">🌿</div>
          <div class="brand-title">Plant Disease Classifier</div>
        </div>
      </div>

      <div class="grid">
        <div class="card">
          <div class="card-title">
            <h3 style="margin:0;">Upload</h3>
          </div>
          <form method="post" action="{{ url_for('predict_route') }}" enctype="multipart/form-data">
            <div class="dropzone">
              <div>
                <div style="font-weight:700;">Choose an image</div>
                <div class="muted" style="font-size: 14px;">Supported: jpg, jpeg, png, webp</div>
              </div>
              <input type="file" name="image" accept="image/*" required />
            </div>
            <div style="display:flex; gap: 12px; align-items:center; margin-top: 14px;">
              <button class="btn" id="predictBtn" type="submit">Predict</button>
              <span class="muted">The model will return class and probabilities.</span>
            </div>
          </form>
        </div>

        <div class="card">
          <div class="card-title">
            <h3 style="margin:0;">Preview</h3>
          </div>
          {% if filename %}
            <img id="previewImg" src="{{ url_for('uploaded_file', filename=filename) }}" alt="Uploaded image" />
            <div class="footer">File: {{ filename }}</div>
          {% else %}
            <div class="muted">No image uploaded yet.</div>
          {% endif %}
        </div>
      </div>

      {% if filename %}
      <div class="card" style="margin-top: 22px;">
        <div class="card-title">
          <h3 style="margin:0;">Result</h3>
          <span class="pill">{{ predicted }} ({{ confidence_percent }}%)</span>
        </div>
        <table>
          <thead>
            <tr><th>Label</th><th style="width: 60%;">Probability</th><th style="text-align:right;">Score</th></tr>
          </thead>
          <tbody>
            {% for label, prob in prob_rows %}
              <tr class="prob-row r{{ loop.index }}">
                <td>{{ label }}</td>
                <td>
                  <div class="bar"><span style="width: {{ (prob*100)|round(2) }}%"></span></div>
                </td>
                <td style="text-align:right; white-space:nowrap;">{{ "%.4f"|format(prob) }} ({{ "%.2f"|format(prob*100) }}%)</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% endif %}

      <div class="footer">Labels: {{ labels|join(", ") }}</div>
    </div>

    <!-- Left-side specific GIF panel (mirrors right) -->
    {% if left_gif %}
    <div class="ai-panel-left">
      <img src="{{ url_for('static', filename=left_gif) }}" alt="left visual" />
      <div class="ai-left-hearts" aria-hidden="true">
        <span class="heart h1">💔</span>
        <span class="heart h2">💔</span>
        <span class="heart h3">💔</span>
      </div>
      <div class="ai-left-caption">Hông Nhận Đâu !!!</div>
    </div>
    {% endif %}

    <!-- Right-side AI panel: fixed GIF if present, else slideshow/fallback -->
    <div class="ai-panel">
      <div class="slides">
        {% if right_gif %}
          <img class="slide d1" src="{{ url_for('static', filename=right_gif) }}" alt="right visual" style="opacity:1; animation:none;" />
        {% elif ai_images and ai_images|length > 0 %}
          {% for img in ai_images %}
            <img class="slide d{{ loop.index }}" src="{{ url_for('static', filename=img) }}" alt="AI visual {{ loop.index }}" />
          {% endfor %}
        {% else %}
          <!-- Fallback animated SVG (neural network style) -->
          <svg viewBox="0 0 300 450" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
            <defs>
              <linearGradient id="g1" x1="0" x2="1" y1="0" y2="1">
                <stop offset="0%" stop-color="#ff4d6d"/>
                <stop offset="50%" stop-color="#7b2ff7"/>
                <stop offset="100%" stop-color="#00c2ff"/>
              </linearGradient>
              <filter id="glow"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
            </defs>
            <rect width="100%" height="100%" fill="url(#g1)" opacity="0.08"/>
            <!-- moving nodes -->
            <g stroke="url(#g1)" stroke-width="2" filter="url(#glow)">
              <circle cx="60" cy="60" r="5">
                <animate attributeName="cy" values="60;90;60" dur="4s" repeatCount="indefinite"/>
              </circle>
              <circle cx="240" cy="100" r="5">
                <animate attributeName="cy" values="100;140;100" dur="5s" repeatCount="indefinite"/>
              </circle>
              <circle cx="150" cy="200" r="5">
                <animate attributeName="cy" values="200;230;200" dur="3.2s" repeatCount="indefinite"/>
              </circle>
              <circle cx="90" cy="300" r="5">
                <animate attributeName="cy" values="300;340;300" dur="4.4s" repeatCount="indefinite"/>
              </circle>
              <circle cx="210" cy="360" r="5">
                <animate attributeName="cy" values="360;400;360" dur="3.6s" repeatCount="indefinite"/>
              </circle>
              <!-- connecting animated paths -->
              <path d="M60 60 C 120 120, 180 80, 240 100" fill="none">
                <animate attributeName="d" dur="6s" repeatCount="indefinite"
                  values="M60 60 C 120 120, 180 80, 240 100; M60 80 C 120 140, 180 100, 240 120; M60 60 C 120 120, 180 80, 240 100"/>
              </path>
              <path d="M150 200 C 110 260, 120 320, 90 300" fill="none">
                <animate attributeName="d" dur="5s" repeatCount="indefinite"
                  values="M150 200 C 110 260, 120 320, 90 300; M150 220 C 110 280, 120 340, 90 320; M150 200 C 110 260, 120 320, 90 300"/>
              </path>
              <path d="M150 200 C 200 260, 230 330, 210 360" fill="none">
                <animate attributeName="d" dur="4.5s" repeatCount="indefinite"
                  values="M150 200 C 200 260, 230 330, 210 360; M150 220 C 200 280, 230 350, 210 380; M150 200 C 200 260, 230 330, 210 360"/>
              </path>
            </g>
            <text x="16" y="430" fill="#ffffff" fill-opacity="0.9" font-size="14" font-weight="700">AI Visualization</text>
          </svg>
        {% endif %}
        <div class="ai-caption">Bắn Tim Nè !!!!</div>
        <div class="ai-hearts" aria-hidden="true">
          <span class="heart h1">❤️</span>
          <span class="heart h2">❤️</span>
          <span class="heart h3">❤️</span>
          <span class="heart h4">❤️</span>
          <span class="heart h5">❤️</span>
        </div>
      </div>
    </div>

    <script>
      // Simple fireworks / confetti effect without external libs
      (function() {
        const canvas = document.getElementById('fx');
        const ctx = canvas.getContext('2d');
        let particles = [];
        let rafId = null;

        function resize() {
          const dpr = window.devicePixelRatio || 1;
          canvas.width = Math.floor(innerWidth * dpr);
          canvas.height = Math.floor(innerHeight * dpr);
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        window.addEventListener('resize', resize, { passive: true });
        resize();

        function spawn(x, y, count = 120) {
          const colors = ['#ff4d6d','#ff9f1c','#22c55e','#06b6d4','#3b82f6','#a78bfa','#f472b6','#f59e0b'];
          for (let i = 0; i < count; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = Math.random() * 4 + 2;
            particles.push({
              x, y,
              vx: Math.cos(angle) * speed,
              vy: Math.sin(angle) * speed,
              life: 180 + Math.random() * 100,
              maxLife: 260 + Math.random() * 80,
              size: 2 + Math.random() * 3,
              color: colors[(Math.random() * colors.length) | 0],
              g: 0.10 + Math.random() * 0.08,
              rotation: Math.random() * 360,
              vr: (Math.random() - 0.5) * 12,
            });
          }
          if (!rafId) rafId = requestAnimationFrame(tick);
        }

        function tick() {
          rafId = null;
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          particles.forEach(p => {
            p.life -= 1;
            p.vy += p.g * 0.85;  // gentle gravity
            p.vx *= 0.992;       // slight air drag
            p.vy *= 0.992;
            p.x += p.vx;
            p.y += p.vy;
            p.rotation += p.vr * 0.985;
          });
          particles = particles.filter(p => p.life > -120 && p.y < innerHeight + 120);

          particles.forEach(p => {
            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(p.rotation * Math.PI / 180);
            const alpha = Math.max(0, Math.min(1, (p.life + 120) / (p.maxLife || 300)));
            ctx.globalAlpha = alpha;
            ctx.fillStyle = p.color;
            const s = p.size;
            ctx.fillRect(-s, -s, s * 2, s * 2);
            ctx.globalAlpha = 1;
            ctx.restore();
          });
          if (particles.length) rafId = requestAnimationFrame(tick);
        }

        function centerBurst(times = 3) {
          const cx = innerWidth / 2;
          const cy = Math.min(innerHeight * 0.35, 360);
          for (let i = 0; i < times; i++) {
            setTimeout(() => spawn(cx, cy, 140), i * 180);
          }
        }

        // Fire on Predict click (does not block submit)
        const btn = document.getElementById('predictBtn');
        if (btn) {
          btn.addEventListener('click', function() {
            // Center fireworks with a few staggered bursts for longer effect
            const x = innerWidth / 2;
            const y = innerHeight / 2;
            spawn(x, y, 160);
            setTimeout(() => spawn(x, y, 120), 180);
            setTimeout(() => spawn(x, y, 100), 360);
          });
        }

        // Fireworks synchronized with preview image appearing (after prediction)
        const preview = document.getElementById('previewImg');
        function celebrateWithImage() {
          const x = innerWidth / 2;
          const y = innerHeight / 2;
          spawn(x, y, 160);
          setTimeout(() => spawn(x, y, 120), 180);
          setTimeout(() => spawn(x, y, 100), 360);
        }
        if (preview) {
          if (preview.complete) {
            // Image already in cache/rendered
            setTimeout(celebrateWithImage, 60);
          } else {
            preview.addEventListener('load', () => setTimeout(celebrateWithImage, 60), { once: true });
          }
        }
      })();
    </script>
  </body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        TEMPLATE,
        filename=None,
        predicted=None,
        confidence_percent=None,
        prob_rows=None,
        labels=LABELS,
        device_str=str(device),
        logo_filename=find_logo(),
        ai_images=list_ai_images(),
        left_gif=left_gif(),
        right_gif=right_gif(),
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return redirect(url_for("index"))
    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    # Save uploaded file for preview
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    file.stream.seek(0)
    file.save(filepath)

    # Open and predict
    with Image.open(filepath) as pil_img:
        pred_label, probs = predict(pil_img)

    prob_rows = list(zip(LABELS, probs.tolist()))
    confidence_percent = round(float(max(probs) * 100.0), 2)

    return render_template_string(
        TEMPLATE,
        filename=file.filename,
        predicted=pred_label,
        confidence_percent=confidence_percent,
        prob_rows=prob_rows,
        labels=LABELS,
        device_str=str(device),
        logo_filename=find_logo(),
        ai_images=list_ai_images(),
        left_gif=left_gif(),
        right_gif=right_gif(),
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 2003))
    app.run(host="0.0.0.0", port=port, debug=False)
