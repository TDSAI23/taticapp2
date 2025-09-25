#!/usr/bin/env bash
set -euo pipefail

# --- Kill old ComfyUI processes (if running) ---
pkill -9 -f "ComfyUI/main.py" 2>/dev/null || true
pkill -9 -f "python.*ComfyUI/main.py" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
pkill -9 -f "cloudflared" 2>/dev/null || true
sleep 1

BASE="/workspace/runpod-slim/ComfyUI"
VENV="$BASE/.venv"
VENVPY="$VENV/bin/python"
PIP="$VENVPY -m pip"
LOG="/workspace/runpod-slim/comfyui.log"

# --- Ensure git is available (best-effort) ---
if ! command -v git >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  apt-get update -y || true
  apt-get install -y git || true
fi

# --- Ensure ComfyUI repo is present (don’t skip just because $BASE exists) ---
if [ ! -d "$BASE/.git" ]; then
  rm -rf "$BASE"
  git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git "$BASE"
fi

# --- Ensure venv exists ---
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi

# --- Base Python tooling + ComfyUI requirements ---
$PIP install --upgrade pip setuptools wheel
if [ -f "$BASE/requirements.txt" ]; then
  $PIP install --no-input -r "$BASE/requirements.txt"
fi

# --- Ensure custom_nodes dir exists ---
mkdir -p "$BASE/custom_nodes"

# --- Custom nodes (idempotent clones + per-node requirements) ---
cd "$BASE/custom_nodes" || { echo "ERROR: custom_nodes missing"; exit 1; }
clone_if_absent(){ [ -d "$2" ] && echo "✓ $2 exists" || git clone --depth=1 "$1" "$2"; }

clone_if_absent https://github.com/Fannovel16/comfyui_controlnet_aux.git comfyui_controlnet_aux
clone_if_absent https://github.com/WASasquatch/was-node-suite-comfyui.git ComfyUI-WAS-NodeSuite
clone_if_absent https://github.com/ltdrdata/ComfyUI-Impact-Pack.git ComfyUI-Impact-Pack
clone_if_absent https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git ComfyUI-Advanced-ControlNet
clone_if_absent https://github.com/yolain/ComfyUI-Easy-Use.git ComfyUI-Easy-Use
clone_if_absent https://github.com/ltdrdata/ComfyUI-Manager.git ComfyUI-Manager
clone_if_absent https://github.com/cubiq/ComfyUI_IPAdapter_plus.git ComfyUI_IPAdapter_plus

# Remove KJNodes (conflicts / not needed)
rm -rf "$BASE/custom_nodes/ComfyUI-KJNodes" || true

# --- Native build deps needed by some custom nodes (pycairo/rlpycairo/svglib) ---
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -y || true
  apt-get install -y \
    build-essential \
    pkg-config \
    libcairo2-dev \
    libffi-dev \
    python3-dev \
    meson \
    ninja-build || true
fi

# Per-node Python requirements (best effort)
for d in "$BASE/custom_nodes"/*; do
  [ -f "$d/requirements.txt" ] && $PIP install --no-input -r "$d/requirements.txt" || true
done

# 4) Models
# ---- Install git-lfs (best-effort, some images restrict apt) ----
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -y || true
  apt-get install -y git-lfs || true
fi
git lfs install || true

# ---- (Docs only) Clone the HF repo OUTSIDE models ----
mkdir -p "$BASE/_hf"
if [ ! -d "$BASE/_hf/Qwen-Image-Edit-2509" ]; then
  git clone https://huggingface.co/Qwen/Qwen-Image-Edit-2509 "$BASE/_hf/Qwen-Image-Edit-2509" || true
fi
if [ -d "$BASE/_hf/Qwen-Image-Edit-2509/transformer" ]; then
  ls -lh "$BASE/_hf/Qwen-Image-Edit-2509/transformer" | head -n 10 || true
fi

# ---- Download assets INTO $BASE/models/** (not inside HF repo) ----
# RealVisXL v4.0
[ -f "$BASE/models/checkpoints/RealVisXL_V4.0.safetensors" ] || \
curl -C - -L -o "$BASE/models/checkpoints/RealVisXL_V4.0.safetensors" \
  "https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors"

# RealVisXL v4.0 VAE
[ -f "$BASE/models/vae/RealVisXL_V4.0_vae.safetensors" ] || \
curl -C - -L -o "$BASE/models/vae/RealVisXL_V4.0_vae.safetensors" \
  "https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/vae/diffusion_pytorch_model.safetensors"

# SwinIR Upscaler
[ -f "$BASE/models/upscale_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth" ] || \
curl -C - -L -o "$BASE/models/upscale_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth" \
  "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"

# Qwen Lightning LoRA
[ -f "$BASE/models/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors" ] || \
curl -C - -L -o "$BASE/models/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors" \
  "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors"

# 5) Write app.py (first part) - includes CN fix: strength/start/end on Apply (not Loader)
APP_DIR="$BASE/simple-ui"; mkdir -p "$APP_DIR/uploads"
cat >"$APP_DIR/app.py" <<'PY'
import os, json, time, uuid, http.client, urllib.parse, asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
import websockets
import cv2
import numpy as np

BASE = "/workspace/runpod-slim/ComfyUI"
OUT_DIR = os.path.join(BASE, "output")
HIST_FILE = os.path.join(BASE, "simple-ui", "history.json")
UPLOAD_DIR = os.path.join(BASE, "simple-ui", "uploads")

CHECKPOINTS_DIR = os.path.join(BASE, "models", "checkpoints")
VAE_DIR         = os.path.join(BASE, "models", "vae")
UPSCALE_DIR     = os.path.join(BASE, "models", "upscale_models")
CONTROLNET_DIR  = os.path.join(BASE, "models", "controlnet")
DIFFUSION_DIR   = os.path.join(BASE, "models", "diffusion_models")

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188

def list_files(path: str, exts: tuple) -> List[str]:
    try:
        return sorted([x for x in os.listdir(path) if x.lower().endswith(exts)])
    except Exception:
        return []

def comfy_request(method: str, path: str, data=None, timeout=600, qs=None):
    q = urllib.parse.urlencode(qs or {}, doseq=True)
    url = path + (("?" + q) if q else "")
    conn = http.client.HTTPConnection(COMFY_HOST, COMFY_PORT, timeout=timeout)
    try:
        body = json.dumps(data) if data is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        conn.request(method, url, body, headers)
        resp = conn.getresponse()
        raw  = resp.read()
        ctype = resp.getheader("Content-Type") or ""
        if "application/json" in ctype:
            try:
                return resp.status, json.loads(raw)
            except Exception:
                try:
                    return resp.status, {"raw": raw.decode("utf-8","ignore")}
                except Exception:
                    return resp.status, {"raw_bytes": True}
        return resp.status, raw
    finally:
        conn.close()

def _object_info_nodes():
    try:
        s, data = comfy_request("GET", "/object_info")
        if s == 200 and isinstance(data, dict):
            return data.get("nodes", {}) or {}
    except Exception:
        pass
    return {}

def pick_qwen_union_class() -> str:
    nodes = _object_info_nodes()
    for name in ("QwenImageDiffsynthControlnet","QwenImageDiffSynthControlnet","QwenImageDiffsynthControlNet"):
        if name in nodes: return name
    return "QwenImageDiffsynthControlnet"

def qwen_union_returns_model(class_name: str) -> bool:
    nodes = _object_info_nodes()
    info = nodes.get(class_name, {})
    outs = info.get("output", info.get("return_types", {}))
    try:
        return "MODEL" in json.dumps(outs).upper()
    except Exception:
        return False

# --- Preprocessor (Canny) helper ---
def add_canny_pre(graph: dict, image_src_id: str, low: int, high: int) -> str:
    low  = max(0, min(255, int(low)))
    high = max(0, min(255, int(high)))
    if low > high: low, high = high, low
    nodes = _object_info_nodes() or {}
    for klass in ("CannyEdgePreprocessor","CannyPreprocessor","Canny","CannyEdgeDetector"):
        if klass in nodes:
            info = nodes[klass] or {}
            ins  = (info.get("input") or info.get("inputs") or {}) if isinstance(info, dict) else {}
            def pick(pref, falls):
                for k in (pref, *falls):
                    if k in ins: return k
                return pref
            img_f  = pick("image", ["images","img"])
            low_f  = pick("low_threshold", ["low","threshold_low","canny_low"])
            high_f = pick("high_threshold", ["high","threshold_high","canny_high"])
            next_id = str(max([int(i) for i in graph.keys() if i.isdigit()] + [0]) + 1)
            graph[next_id] = {"class_type": klass, "inputs": {img_f:[image_src_id,0], low_f:low, high_f:high}}
            return next_id
    return image_src_id

def add_qwen2509_builtin_controlnet(
    g: dict,
    *,
    vae_id: str,
    cn_image_id: str,
    ksampler_id: str,
    positive_cond_id: str,
    cn_mode: str,
    cn_strength: float = 1.0,
    cn_start: float = 0.0,
    cn_end: float = 1.0,
):
    # built-in modes
    mode_map = {"edges":"edges","depth":"depth","keypoints":"keypoints"}
    key = (cn_mode or "none").lower().strip()
    if key not in mode_map: return
    builtin_name = mode_map[key]

    base_id = max([int(i) for i in g.keys() if i.isdigit()] + [0]) + 1
    cn_loader_id  = str(base_id)
    cn_apply_id   = str(base_id + 1)
    cn_preview_id = str(base_id + 2)

    # Loader (ONLY name)
    g[cn_loader_id] = {
        "class_type": "ControlNetLoaderAdvanced",
        "inputs": { "control_net_name": builtin_name },
    }

    # preview raw CN image
    g[cn_preview_id] = {
        "class_type": "SaveImage",
        "inputs": {"images": [cn_image_id, 0], "filename_prefix": "cn_preview"}
    }

    # Apply (puts strength/start/end HERE)
    g[cn_apply_id] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {
            "positive":      [positive_cond_id, 0],
            "control_net":   [cn_loader_id, 0],
            "image":         [cn_image_id, 0],
            "vae":           [vae_id, 0],
            "strength":      float(cn_strength),
            "start_percent": float(cn_start),
            "end_percent":   float(cn_end),
        },
    }

    # wire into sampler
    g[ksampler_id]["inputs"]["positive"] = [cn_apply_id, 0]

# -------- Helpers: history load/save --------

def _ensure_dir(p: str):
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    except Exception:
        pass


def load_hist() -> list:
    _ensure_dir(HIST_FILE)
    try:
        if os.path.isfile(HIST_FILE):
            with open(HIST_FILE, "r", encoding="utf-8") as r:
                return json.load(r) or []
    except Exception:
        pass
    return []


def add_hist(rec: dict) -> None:
    items = load_hist()
    items.append(rec)
    try:
        with open(HIST_FILE, "w", encoding="utf-8") as w:
            json.dump(items, w, ensure_ascii=False, indent=2)
    except Exception:
        pass


# -------- Upscale workflow (SwinIR/ESRGAN class) --------

def build_upscale(path: str, model_name: str, *, prefix: str = "upscaled") -> Dict[str, Any]:
    """
    Simple pixel-space upscaler using a .pth model in models/upscale_models.
    - model_name: filename in UPSCALE_DIR (e.g., SwinIR .pth)
    - path: absolute path to input image (we copy uploads into ComfyUI/input already)
    """
    # Try to resolve to a relative name that Comfy's LoadImage can read
    # If path is already inside BASE/input, keep its basename.
    rel_name = os.path.basename(path)
    if not os.path.isfile(os.path.join(BASE, "input", rel_name)):
        # best effort copy for Comfy LoadImage
        try:
            os.makedirs(os.path.join(BASE, "input"), exist_ok=True)
            with open(path, "rb") as r, open(os.path.join(BASE, "input", rel_name), "wb") as w:
                w.write(r.read())
        except Exception:
            pass

    g = {
        # Load pixels
        "1": {"class_type": "LoadImage", "inputs": {"image": rel_name}},
        # Load upscaler model
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}},
        # Apply upscaler
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"image": ["1", 0], "upscale_model": ["2", 0]}},
        # Save
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": prefix}},
    }
    return {"prompt": g}


# -------- Root page (optional) --------
@app.get("/")
def root():
    return {"ok": True, "service": "Simple ComfyUI UI", "models": {
        "checkpoints": len(list_files(CHECKPOINTS_DIR, (".safetensors", ".ckpt"))),
        "vae": len(list_files(VAE_DIR, (".safetensors", ".ckpt"))),
        "upscalers": len(list_files(UPSCALE_DIR, (".pth",))),
        "unets": len(list_files(DIFFUSION_DIR, (".safetensors", ".ckpt", ".bin"))),
    }}
# ---------------- UI ----------------
# Updated to: (1) fix built-in CN model dropdown bug (no external model list needed),
# (2) make numeric fields use <input type="number"> with sensible step/min,
# (3) add a /ui route that serves this HTML directly,
# (4) tighten progress/log handling, (5) small UX polish.

HTML = '''<!doctype html><html><head><meta charset="utf-8"/>
<title>Simple ComfyUI WebUI</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  :root{
    --bg:#0e1015;--card:#141821;--muted:#8ea0b5;--text:#e8edf2;--accent:#2c7be5;--accent-2:#7aa2f7;
    --line:#222a36;--ok:#22c55e;--warn:#f59e0b;--err:#ef4444;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto}
  header{position:sticky;top:0;z-index:2;background:var(--card);border-bottom:1px solid var(--line);padding:14px 18px}
  h1{margin:0;font-size:18px;letter-spacing:.3px}
  main{padding:18px;max-width:1400px;margin:0 auto}
  .layout{display:grid;grid-template-columns:300px 1fr;gap:14px}
  .tabs{display:flex;gap:8px;margin:0 0 14px}
  .tabs button{background:var(--card);border:1px solid var(--line);color:var(--text);padding:10px 14px;border-radius:10px;cursor:pointer}
  .tabs button.active{background:var(--accent);border-color:var(--accent)}
  .grid{display:grid;gap:12px}
  .g2{grid-template-columns:1fr 1fr}
  label{opacity:.9;margin:6px 0 6px;display:block}
  input,select,textarea{width:100%;padding:10px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--text)}
  input[type=number]{appearance:textfield}
  input[type=number]::-webkit-outer-spin-button,input[type=number]::-webkit-inner-spin-button{appearance:none;margin:0}
  textarea{min-height:80px}
  .card{border:1px solid var(--line);border-radius:14px;background:var(--card);padding:14px}
  .stack{display:flex;flex-direction:column;gap:10px}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  .btn{padding:10px 14px;border-radius:10px;border:1px solid var(--line);background:#1e2430;color:var(--text);cursor:pointer;transition:transform .06s ease, opacity .2s}
  .btn[disabled]{opacity:.6;cursor:not-allowed}
  .btn.pressed{transform:translateY(1px)}
  .btn.pri{background:var(--accent);border-color:var(--accent);color:white}
  .thumbs img{max-width:256px;border-radius:10px;border:1px solid var(--line);margin:6px}
  #i_out, #r_out, #u_out {max-height:260px;overflow:auto;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;white-space:pre-wrap;word-break:break-word;overflow-wrap:anywhere;padding:8px;border-radius:8px}
  #uilog { max-height: 420px; overflow: auto; overflow-wrap: anywhere; }
  .progress{height:10px;background:#101420;border:1px solid var(--line);border-radius:999px;overflow:hidden}
  .progress > div{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent-2));width:0%}
  .muted{color:var(--muted)}
  .pill{display:inline-block;padding:2px 8px;border:1px solid var(--line);border-radius:999px;background:#1b2230}
  .notice{margin-top:6px;font-size:14px}
  .ok{color:#22c55e} .err{color:#ef4444}
  .logline{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#cbd5e1;border-bottom:1px dotted #243041;padding:6px 0}
  .logline b{color:#93c5fd}
  @media (max-width:1000px){.layout{grid-template-columns:1fr}.g2{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><h1>Simple ComfyUI WebUI</h1></header>
<main>
  <div class="layout">
    <aside class="card stack" style="min-height:120px">
      <div class="row">
        <b>UI Logs</b>
        <button class="btn" onclick="clearLogs()" style="margin-left:auto">Clear</button>
      </div>
      <div id="uilog"></div>
    </aside>

    <section>
      <div class="tabs">
        <button id="tabIdeate"  class="active" onclick="tab('ideate')">Ideate</button>
        <button id="tabRender"               onclick="tab('render')">Render</button>
        <button id="tabUpscale"              onclick="tab('upscale')">Upscale</button>
        <button id="tabHistory"              onclick="tab('history')">History</button>
      </div>

      <div class="card stack" id="progressCard" style="display:none">
        <div class="row" style="gap:12px">
          <div class="pill" id="stagePill">idle</div>
          <div class="progress" style="flex:1"><div id="bar"></div></div>
          <div id="pct" class="muted">0%</div>
        </div>
        <div id="previewWrap" class="thumbs"></div>
      </div>

      <!-- IDEATE (SDXL) -->
      <section id="pane_ideate">
        <div class="grid g2">
          <div class="card">
            <label>Prompt</label><textarea id="i_prompt"></textarea>
            <label>Negative</label><textarea id="i_negative"></textarea>
            <div class="grid g2">
              <div><label>Steps</label><input id="i_steps" type="number" value="20" min="1" max="200" step="1"></div>
              <div><label>CFG</label><input id="i_cfg" type="number" value="6.0" step="0.1" min="0"></div>
            </div>
            <div class="grid g2">
              <div><label>Width</label><input id="i_w" type="number" value="1024" step="8" min="64" max="2048"></div>
              <div><label>Height</label><input id="i_h" type="number" value="1024" step="8" min="64" max="2048"></div>
            </div>
            <div class="grid g2">
              <div><label>Sampler</label><input id="i_sampler" value="euler"></div>
              <div><label>Scheduler</label><input id="i_sched" value="normal"></div>
            </div>
            <div class="grid g2">
              <div><label>Checkpoint</label><select id="i_ckpt"></select></div>
              <div><label>VAE (optional)</label><select id="i_vae"><option value="">(from ckpt)</option></select></div>
            </div>
            <div class="grid g2">
              <div><label>Filename prefix</label><input id="i_prefix" value="ideate_sdxl"></div>
              <div><label>Seed</label><input id="i_seed" type="number" value="0" step="1" min="0"></div>
            </div>
          </div>

          <div class="card">
            <h3 style="margin:0 0 8px">ControlNets</h3>
            <div class="grid g2">
              <div>
                <label>CN1 model</label>
                <select id="i_c1m"></select>
              </div>
              <div>
                <label>CN1 image path</label>
                <input id="i_c1img" placeholder="/path/to/c1.png">
                <div class="row" style="margin-top:6px">
                  <input type="file" id="i_c1file" accept="image/*">
                  <small id="i_c1note" class="muted"></small>
                </div>
                <div id="i_c1prev" class="thumbs"></div>
              </div>
              <div>
                <label>CN1 weight</label>
                <input id="i_c1w" type="number" value="1.0" step="0.05" min="0" max="2">
              </div>
              <div>
                <label>CN1 start / end</label>
                <input id="i_c1s" type="number" value="0.0" step="0.05" min="0" max="1">
                <input id="i_c1e" type="number" value="1.0" step="0.05" min="0" max="1">
              </div>
            </div>
            <div class="row" style="margin-top:6px; align-items:center; gap:8px">
              <label style="display:flex;align-items:center;gap:8px;margin:0">
                <input type="checkbox" id="i_c1pre"> Preprocess (Canny)
              </label>
              <label for="i_c1low" class="muted" style="margin:0">low</label>
              <input id="i_c1low" type="number" value="100" style="width:80px" min="0" max="255" step="1">
              <label for="i_c1high" class="muted" style="margin:0">high</label>
              <input id="i_c1high" type="number" value="200" style="width:80px" min="0" max="255" step="1">
            </div>
            <hr style="border:none;border-top:1px solid var(--line);margin:12px 0">
            <div class="grid g2">
              <div>
                <label>CN2 model</label>
                <select id="i_c2m"></select>
              </div>
              <div>
                <label>CN2 image path</label>
                <input id="i_c2img" placeholder="/path/to/c2.png">
                <div class="row" style="margin-top:6px">
                  <input type="file" id="i_c2file" accept="image/*">
                  <small id="i_c2note" class="muted"></small>
                </div>
                <div id="i_c2prev" class="thumbs"></div>
              </div>
              <div>
                <label>CN2 weight</label>
                <input id="i_c2w" type="number" value="1.0" step="0.05" min="0" max="2">
              </div>
              <div>
                <label>CN2 start / end</label>
                <input id="i_c2s" type="number" value="0.0" step="0.05" min="0" max="1">
                <input id="i_c2e" type="number" value="1.0" step="0.05" min="0" max="1">
              </div>
            </div>
            <div class="row" style="margin-top:6px; align-items:center; gap:8px">
              <label style="display:flex;align-items:center;gap:8px;margin:0">
                <input type="checkbox" id="i_c2pre"> Preprocess (Canny)
              </label>
              <label for="i_c2low" class="muted" style="margin:0">low</label>
              <input id="i_c2low" type="number" value="100" style="width:80px" min="0" max="255" step="1">
              <label for="i_c2high" class="muted" style="margin:0">high</label>
              <input id="i_c2high" type="number" value="200" style="width:80px" min="0" max="255" step="1">
            </div>
          </div>
        </div>
        <p>
          <button class="btn pri" id="btnIdeate">Generate (Ideate)</button>
          <div id="i_notice" class="notice"></div>
        </p>
        <div id="i_out" class="card"></div>
        <div id="i_imgs" class="thumbs"></div>
      </section>

      <!-- RENDER (Qwen Image Edit 2509) -->
      <section id="pane_render" style="display:none">
        <div class="grid g2">
          <div class="card">
            <label>Prompt</label><textarea id="r_prompt"></textarea>
            <label>Negative</label><textarea id="r_negative"></textarea>
            <div class="grid g2">
              <div><label>Steps</label><input id="r_steps" type="number" value="5" min="1" max="100" step="1"></div>
              <div><label>CFG</label><input id="r_cfg" type="number" value="1.0" step="0.1" min="0"></div>
            </div>
            <div class="grid g2">
              <div><label>Denoise</label><input id="r_denoise" type="number" value="0.4" step="0.05" min="0" max="1"></div>
              <div><label>UNET</label><select id="r_unet"></select></div>
            </div>
            <div class="grid g2">
              <div><label>LoRA #1</label><input id="r_l1" value="Qwen-Image-Lightning-4steps-V1.0.safetensors"></div>
              <div><label>Strength #1</label><input id="r_l1s" type="number" value="1.0" step="0.05" min="0" max="2"></div>
            </div>
            <div class="grid g2">
              <div><label>LoRA #2 (opt)</label><input id="r_l2"></div>
              <div><label>Strength #2</label><input id="r_l2s" type="number" value="0.0" step="0.05" min="0" max="2"></div>
            </div>
            <div class="grid g2">
              <div><label>Filename prefix</label><input id="r_prefix" value="render_qwen"></div>
              <div></div>
            </div>
          </div>

          <div class="card">
            <h3 style="margin:0 0 8px">Input image</h3>
            <div class="row" style="gap:10px">
              <input id="r_in" placeholder="/path/to/input.jpg">
              <input type="file" id="r_in_file" accept="image/*">
            </div>
            <small id="r_in_note" class="muted"></small>
            <div id="r_in_prev" class="thumbs"></div>

            <hr style="border:none;border-top:1px solid var(--line);margin:12px 0">

            <h3 style="margin:0 0 8px">ControlNet (built-in)</h3>
            <div class="grid g2">
              <div>
                <label>Type</label>
                <select id="r_cn_mode">
                  <option value="none">(none)</option>
                  <option value="edges">Edges</option>
                  <option value="depth">Depth</option>
                  <option value="keypoints">Keypoints</option>
                </select>
              </div>
              <div>
                <label>CN image path</label>
                <input id="r_cn_image" placeholder="/path/to/cn.png">
                <div class="row" style="margin-top:6px">
                  <input type="file" id="r_cn_file" accept="image/*">
                  <small id="r_cn_note" class="muted"></small>
                </div>
                <div id="r_cn_prev" class="thumbs"></div>
              </div>
              <div>
                <label>Strength</label>
                <input id="r_cn_w" type="number" value="1.0" step="0.05" min="0" max="2">
              </div>
              <div>
                <label>Start / End</label>
                <input id="r_cn_s" type="number" value="0.0" step="0.05" min="0" max="1">
                <input id="r_cn_e" type="number" value="1.0" step="0.05" min="0" max="1">
              </div>
            </div>
            <div id="r_cn_processed" class="thumbs"></div>
          </div>
        </div>

        <p>
          <button class="btn pri" id="btnRender">Generate (Render)</button>
          <div id="r_notice" class="notice"></div>
        </p>
        <div id="r_out" class="card"></div>
        <div id="r_imgs" class="thumbs"></div>
      </section>

      <!-- UPSCALE -->
      <section id="pane_upscale" style="display:none">
        <div class="card">
          <div class="grid g2">
            <div><label>Upscale model</label><select id="u_model"></select></div>
            <div><label>Prefix</label><input id="u_prefix" value="upscaled"></div>
          </div>
          <div class="grid g2" style="margin-top:10px">
            <div>
              <label>Upload image</label>
              <input type="file" id="u_file" accept="image/*">
              <div class="thumbs" id="u_preview"></div>
              <small id="u_file_note" class="muted"></small>
            </div>
            <div>
              <label>Or path</label>
              <input id="u_path" placeholder="/workspace/runpod-slim/ComfyUI/output/xxx.png">
              <small class="muted">If empty, uploaded image is used.</small>
            </div>
          </div>
          <div style="text-align:right;margin-top:10px">
            <button class="btn pri" id="btnUpscale">Generate (Upscale)</button>
            <div id="u_notice" class="notice"></div>
          </div>
        </div>
        <div id="u_out" class="card"></div>
        <div id="u_imgs" class="thumbs"></div>
      </section>

      <section id="pane_history" style="display:none">
        <div id="h_list"></div>
      </section>
    </section>
  </div>

<script>
let clientId = Math.random().toString(36).slice(2) + Date.now().toString(36);
let ws=null, wsConnected=false, currentPID=null;
let uploadedUpscale = null;
let lastTab = "ideate";

function log(...args){
  const box=document.getElementById('uilog');
  const line=document.createElement('div'); line.className='logline';
  const stamp=new Date().toLocaleTimeString();
  line.innerHTML = '<b>['+stamp+']</b> ' + args.map(a => {
    try { return (typeof a==='string') ? a : JSON.stringify(a); }
    catch(_) { return String(a); }
  }).join(' ');
  box.prepend(line);
}
function logErr(msg){ log('❌', msg); }
function clearLogs(){ document.getElementById('uilog').innerHTML=''; }

function tab(which){
  const ids = ["ideate","render","upscale","history"];
  for(const id of ids){
    document.getElementById("pane_"+id).style.display = (id===which ? "block":"none");
    document.getElementById("tab"+id.charAt(0).toUpperCase()+id.slice(1)).classList.toggle("active", id===which);
  }
  if (which !== lastTab) {
    fetch('/models/unload').then(r=>r.json()).then(j=>log('Unload models', j)).catch(()=>{});
    lastTab = which;
  }
  if(which==="ideate"){ loadIdeateModels(); }
  if(which==="render"){ loadUnets(); }
  if(which==="upscale"){ loadUpscalers(); }
  if(which==="history"){ loadHist(); }
}

async function fetchJSON(url){ const r=await fetch(url); return r.json(); }

async function loadIdeateModels(){
  const ck = await fetchJSON('/models/checkpoints');
  const va = await fetchJSON('/models/vae');
  const cn = await fetchJSON('/models/controlnets');
  fillSelect('i_ckpt', ck.models);
  fillSelect('i_vae',  [''].concat(va.models));
  fillSelect('i_c1m',  [''].concat(cn.models));
  fillSelect('i_c2m',  [''].concat(cn.models));
  log('Loaded models for Ideate');
}
async function loadUpscalers(){
  const up = await fetchJSON('/models/upscalers');
  fillSelect('u_model', up.models);
  log('Loaded upscalers');
}
async function loadUnets(){
  const u = await fetchJSON('/models/unets');
  fillSelect('r_unet', u.models);
  log('Loaded UNETs');
}
function fillSelect(id, arr){
  const el=document.getElementById(id); if (!el) return;
  el.innerHTML='';
  for(const x of arr){ const o=document.createElement('option'); o.value=x; o.textContent=x||'(none)'; el.appendChild(o); }
}

function ensureWS(){
  if(ws && wsConnected) return;
  try{
    const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws?clientId=${clientId}`);
    ws.onopen   = ()=>{ wsConnected=true; log('WS connected'); };
    ws.onclose  = ()=>{ wsConnected=false; log('WS closed'); };
    ws.onerror  = ()=>{ logErr('WS error (will still poll)'); };
    ws.onmessage = (evt)=>{
      try{
        const msg = JSON.parse(evt.data);
        if(msg.type==='progress'){ updateProgress(msg.data?.value||0, msg.data?.max||1); }
        if(msg.type==='executing' && msg.data?.prompt_id){ currentPID = msg.data.prompt_id; log('Executing pid', currentPID); }
        if(msg.type==='preview' && msg.data && msg.data.filename){ showPreview(msg.data); }
      }catch(e){}
    };
  }catch(_){ }
}

function showProgressCard(on){
  document.getElementById('progressCard').style.display = on ? 'block' : 'none';
  if(on){ setStage('running'); updateProgress(0,1); document.getElementById('previewWrap').innerHTML=''; }
}
function setStage(t){ document.getElementById('stagePill').textContent=t; }
function updateProgress(v, m){
  const pct = m>0 ? Math.min(100, Math.round(100*v/m)) : 0;
  document.getElementById('bar').style.width = pct+'%';
  document.getElementById('pct').textContent = pct + '%';
}
function showPreview(d){
  if (!d || !d.filename) return;
  const wrap = document.getElementById('previewWrap');
  const url = `/view?filename=${encodeURIComponent(d.filename)}&type=${encodeURIComponent(d.type || 'temp')}&subfolder=${encodeURIComponent(d.subfolder || '')}&t=${Date.now()}`;
  const img = document.createElement('img');
  img.src = url; img.loading = 'lazy'; img.decoding = 'async';
  wrap.innerHTML = '';
  wrap.appendChild(img);
}

function val(id){ const el=document.getElementById(id); return el?el.value:''; }
function setVal(id, v){ const el=document.getElementById(id); if(el) el.value=v; }
function withBtnFeedback(btnId, noticeId, run){
  const btn=document.getElementById(btnId);
  const note=document.getElementById(noticeId);
  note.textContent=''; btn.disabled=true; btn.classList.add('pressed');
  return run().then(ok=>{
    if(ok){ note.innerHTML='<span class="ok">Job successfully sent</span>'; }
    else  { note.innerHTML='<span class="err">Failed to queue job</span>'; }
  }).catch(e=>{
    note.innerHTML='<span class="err">Error: '+(e?.message||e)+'</span>'; logErr(e);
  }).finally(()=>{
    btn.disabled=false; btn.classList.remove('pressed');
  });
}

async function uploadTo(pathFieldId, fileInputId, previewId, noteId){
  const input = document.getElementById(fileInputId);
  const f = input?.files?.[0]; if(!f){ return null; }
  const fd = new FormData(); fd.append('file', f);
  const r = await fetch('/upload',{method:'POST', body:fd});
  const j = await r.json();
  if(!j.ok){ logErr('Upload failed', j.error); return null; }
  setVal(pathFieldId, j.comfy_rel || j.path);
  const prev=document.getElementById(previewId); prev.innerHTML='';
  const img=document.createElement('img'); img.src=j.url; img.loading='lazy'; img.decoding='async';
  prev.appendChild(img);
  const note=document.getElementById(noteId); if(note) note.textContent=j.path;
  log('Uploaded', fileInputId, '->', j.path);
  return j;
}

async function fireIdeate(){
  ensureWS(); showProgressCard(true);
  const overrides={
    prompt: val('i_prompt'), negative: val('i_negative'),
    steps: val('i_steps'), cfg: val('i_cfg'),
    width: val('i_w'), height: val('i_h'),
    sampler: val('i_sampler'), scheduler: val('i_sched'),
    seed: val('i_seed'), ckpt: val('i_ckpt'), vae: val('i_vae'),
    c1_model: val('i_c1m'), c1_image: val('i_c1img'),
    c1_weight: val('i_c1w'), c1_start: val('i_c1s'), c1_end: val('i_c1e'),
    c1_preprocess: document.getElementById('i_c1pre').checked, c1_low: val('i_c1low'), c1_high: val('i_c1high'),
    c2_model: val('i_c2m'), c2_image: val('i_c2img'),
    c2_weight: val('i_c2w'), c2_start: val('i_c2s'), c2_end: val('i_c2e'),
    c2_preprocess: document.getElementById('i_c2pre').checked, c2_low: val('i_c2low'), c2_high: val('i_c2high'),
    filename_prefix: val('i_prefix')
  };
  log('POST /ideate', overrides);
  const run = async ()=>{
    const r = await fetch('/ideate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_id:clientId,overrides})});
    const t = await r.text(); document.getElementById('i_out').textContent=t;
    log('ideate response', t.slice(0,200));
    try{ const j=JSON.parse(t); if(j.prompt_id){ pollForImages(j.prompt_id,'i_imgs','ideate',overrides); return true; } }catch(e){}
    showProgressCard(false); return false;
  };
  return withBtnFeedback('btnIdeate','i_notice', run);
}

async function fireRender(){
  ensureWS(); showProgressCard(true);
  const overrides={
    prompt: val('r_prompt'), negative: val('r_negative'),
    steps: val('r_steps'), cfg: val('r_cfg'), denoise: val('r_denoise'),
    input_image: val('r_in'),
    lora1: val('r_l1'), lora1_strength: val('r_l1s'),
    lora2: val('r_l2'), lora2_strength: val('r_l2s'),
    filename_prefix: val('r_prefix'),
    unet_name: val('r_unet'),
    cn_mode: document.getElementById('r_cn_mode')?.value || 'none',
    cn_image: val('r_cn_image'),
    cn_strength: val('r_cn_w'),
    cn_start: val('r_cn_s'),
    cn_end: val('r_cn_e'),
  };
  log('POST /render', overrides);
  const run = async ()=>{
    const r = await fetch('/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_id:clientId,overrides})});
    const t = await r.text(); document.getElementById('r_out').textContent=t;
    log('render response', t.slice(0,200));
    try{ const j=JSON.parse(t); if(j.prompt_id){ pollForImages(j.prompt_id,'r_imgs','render',overrides); return true; } }catch(e){}
    showProgressCard(false); return false;
  };
  return withBtnFeedback('btnRender','r_notice', run);
}

async function fireUpscale(){
  ensureWS(); showProgressCard(true);
  const path = val('u_path') || (uploadedUpscale?.path || '');
  if(!path){ document.getElementById('u_out').textContent='Please upload a file or provide a path.'; showProgressCard(false); return; }
  const payload = { client_id: clientId, model_name: val('u_model'), prefix: val('u_prefix'), path };
  log('POST /upscale', payload);
  const run = async ()=>{
    const r = await fetch('/upscale',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const t = await r.text(); document.getElementById('u_out').textContent=t;
    log('upscale response', t.slice(0,200));
    try{ const j=JSON.parse(t); if(j.prompt_id){ pollForImages(j.prompt_id,'u_imgs','upscale',{path}); return true; } }catch(e){}
    showProgressCard(false); return false;
  };
  return withBtnFeedback('btnUpscale','u_notice', run);
}

async function pollForImages(pid, targetId, tabName, overrides){
  currentPID = pid;
  setStage('running');
  log('Polling history for', pid);
  for (let i = 0; i < 240; i++) {
    await new Promise(res => setTimeout(res, 1200));
    const r = await fetch('/history/' + encodeURIComponent(pid));
    const j = await r.json();
    const tgt = document.getElementById(targetId);
    if (j && j[pid] && j[pid].outputs) {
      const outs = j[pid].outputs;
      const imgs = [];
      for (const nid in outs) {
        const node = outs[nid];
        for (const port in node) {
          const arr = node[port];
          if (Array.isArray(arr)) {
            for (const it of arr) {
              if (it && it.filename) imgs.push(it);
            }
          }
        }
      }
      if (imgs.length) {
        showProgressCard(false);
        tgt.innerHTML = '';
        for (const im of imgs) {
          const url = '/view?filename=' + encodeURIComponent(im.filename)
                   + '&type=' + encodeURIComponent(im.type || 'output')
                   + '&subfolder=' + encodeURIComponent(im.subfolder || '')
                   + '&t=' + Date.now();
          const img = document.createElement('img');
          img.src = url; img.loading = 'lazy'; img.decoding = 'async';
          tgt.appendChild(img);
        }
        try {
          if (targetId === 'r_imgs') {
            const cnBox = document.getElementById('r_cn_processed');
            if (cnBox) {
              cnBox.innerHTML = '';
              const cnItems = imgs.filter(im => (im.filename || '').includes('cn_preview'));
              for (const im of cnItems) {
                const url = '/view?filename=' + encodeURIComponent(im.filename)
                          + '&type=' + encodeURIComponent(im.type || 'output')
                          + '&subfolder=' + encodeURIComponent(im.subfolder || '')
                          + '&t=' + Date.now();
                const cnImg = document.createElement('img');
                cnImg.src = url; cnImg.loading = 'lazy'; cnImg.decoding = 'async';
                cnBox.appendChild(cnImg);
              }
            }
          }
        } catch (_) {}
        try {
          await fetch('/history/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ tab: tabName, prompt: overrides?.prompt || '', negative: overrides?.negative || '', params: overrides, images: imgs })
          });
        } catch (_) {}
        log('Images received:', imgs.length);
        return;
      }
    }
  }
  setStage('timeout');
  logErr('Poll timeout for ' + pid);
}

async function onFileChange_map(toFieldId, fileInputId, previewId, noteId){
  await uploadTo(toFieldId, fileInputId, previewId, noteId);
}

document.getElementById('btnIdeate').addEventListener('click', fireIdeate);
document.getElementById('btnRender').addEventListener('click', fireRender);
document.getElementById('btnUpscale').addEventListener('click', fireUpscale);

document.getElementById('i_c1file').addEventListener('change', ()=>onFileChange_map('i_c1img','i_c1file','i_c1prev','i_c1note'));
document.getElementById('i_c2file').addEventListener('change', ()=>onFileChange_map('i_c2img','i_c2file','i_c2prev','i_c2note'));

document.getElementById('r_in_file').addEventListener('change', ()=>onFileChange_map('r_in','r_in_file','r_in_prev','r_in_note'));
document.getElementById('r_cn_file').addEventListener('change', ()=>onFileChange_map('r_cn_image','r_cn_file','r_cn_prev','r_cn_note'));

document.getElementById('u_file').addEventListener('change', async ()=>{
  const r = await uploadTo('u_path','u_file','u_preview','u_file_note');
  if (r) { uploadedUpscale = r; }
});

async function loadHist(){
  const box = document.getElementById('h_list');
  if(!box) return;
  box.innerHTML = '';
  try{
    const r = await fetch('/history/list');
    const j = await r.json();
    const items = j.items || [];
    for(const it of items){
      const card = document.createElement('div');
      card.className = 'card';
      const when = new Date((it.ts||0)*1000).toLocaleString();
      card.innerHTML = `<div class="muted">${when} · ${it.tab||''}</div><div style="margin:6px 0"><b>Prompt:</b> ${(it.prompt||'').slice(0,500)}</div>`;
      const wrap = document.createElement('div');
      wrap.className = 'thumbs';
      for(const im of (it.images||[])){
        const url = '/view?filename=' + encodeURIComponent(im.filename)
                  + '&type=' + encodeURIComponent(im.type || 'output')
                  + '&subfolder=' + encodeURIComponent(im.subfolder || '')
                  + '&t=' + Date.now();
        const img = document.createElement('img');
        img.src = url; img.loading = 'lazy'; img.decoding = 'async';
        wrap.appendChild(img);
      }
      card.appendChild(wrap);
      box.appendChild(card);
    }
  }catch(e){ console.error(e); box.textContent = 'Failed to load history.'; }
}

tab('ideate');
log('UI ready');
</script>
</main>
</body></html>
'''

# Serve the UI directly at /ui (root stays JSON health unless you prefer otherwise)
from fastapi.responses import HTMLResponse

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse(HTML)
PY

# now back in bash, start services
cd "$BASE"
nohup "$VENVPY" main.py --listen 0.0.0.0 --port 8188 >"$LOG" 2>&1 &

cd "$APP_DIR"
nohup "$VENVPY" -m uvicorn app:app --host 0.0.0.0 --port 9999 >/workspace/runpod-slim/ui.log 2>&1 &

echo "ComfyUI running on :8188, UI running on :9999"
