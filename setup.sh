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

# Make sure these exist before writing/reading files
mkdir -p "$BASE/simple-ui" "$BASE/simple-ui/uploads" "$BASE/input" "$BASE/output"
mkdir -p "$BASE/models/diffusion_models" \
         "$BASE/models/text_encoders" \
         "$BASE/models/vae" \
         "$BASE/models/controlnet" \
         "$BASE/models/checkpoints" \
         "$BASE/models/upscale_models" \
         "$BASE/models/loras"

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

# --- ComfyUI requirements only (base deps already installed in Start) ---
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

if [ ! -d "$BASE/.git" ]; then
  echo "ERROR: ComfyUI repo not present at $BASE"; exit 1
fi

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

# ---- Qwen Image Edit 2509 (FP8 single file for ComfyUI) ----
mkdir -p "$BASE/models/diffusion_models" "$BASE/models/text_encoders" "$BASE/models/vae"

# Diffusion / UNet (2509 FP8)
[ -f "$BASE/models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors" ] || \
curl -C - -L -o "$BASE/models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors" \
  "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"

# InstantX ControlNet Union  (FIXED: add $BASE/)
[ -f "$BASE/models/controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors" ] || \
curl -C - -L -o "$BASE/models/controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors" \
  "https://huggingface.co/Comfy-Org/Qwen-Image-InstantX-ControlNets/resolve/main/split_files/controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors"

# Text encoder (same as before)
[ -f "$BASE/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" ] || \
curl -C - -L -o "$BASE/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
  "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"

# VAE (same as before)
[ -f "$BASE/models/vae/qwen_image_vae.safetensors" ] || \
curl -C - -L -o "$BASE/models/vae/qwen_image_vae.safetensors" \
  "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"

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
import websockets
import shutil

BASE = "/workspace/runpod-slim/ComfyUI"
OUT_DIR = os.path.join(BASE, "output")
HIST_FILE = os.path.join(BASE, "simple-ui", "history.json")
UPLOAD_DIR = os.path.join(BASE, "simple-ui", "uploads")

app = FastAPI()
app.mount("/output", StaticFiles(directory=OUT_DIR), name="output")

@app.get("/health")
def health():
    return {"ok": True}
    
# -------- extra endpoints the UI expects --------

# simple list of ControlNet “models”: include built-ins and anything on disk
@app.get("/models/controlnets")
def list_controlnets():
    builtins = ["edges", "depth", "keypoints"]
    disk = list_files(CONTROLNET_DIR, (".safetensors", ".pth", ".onnx"))
    return {"models": builtins + disk}

# proxy history to Comfy so the UI can poll images by prompt_id
@app.get("/history/{pid}")
def get_history(pid: str):
    status, data = comfy_request("GET", f"/history/{pid}")
    return JSONResponse(data, status_code=status)

# allow the UI to persist a tiny history file (best-effort)
@app.post("/history/save")
async def history_save(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}
    items = []
    try:
        if os.path.isfile(HIST_FILE):
            with open(HIST_FILE, "r", encoding="utf-8") as r:
                items = json.load(r) or []
    except Exception:
        pass
    body["ts"] = int(time.time())
    items.append(body)
    try:
        os.makedirs(os.path.dirname(HIST_FILE), exist_ok=True)
        with open(HIST_FILE, "w", encoding="utf-8") as w:
            json.dump(items, w, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return {"ok": True}

@app.post("/ideate")
async def ideate(req: Request):
    payload   = await req.json()
    overrides = payload.get("overrides", {}) or {}
    client_id = payload.get("client_id") or uuid.uuid4().hex
    g = build_ideate(overrides)
    status, data = comfy_request("POST", "/prompt", {"prompt": g["prompt"], "client_id": client_id})
    return JSONResponse(data if isinstance(data, dict) else {"status": status}, status_code=status)

@app.post("/render")
async def render_qwen(req: Request):
    payload   = await req.json()
    overrides = payload.get("overrides", {}) or {}
    client_id = payload.get("client_id") or uuid.uuid4().hex
    g = build_render(overrides)
    status, data = comfy_request("POST", "/prompt", {"prompt": g["prompt"], "client_id": client_id})
    return JSONResponse(data if isinstance(data, dict) else {"status": status}, status_code=status)

@app.post("/upscale")
async def upscale(req: Request):
    payload = await req.json()
    client_id = payload.get("client_id") or uuid.uuid4().hex
    path = payload.get("path") or ""
    model_name = payload.get("model_name") or ""
    prefix = payload.get("prefix") or "upscaled"
    if not (path and model_name):
        return JSONResponse({"error":"path and model_name required"}, status_code=400)
    g = build_upscale(path, model_name, prefix=prefix)
    status, data = comfy_request("POST", "/prompt", {"prompt": g["prompt"], "client_id": client_id})
    return JSONResponse(data if isinstance(data, dict) else {"status": status}, status_code=status)

# very lightweight websocket so UI can connect without 403
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass

# --- Minimal helpers & routes the UI expects ---

# ensure upload dir
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "input"), exist_ok=True)

@app.post("/cn/preprocess")
async def cn_preprocess(req: Request):
    body = await req.json()
    mode = (body.get("cn_mode") or "none").strip().lower()
    img_rel = os.path.basename(body.get("cn_image") or "")
    low = int(body.get("canny_low", 100))
    high = int(body.get("canny_high", 200))

    if mode not in ("edges", "depth", "keypoints") or not img_rel:
        return JSONResponse({"ok": False, "error": "invalid input"}, status_code=400)

    # Build a tiny Comfy graph: LoadImage -> (preprocessor) -> SaveImage(prefix='cn_preview')
    g = {"1": {"class_type": "LoadImage", "inputs": {"image": img_rel}}}

    nodes = _object_info_nodes() or {}
    def pick(*names):
        for n in names:
            if n in nodes: return n
        return None

    if mode == "edges":
        pre = pick("CannyEdgePreprocessor", "CannyPreprocessor", "Canny", "CannyEdgeDetector")
        inputs = {"image": ["1",0], "low_threshold": low, "high_threshold": high}
    elif mode == "depth":
        pre = pick("DepthAnythingPreprocessor", "DepthAnythingDetector", "MidasDepthPreprocessor", "LeReS-DepthMapPreprocessor")
        inputs = {"image": ["1",0]}
    else:  # keypoints
        pre = pick("DWPreprocessor", "DWposePreprocessor", "OpenposePreprocessor", "OpenPosePreprocessor", "OpenPose")
        inputs = {"image": ["1",0]}

    src_id = "1"
    if pre:
        g["2"] = {"class_type": pre, "inputs": inputs}
        src_id = "2"

    g["3"] = {"class_type": "SaveImage", "inputs": {"images": [src_id,0], "filename_prefix": "cn_preview"}}

    status, data = comfy_request("POST", "/prompt", {"prompt": g, "client_id": uuid.uuid4().hex})
    if status != 200 or not isinstance(data, dict) or "prompt_id" not in data:
        return JSONResponse({"ok": False, "error": "comfy rejected"}, status_code=500)

    im = _first_hist_image(data["prompt_id"], 60)
    if not im:
        return JSONResponse({"ok": False, "error": "no preview"}, status_code=504)

    rel = _copy_output_to_input(im)  # copy preview into /input so it can be used as CN map
    preview_url = f"/view?filename={urllib.parse.quote(im['filename'])}&type=output&subfolder={urllib.parse.quote(im.get('subfolder',''))}"
    return {"ok": True, "cn_proc_rel": rel or "", "preview_url": preview_url}

@app.get("/history/list")
def history_list():
    return {"items": list(reversed(load_hist()))}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # save to our uploads folder
    name = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, name)
    with open(save_path, "wb") as w:
        w.write(await file.read())
    # also copy into ComfyUI/input so LoadImage can read it
    try:
        with open(save_path, "rb") as r, open(os.path.join(BASE, "input", name), "wb") as w:
            w.write(r.read())
    except Exception:
        pass
    # UI expects a preview URL
    return {
        "ok": True,
        "path": save_path,
        "comfy_rel": name,
        "url": f"/view?filename={name}&type=input&subfolder="
    }

# serve images back to the UI (from ComfyUI folders)
from fastapi.responses import FileResponse

@app.get("/view")
def view(filename: str, type: str = "output", subfolder: str = ""):
    roots = {
        "output": os.path.join(BASE, "output"),
        "temp":   os.path.join(BASE, "temp"),
        "input":  os.path.join(BASE, "input"),
    }
    root = roots.get(type, roots["output"])
    filename = os.path.basename(filename)
    subfolder = os.path.normpath(subfolder).lstrip("./\\")
    path = os.path.join(root, subfolder, filename) if subfolder else os.path.join(root, filename)
    return FileResponse(path)

# simple model listings the UI fills dropdowns with
@app.get("/models/checkpoints")
def list_ckpts():
    return {"models": list_files(CHECKPOINTS_DIR, (".safetensors", ".ckpt"))}

@app.get("/models/vae")
def list_vae():
    return {"models": list_files(VAE_DIR, (".safetensors", ".ckpt"))}

@app.get("/models/upscalers")
def list_upscalers():
    return {"models": list_files(UPSCALE_DIR, (".pth",))}

@app.get("/models/unets")
def list_unets():
    return {"models": list_files(DIFFUSION_DIR, (".safetensors", ".ckpt", ".bin"))}

# UI sometimes calls this to “free” VRAM; just accept it
@app.get("/models/unload")
def unload_models():
    return {"ok": True}

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

def add_builtin_cn_with_aux(
    g: dict,
    *,
    vae_id: str,
    cn_image_id: str,       # id of a LoadImage node that loads the CN image
    ksampler_id: str,       # id of your KSampler node
    positive_cond_id: str,  # id of your positive conditioning node
    cn_mode: str,           # "edges" | "depth" | "keypoints"
    cn_strength: float = 1.0,
    cn_start: float = 0.0,
    cn_end: float = 1.0,
) -> None:
    """
    Implements built-in CN types using comfyui_controlnet_aux preprocessors, then
    ControlNetApplyAdvanced. Works even if Qwen-specific CN node isn't installed.
    """
    mode = (cn_mode or "none").strip().lower()
    if mode not in ("edges", "depth", "keypoints"):
        return

    nodes = _object_info_nodes() or {}
    def has_node(*cands): return next((c for c in cands if c in nodes), None)

    # 1) Pick a preprocessor for the selected mode
    pre_class = None
    pre_inputs = {}

    if mode == "edges":
        # a few common names shipped by controlnet_aux
        pre_class = has_node(
            "CannyEdgePreprocessor", "CannyPreprocessor", "Canny", "CannyEdgeDetector"
        )
        if pre_class:
            pre_inputs = {
                # map common input names; fallbacks are tolerated by Comfy
                "image": [cn_image_id, 0],
                "low_threshold": 100,
                "high_threshold": 200,
            }

    elif mode == "depth":
        pre_class = has_node(
            "DepthAnythingPreprocessor",
            "DepthAnythingDetector",
            "MidasDepthPreprocessor",
            "LeReS-DepthMapPreprocessor",
            "MiDaS-Depth Map Preprocessor",
        )
        if pre_class:
            pre_inputs = {"image": [cn_image_id, 0]}

    elif mode == "keypoints":
        pre_class = has_node(
            "DWPreprocessor", "DWposePreprocessor", "OpenposePreprocessor",
            "OpenPosePreprocessor", "OpenPose"
        )
        if pre_class:
            pre_inputs = {"image": [cn_image_id, 0]}

    # 2) If we found a preprocessor, insert it and replace cn_image_id with its output
    if pre_class:
        next_id = str(max([int(i) for i in g.keys() if i.isdigit()] + [0]) + 1)
        g[next_id] = {"class_type": pre_class, "inputs": pre_inputs}
        cn_proc_id = next_id
    else:
        # no preprocessor available; just pass the raw CN image
        cn_proc_id = cn_image_id

    # 3) Loader + ApplyAdvanced
    base_id = max([int(i) for i in g.keys() if i.isdigit()] + [0]) + 1
    cn_loader_id  = str(base_id)
    cn_apply_id   = str(base_id + 1)
    cn_preview_id = str(base_id + 2)

    # Loader: use the built-in name (“edges”/“depth”/“keypoints”) directly
    g[cn_loader_id] = {
        "class_type": "ControlNetLoaderAdvanced",
        "inputs": {"control_net_name": mode}
    }

    # Optional: save the processed map for debugging/preview
    g[cn_preview_id] = {
        "class_type": "SaveImage",
        "inputs": {"images": [cn_proc_id, 0], "filename_prefix": "cn_preview"}
    }

    # Introspect ApplyAdvanced so we set the right field names
    info = nodes.get("ControlNetApplyAdvanced", {}) or {}
    inputs_spec  = (info.get("input") or info.get("inputs") or {}) if isinstance(info, dict) else {}
    outputs_spec = (info.get("output") or info.get("outputs") or {}) if isinstance(info, dict) else {}

    def in_has(name):  return name in inputs_spec
    def out_has(name): return name in outputs_spec

    cn_inputs = {
        "positive":      [positive_cond_id, 0],
        "control_net":   [cn_loader_id, 0],
        "image":         [cn_proc_id, 0],
        "vae":           [vae_id, 0],
        "strength":      float(cn_strength),
        "start_percent": float(cn_start),
        "end_percent":   float(cn_end),
    }
    # accept common aliases
    if in_has("start"):        cn_inputs["start"] = cn_inputs.pop("start_percent")
    if in_has("end"):          cn_inputs["end"]   = cn_inputs.pop("end_percent")
    if in_has("cn_mode"):      cn_inputs["cn_mode"] = mode
    if in_has("control_type"): cn_inputs["control_type"] = mode
    if in_has("preprocess"):   cn_inputs["preprocess"] = False  # if supported, ensure “preprocess” path is chosen

    g[cn_apply_id] = {"class_type": "ControlNetApplyAdvanced", "inputs": cn_inputs}

    # Wire sampler
    two_outs = out_has("negative") or (len({k.lower() for k in outputs_spec}) >= 2)
    g[ksampler_id]["inputs"]["positive"] = [cn_apply_id, 0]
    if two_outs and "negative" in g[ksampler_id]["inputs"]:
        g[ksampler_id]["inputs"]["negative"] = [cn_apply_id, 1]

def _first_hist_image(pid: str, timeout_s: int = 45) -> Optional[dict]:
    """Poll Comfy history for first produced image of this prompt_id."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        s, data = comfy_request("GET", f"/history/{pid}")
        if s == 200 and isinstance(data, dict) and pid in data:
            outs = data[pid].get("outputs", {}) or {}
            for _, node in outs.items():
                for _, arr in node.items():
                    if isinstance(arr, list):
                        for it in arr:
                            if it and it.get("filename"):
                                return it
        time.sleep(1.0)
    return None

def _copy_output_to_input(im: dict) -> Optional[str]:
    """Copy a saved output image into BASE/input and return a new relative filename."""
    try:
        filename   = im["filename"]
        subfolder  = im.get("subfolder","")
        src_dir    = os.path.join(BASE, "output", subfolder) if subfolder else os.path.join(BASE, "output")
        src_path   = os.path.join(src_dir, filename)
        rel_name   = f"cnproc_{uuid.uuid4().hex}_{os.path.basename(filename)}"
        dst_path   = os.path.join(BASE, "input", rel_name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(src_path, "rb") as r, open(dst_path, "wb") as w:
            w.write(r.read())
        return rel_name
    except Exception:
        return None


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
# ---------------- Workflows ----------------
def build_ideate(o: Dict[str, Any]) -> Dict[str, Any]:
    """
    SDXL text2img:
      CKPT[/VAE] -> CLIP encodes -> EmptyLatent -> KSampler -> VAEDecode -> Save
    Optional ControlNets (2 slots), each with optional Canny preprocessing.
    """
    prompt   = o.get("prompt","")
    negative = o.get("negative","")
    steps    = int(float(o.get("steps",20)))
    cfg      = float(o.get("cfg", 6.0))
    sampler  = o.get("sampler","euler")
    sched    = o.get("scheduler","normal")
    seed     = int(float(o.get("seed", 0)))
    width    = int(float(o.get("width", 1024)))
    height   = int(float(o.get("height",1024)))

    ckpt     = (o.get("ckpt") or "RealVisXL_V4.0.safetensors").strip()
    vae_opt  = (o.get("vae")  or "").strip()

    # Auto-pick "<ckpt>_vae.*" if present and user didn't set a VAE
    if not vae_opt and ckpt:
        base = os.path.splitext(ckpt)[0]
        for ext in (".safetensors", ".ckpt"):
            cand = f"{base}_vae{ext}"
            if os.path.isfile(os.path.join(VAE_DIR, cand)):
                vae_opt = cand
                break

    # ControlNet slot #1
    c1_model = (o.get("c1_model") or "").strip()
    c1_img   = (o.get("c1_image") or "").strip()
    c1_w     = float(o.get("c1_weight", 1.0))
    c1_s     = float(o.get("c1_start", 0.0))
    c1_e     = float(o.get("c1_end",   1.0))
    c1_pre   = bool(o.get("c1_preprocess", False))
    c1_low   = int(float(o.get("c1_low", 100)))
    c1_high  = int(float(o.get("c1_high",200)))

    # ControlNet slot #2
    c2_model = (o.get("c2_model") or "").strip()
    c2_img   = (o.get("c2_image") or "").strip()
    c2_w     = float(o.get("c2_weight", 1.0))
    c2_s     = float(o.get("c2_start", 0.0))
    c2_e     = float(o.get("c2_end",   1.0))
    c2_pre   = bool(o.get("c2_preprocess", False))
    c2_low   = int(float(o.get("c2_low", 100)))
    c2_high  = int(float(o.get("c2_high",200)))
    
    # after reading c1_* and c2_* fields
    c1_proc_rel = os.path.basename(o.get("c1_proc_rel", "") or o.get("c1_image", "") or "")
    c2_proc_rel = os.path.basename(o.get("c2_proc_rel", "") or o.get("c2_image", "") or "")

    prefix   = (o.get("filename_prefix","ideate_sdxl") or "ideate_sdxl").strip()

    # Base graph
    g = {
        "1": {"class_type":"CheckpointLoaderSimple",
              "inputs":{"ckpt_name": ckpt}},
        "2": {"class_type":"CLIPTextEncode",
              "inputs":{"clip":["1",1], "text": prompt}},
        "3": {"class_type":"CLIPTextEncode",
              "inputs":{"clip":["1",1], "text": negative}},
        "4": {"class_type":"EmptyLatentImage",
              "inputs":{"width":width,"height":height,"batch_size":1}},
        "5": {"class_type":"KSampler","inputs":{
              "model":["1",0],"positive":["2",0],"negative":["3",0],
              "latent_image":["4",0],
              "seed":seed,"steps":steps,"cfg":cfg,
              "sampler_name":sampler,"scheduler":sched,"denoise":1.0}},
        "6": {"class_type":"VAEDecode",
              "inputs":{"samples":["5",0],"vae":["1",2]}},
        "7": {"class_type":"SaveImage",
              "inputs":{"images":["6",0],"filename_prefix": prefix}}
    }

    # If user picked an explicit VAE, load and use it
    if vae_opt:
        g["10"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae_opt}}
        g["6"]["inputs"]["vae"] = ["10", 0]

    def _vae_ref():
        return ["10", 0] if "10" in g else ["1", 2]

    # ---- helper to attach a CN ----
    idx = 20
    def plug_control(model_name, image_path, weight, start, end, pre, low, high, *, proc_rel=""):
        nonlocal idx, g
        if not (model_name and (image_path or proc_rel)):
            return

        # prefer preprocessed map if provided
        chosen_rel = os.path.basename(proc_rel or image_path)

        # loader + image
        g[str(idx)]   = {"class_type": "ControlNetLoader",
                         "inputs": {"control_net_name": model_name}}
        g[str(idx+1)] = {"class_type": "LoadImage",
                         "inputs": {"image": chosen_rel}}
        image_src_id  = str(idx+1)

        # only run inline Canny if user ticked it AND there's no preprocessed map
        if pre and not proc_rel:
            image_src_id = add_canny_pre(g, image_src_id, low, high)

        # wire ApplyAdvanced
        info = _object_info_nodes().get("ControlNetApplyAdvanced", {}) or {}
        inputs_spec  = (info.get("input") or info.get("inputs") or {})
        outputs_spec = (info.get("output") or info.get("outputs") or {})
        in_keys      = {k.lower() for k in inputs_spec} if isinstance(inputs_spec, dict) else set()
        out_keys     = {k.lower() for k in outputs_spec} if isinstance(outputs_spec, dict) else set()

        cn_inputs = {
            "control_net":   [str(idx), 0],
            "image":         [image_src_id, 0],
            "strength":      float(weight),
            "start_percent": float(start),
            "end_percent":   float(end),
            "vae":           _vae_ref(),
        }
        if "conditioning" in in_keys:
            cn_inputs["conditioning"] = ["2", 0]   # single-port variant
        else:
            cn_inputs["positive"] = ["2", 0]
            cn_inputs["negative"] = ["3", 0]

        if "start" in in_keys: cn_inputs["start"] = cn_inputs.pop("start_percent")
        if "end"   in in_keys: cn_inputs["end"]   = cn_inputs.pop("end_percent")
        if "cn_mode" in in_keys:      cn_inputs["cn_mode"] = model_name
        if "control_type" in in_keys: cn_inputs["control_type"] = model_name
        if "preprocess" in in_keys:   cn_inputs["preprocess"] = False

        g[str(idx+2)] = {"class_type": "ControlNetApplyAdvanced", "inputs": cn_inputs}

        has_two_out = ("negative" in out_keys) or (len(out_keys) >= 2)
        g["5"]["inputs"]["positive"] = [str(idx+2), 0]
        g["5"]["inputs"]["negative"] = [str(idx+2), 1] if has_two_out else ["3", 0]

        idx += 3

    # attach up to two CNs (using preprocessed maps if given)
    plug_control(c1_model, c1_img, c1_w, c1_s, c1_e, c1_pre, c1_low, c1_high, proc_rel=c1_proc_rel)
    plug_control(c2_model, c2_img, c2_w, c2_s, c2_e, c2_pre, c2_low, c2_high, proc_rel=c2_proc_rel)

    return {"prompt": g}

def _infer_qwen_unet_dtype(unet_name: str):
    """
    Guess the correct UNet weight dtype enum for UNETLoader.
    Returns (tag, comfy_dtype), comfy_dtype is what UNETLoader expects.
    """
    n = (unet_name or "").lower()
    if "fp8" in n:
        return ("fp8", "fp8_e4m3fn")
    if "bf16" in n or "bfloat16" in n:
        return ("bf16", "bf16")
    if "fp16" in n or "half" in n:
        return ("fp16", "fp16")
    return ("default", "default")

def _union_cn_filename() -> Optional[str]:
    """Return the filename Comfy expects for the InstantX Union model if present, else None."""
    try:
        fn = "Qwen-Image-InstantX-ControlNet-Union.safetensors"
        path = os.path.join(CONTROLNET_DIR, fn)
        return fn if os.path.isfile(path) else None
    except Exception:
        return None


def build_render(o: Dict[str, Any]) -> Dict[str, Any]:
    """
    Qwen Image Edit 2509 (image-to-image) with optional CN:
      - CN is used ONLY if a preprocessed map (cn_proc_rel) is provided.
      - Prefers InstantX Union; else uses built-in control types.
    """
    # ---- Core params ----
    prompt    = o.get("prompt", "")
    negative  = o.get("negative", "")
    steps     = int(float(o.get("steps", 5)))
    cfg       = float(o.get("cfg", 1.0))
    denoise   = float(o.get("denoise", 0.40))
    prefix    = (o.get("filename_prefix", "render_qwen") or "render_qwen").strip()

    # Input image (already copied into BASE/input by /upload)
    input_rel = os.path.basename(o.get("input_image", "") or "") or "MISSING_INPUT.png"

    # ---- Qwen Edit 2509 UNET (FP8) ----
    unet_name = (o.get("unet_name") or "qwen_image_edit_2509_fp8_e4m3fn.safetensors").strip()
    _, udtype = _infer_qwen_unet_dtype(unet_name)

    # ---- LoRAs ----
    l1, l1s = (o.get("lora1") or "Qwen-Image-Lightning-4steps-V1.0.safetensors").strip(), float(o.get("lora1_strength", 1.0))
    l2, l2s = (o.get("lora2") or "").strip(), float(o.get("lora2_strength", 0.0))

    # ---- Base Qwen edit pipeline ----
    g = {
        "1":  {"class_type": "LoadImage", "inputs": {"image": input_rel}},
        "2":  {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "3":  {"class_type": "CLIPLoader", "inputs": {
                  "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                  "type": "qwen_image", "variant": "default"}},
        "4":  {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": udtype}},
    }

    # LoRA chain (model branch)
    model_ref = ["4", 0]
    if l1:
        g["5"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": model_ref, "lora_name": l1, "strength_model": l1s}}
        model_ref = ["5", 0]
    if l2 and l2s > 0.0:
        g["12"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model_ref, "lora_name": l2, "strength_model": l2s}}
        model_ref = ["12", 0]

    # Text encoder specific to Qwen Edit; negative text via CLIPTextEncode
    _nodes      = _object_info_nodes()
    _qie_info   = _nodes.get("TextEncodeQwenImageEdit", {}) if isinstance(_nodes, dict) else {}
    _qie_inputs = (_qie_info.get("input") or _qie_info.get("inputs") or {}) if isinstance(_qie_info, dict) else {}
    _text_field = "text" if "text" in _qie_inputs else ("prompt" if "prompt" in _qie_inputs else "text")

    g["6"] = {"class_type": "TextEncodeQwenImageEdit",
              "inputs": {"clip": ["3", 0], "vae": ["2", 0], "image": ["1", 0], _text_field: prompt}}

    g["7"] = {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["3", 0], "text": negative}}

    # Encode the input to latents
    g["8"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["2", 0]}}

    # Sampler
    g["9"] = {"class_type": "KSampler", "inputs": {
        "model":        model_ref,
        "positive":     ["6", 0],
        "negative":     ["7", 0],
        "latent_image": ["8", 0],
        "seed": 0, "steps": steps, "cfg": cfg,
        "sampler_name": "euler", "scheduler": "simple", "denoise": denoise,
    }}

    # Decode & save
    g["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["2", 0]}}
    g["11"] = {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}}

    # ---- Optional ControlNet (requires preprocessed map) ----
    cn_mode     = (o.get("cn_mode") or "none").strip().lower()             # edges|depth|keypoints|none
    cn_proc_rel = os.path.basename(o.get("cn_proc_rel","") or "")           # set by /cn/preprocess
    cn_w        = float(o.get("cn_strength", 1.0))
    cn_s        = float(o.get("cn_start", 0.0))
    cn_e        = float(o.get("cn_end",   1.0))

    use_cn = (cn_mode in ("edges","depth","keypoints")) and bool(cn_proc_rel)
    if use_cn:
        # Load the PREPROCESSED map and save a preview
        g["20"] = {"class_type": "LoadImage", "inputs": {"image": cn_proc_rel}}
        g["21"] = {"class_type": "SaveImage", "inputs": {"images": ["20", 0], "filename_prefix": "cn_preview"}}

        # Prefer InstantX Union, else fallback to built-in advanced types
        union_name = _union_cn_filename()
        loader_id  = "22"
        apply_id   = "23"

        if union_name:
            g[loader_id] = {"class_type":"ControlNetLoader", "inputs":{"control_net_name": union_name}}
        else:
            g[loader_id] = {"class_type":"ControlNetLoaderAdvanced", "inputs":{"control_net_name": cn_mode}}

        # Introspect ApplyAdvanced for input aliases
        info = (_nodes.get("ControlNetApplyAdvanced") or {}) if isinstance(_nodes, dict) else {}
        ins  = (info.get("input") or info.get("inputs") or {}) if isinstance(info, dict) else {}
        outs = (info.get("output") or info.get("outputs") or {}) if isinstance(info, dict) else {}
        in_keys  = {k.lower() for k in ins.keys()} if isinstance(ins, dict) else set()
        out_keys = {k.lower() for k in outs.keys()} if isinstance(outs, dict) else set()

        cn_inputs = {
            "control_net":   [loader_id, 0],
            "image":         ["20", 0],    # preprocessed hint
            "vae":           ["2", 0],
            "strength":      cn_w,
            "start_percent": cn_s,
            "end_percent":   cn_e,
        }
        # conditioning ports
        if "conditioning" in in_keys:
            cn_inputs["conditioning"] = ["6", 0]
        else:
            cn_inputs["positive"] = ["6", 0]
            cn_inputs["negative"] = ["7", 0]

        # mode/alias hints
        if "control_type" in in_keys: cn_inputs["control_type"] = cn_mode
        if "cn_mode" in in_keys:      cn_inputs["cn_mode"]      = cn_mode
        if "preprocess" in in_keys:   cn_inputs["preprocess"]   = False  # already preprocessed
        if "start" in in_keys:        cn_inputs["start"]        = cn_inputs.pop("start_percent")
        if "end" in in_keys:          cn_inputs["end"]          = cn_inputs.pop("end_percent")

        g[apply_id] = {"class_type":"ControlNetApplyAdvanced", "inputs": cn_inputs}

        # Recommended for Qwen Edit: use CN for positive only; keep plain negative text
        g["9"]["inputs"]["positive"] = [apply_id, 0]
        g["9"]["inputs"]["negative"] = ["7", 0]

    return {"prompt": g}

def _ensure_in_input(path: str) -> str:
    """
    Return a filename that exists under BASE/input.
    If an absolute path is provided, copy it into BASE/input and return the new relative name.
    """
    if not path:
        return ""
    name = os.path.basename(path)
    dst = os.path.join(BASE, "input", name)
    if os.path.isabs(path):
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(path, dst)
        except Exception:
            pass
    return name

def build_upscale(path: str, model_name: str, *, prefix: str = "upscaled") -> Dict[str, Any]:
    rel = _ensure_in_input(path)
    g = {
        "1": {"class_type": "LoadImage", "inputs": {"image": rel}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"image": ["1", 0], "upscale_model": ["2", 0]}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": prefix}},
    }
    return {"prompt": g}

# ---------------- UI ----------------
# Updated to: (1) fix built-in CN model dropdown bug (no external model list needed),
# (2) make numeric fields use <input type="number"> with sensible step/min,
# (3) add a /ui route that serves this HTML directly,
# (4) tighten progress/log handling, (5) small UX polish.
# (Plus: ControlNet preprocess flow: upload → preprocess → preview → enforce-before-render)

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

            <!-- NEW: CN1 preprocess controls -->
            <div class="row" style="margin-top:8px; align-items:center; gap:8px">
              <label>Preprocess type</label>
              <select id="i_c1mode" style="min-width:140px">
                <option value="none">(none)</option>
                <option value="edges">Edges</option>
                <option value="depth">Depth</option>
                <option value="keypoints">Keypoints</option>
              </select>

              <span id="i_c1canny_row" class="muted" style="display:none; margin-left:12px">
                low <input id="i_c1low2"  type="number" value="100" style="width:80px" min="0" max="255" step="1">
                high <input id="i_c1high2" type="number" value="200" style="width:80px" min="0" max="255" step="1">
              </span>

              <button class="btn" id="i_c1pre_btn" style="margin-left:auto">Preprocess</button>
            </div>
            <small id="i_c1pre_note" class="muted"></small>
            <div id="i_c1proc_prev" class="thumbs"></div>

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
            
            <!-- NEW: CN2 preprocess controls -->
            <div class="row" style="margin-top:8px; align-items:center; gap:8px">
              <label>Preprocess type</label>
              <select id="i_c2mode" style="min-width:140px">
                <option value="none">(none)</option>
                <option value="edges">Edges</option>
                <option value="depth">Depth</option>
                <option value="keypoints">Keypoints</option>
              </select>
      
              <span id="i_c2canny_row" class="muted" style="display:none; margin-left:12px">
                low <input id="i_c2low2"  type="number" value="100" style="width:80px" min="0" max="255" step="1">
                high <input id="i_c2high2" type="number" value="200" style="width:80px" min="0" max="255" step="1">
              </span>
              
              <button class="btn" id="i_c2pre_btn" style="margin-left:auto">Preprocess</button>
            </div>
            <small id="i_c2pre_note" class="muted"></small>
            <div id="i_c2proc_prev" class="thumbs"></div>
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

            <!-- NEW: Edges-only canny thresholds + preprocess button -->
            <div class="row" id="r_canny_row" style="margin-top:8px; align-items:center; gap:8px; display:none">
              <span class="muted">Canny:</span>
              <label for="r_canny_low" class="muted" style="margin:0">low</label>
              <input id="r_canny_low" type="number" value="100" style="width:90px" min="0" max="255" step="1">
              <label for="r_canny_high" class="muted" style="margin:0">high</label>
              <input id="r_canny_high" type="number" value="200" style="width:90px" min="0" max="255" step="1">
              <button class="btn" id="r_cn_pre_btn" style="margin-left:auto">Preprocess</button>
            </div>
            <small id="r_cn_pre_note" class="muted"></small>

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

// Track if CN has been preprocessed for current input/mode
let rCnPreprocessed = false;
let rCnProcessedRel = ""; // server-returned comfy-relative filename (goes into r_cn_image)

let iC1Preprocessed = false, iC1ProcRel = "";
let iC2Preprocessed = false, iC2ProcRel = "";

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

async function fetchJSON(url, opts){ const r=await fetch(url, opts); return r.json(); }

async function loadIdeateModels(){
  const ck = await fetchJSON('/models/checkpoints');
  const va = await fetchJSON('/models/vae');
  const cn = await fetchJSON('/models/controlnets');

  const builtins = new Set(['edges','depth','keypoints']);
  const diskOnly = (cn.models || []).filter(m => !builtins.has(m));

  fillSelect('i_ckpt', ck.models);
  fillSelect('i_vae',  [''].concat(va.models));
  fillSelect('i_c1m',  [''].concat(diskOnly));
  fillSelect('i_c2m',  [''].concat(diskOnly));
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
  // any upload changes invalidate preprocessing state
  rCnPreprocessed = false; rCnProcessedRel = "";
  document.getElementById('r_cn_pre_note')?.textContent = '';
  document.getElementById('r_cn_processed').innerHTML = '';
  return j;
}

function requireCnPreprocessedIfNeeded(){
  const mode = (document.getElementById('r_cn_mode')?.value || 'none');
  if (mode === 'none') return true;
  if (!val('r_cn_image')) return false;
  return rCnPreprocessed; // must preprocess first
}

async function fireIdeate(){
  ensureWS();
  showProgressCard(true);

  const overrides = {
    prompt: val('i_prompt'), negative: val('i_negative'),
    steps: val('i_steps'), cfg: val('i_cfg'),
    width: val('i_w'), height: val('i_h'),
    sampler: val('i_sampler'), scheduler: val('i_sched'),
    seed: val('i_seed'), ckpt: val('i_ckpt'), vae: val('i_vae'),

    c1_model: val('i_c1m'), c1_image: val('i_c1img'),
    c1_weight: val('i_c1w'), c1_start: val('i_c1s'), c1_end: val('i_c1e'),

    c2_model: val('i_c2m'), c2_image: val('i_c2img'),
    c2_weight: val('i_c2w'), c2_start: val('i_c2s'), c2_end: val('i_c2e'),

    filename_prefix: val('i_prefix'),

    // NEW: ideate CN preprocess selections + results
    c1_mode: document.getElementById('i_c1mode')?.value || 'none',
    c1_low: document.getElementById('i_c1low2')?.value || '100',
    c1_high: document.getElementById('i_c1high2')?.value || '200',
    c1_proc_rel: iC1ProcRel,

    c2_mode: document.getElementById('i_c2mode')?.value || 'none',
    c2_low: document.getElementById('i_c2low2')?.value || '100',
    c2_high: document.getElementById('i_c2high2')?.value || '200',
    c2_proc_rel: iC2ProcRel,
  };

  log('POST /ideate', overrides);

  const run = async ()=>{
    const r = await fetch('/ideate', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ client_id: clientId, overrides })
    });
    const t = await r.text();
    document.getElementById('i_out').textContent = t;
    log('ideate response', t.slice(0,200));
    try {
      const j = JSON.parse(t);
      if (j.prompt_id){
        pollForImages(j.prompt_id, 'i_imgs', 'ideate', overrides);
        return true;
      }
    } catch(_) {}
    showProgressCard(false);
    return false;
  };

  return withBtnFeedback('btnIdeate','i_notice', run);
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

async function fireRender(){
  ensureWS();
  showProgressCard(true);

  // if CN type selected, make sure the map was preprocessed first
  if (!requireCnPreprocessedIfNeeded()){
    document.getElementById('r_notice').innerHTML =
      '<span class="err">Please preprocess the ControlNet image first (click "Preprocess").</span>';
    showProgressCard(false);
    return;
  }

  const overrides = {
    // core
    prompt:        val('r_prompt'),
    negative:      val('r_negative'),
    steps:         val('r_steps'),
    cfg:           val('r_cfg'),
    denoise:       val('r_denoise'),
    unet_name:     val('r_unet'),
    filename_prefix: val('r_prefix'),

    // input image
    input_image:   val('r_in'),

    // ControlNet (built-in) — uses preprocessed map if available
    cn_mode:       (document.getElementById('r_cn_mode')?.value || 'none'),
    cn_proc_rel:   (rCnProcessedRel || val('r_cn_image') || ''),
    cn_strength:   val('r_cn_w'),
    cn_start:      val('r_cn_s'),
    cn_end:        val('r_cn_e'),

    // LoRAs
    lora1:        val('r_l1'),
    lora1_strength: val('r_l1s'),
    lora2:        val('r_l2'),
    lora2_strength: val('r_l2s'),
  };

  log('POST /render', overrides);

  const run = async ()=>{
    const r = await fetch('/render', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ client_id: clientId, overrides })
    });
    const t = await r.text();
    document.getElementById('r_out').textContent = t;
    log('render response', t.slice(0,200));
    try{
      const j = JSON.parse(t);
      if (j.prompt_id){
        pollForImages(j.prompt_id, 'r_imgs', 'render', overrides);
        return true;
      }
    }catch(_){}
    showProgressCard(false);
    return false;
  };

  return withBtnFeedback('btnRender','r_notice', run);
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
              // also pull out any cn_preview saved during render (fallback)
              const cnItems = imgs.filter(im => (im.filename || '').includes('cn_preview'));
              if (cnItems.length){
                cnBox.innerHTML = '';
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

// ---------- CN preprocess UI wiring ----------
function updateCannyRowVisibility(){
  const mode = (document.getElementById('r_cn_mode')?.value || 'none');
  const row = document.getElementById('r_canny_row');
  if (row) row.style.display = (mode === 'edges') ? 'flex' : 'none';
  // changing mode invalidates preprocess
  rCnPreprocessed = false; rCnProcessedRel = "";
  document.getElementById('r_cn_pre_note')?.textContent = '';
  document.getElementById('r_cn_processed').innerHTML = '';
}
document.getElementById('r_cn_mode').addEventListener('change', updateCannyRowVisibility);

async function preprocessIdeateCN(slot){
  const modeSel = document.getElementById(slot===1 ? 'i_c1mode' : 'i_c2mode');
  const imgField = document.getElementById(slot===1 ? 'i_c1img'  : 'i_c2img');
  const note = document.getElementById(slot===1 ? 'i_c1pre_note' : 'i_c2pre_note');
  const prev = document.getElementById(slot===1 ? 'i_c1proc_prev' : 'i_c2proc_prev');

  const mode = (modeSel?.value || 'none');
  const imgPath = imgField?.value || '';
  if (mode === 'none'){ note.textContent = 'Pick a preprocess type first.'; return; }
  if (!imgPath){ note.textContent = 'Upload or choose an image first.'; return; }

  const low = parseInt(document.getElementById(slot===1 ? 'i_c1low2'  : 'i_c2low2')?.value || '100', 10);
  const high= parseInt(document.getElementById(slot===1 ? 'i_c1high2' : 'i_c2high2')?.value || '200', 10);

  try{
    note.textContent = 'Preprocessing…';
    const j = await fetchJSON('/cn/preprocess', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        cn_mode: mode,
        cn_image: imgPath,
        canny_low: low,
        canny_high: high
      })
    });
    if (!j || !j.ok){ note.innerHTML = '<span class="err">Preprocess failed.</span>'; return; }

    const rel = j.cn_proc_rel || '';
    if (slot===1){ iC1ProcRel = rel; iC1Preprocessed = !!rel; }
    else         { iC2ProcRel = rel; iC2Preprocessed = !!rel; }

    // show preview
    prev.innerHTML = '';
    if (j.preview_url){
      const im = document.createElement('img');
      im.src = j.preview_url + '&t=' + Date.now();
      im.loading = 'lazy'; im.decoding = 'async';
      prev.appendChild(im);
    }
    note.innerHTML = '<span class="ok">Preprocessed ✔</span>';
    log('Ideate CN preprocess result', {slot, j});
  }catch(e){
    note.innerHTML = '<span class="err">Preprocess error.</span>';
    logErr(e?.message || e);
  }
}

async function preprocessCN(){
  const mode = (document.getElementById('r_cn_mode')?.value || 'none');
  const imgPath = val('r_cn_image');
  const note = document.getElementById('r_cn_pre_note');
  if (mode === 'none'){ note.textContent = 'Pick a ControlNet type first.'; return; }
  if (!imgPath){ note.textContent = 'Upload or choose a CN image first.'; return; }

  const payload = {
    cn_mode: mode,
    cn_image: imgPath,
    canny_low: parseInt(val('r_canny_low') || '100', 10),
    canny_high: parseInt(val('r_canny_high') || '200', 10)
  };
  try{
    note.textContent = 'Preprocessing…';
    const j = await fetchJSON('/cn/preprocess', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    if (!j || !j.ok){
      note.innerHTML = '<span class="err">Preprocess failed.</span>';
      return;
    }
    // Replace CN input with processed result
    rCnProcessedRel = j.cn_proc_rel || '';
    if (rCnProcessedRel){
      setVal('r_cn_image', rCnProcessedRel);
      rCnPreprocessed = true;
    }
    // Show preview image
    const box = document.getElementById('r_cn_processed');
    if (box){
      box.innerHTML = '';
      if (j.preview_url){
        const im = document.createElement('img');
        im.src = j.preview_url + '&t=' + Date.now();
        im.loading = 'lazy'; im.decoding = 'async';
        box.appendChild(im);
      }
    }
    note.innerHTML = '<span class="ok">Preprocessed ✔ (wired to workflow)</span>';
    log('CN preprocess result', j);
  }catch(e){
    note.innerHTML = '<span class="err">Preprocess error.</span>';
    logErr(e?.message || e);
  }
}

document.getElementById('r_cn_pre_btn').addEventListener('click', preprocessCN);

// invalidate preprocess state if user edits the CN image path manually
document.getElementById('r_cn_image').addEventListener('input', ()=>{
  rCnPreprocessed = false; rCnProcessedRel = "";
  document.getElementById('r_cn_pre_note')?.textContent = '';
  document.getElementById('r_cn_processed').innerHTML = '';
});

// -------- existing listeners --------
document.getElementById('btnIdeate').addEventListener('click', fireIdeate);
document.getElementById('btnRender').addEventListener('click', fireRender);
document.getElementById('btnUpscale').addEventListener('click', fireUpscale);

/* ===== IDEATE/RENDER CN helpers & listeners (attach once) ===== */

// Show Canny threshold inputs for Ideate only when mode === 'edges'
function updateICannyVisibility(){
  const m1 = (document.getElementById('i_c1mode')?.value || 'none');
  const m2 = (document.getElementById('i_c2mode')?.value || 'none');
  const r1 = document.getElementById('i_c1canny_row');
  const r2 = document.getElementById('i_c2canny_row');
  if (r1) r1.style.display = (m1 === 'edges') ? 'inline-flex' : 'none';
  if (r2) r2.style.display = (m2 === 'edges') ? 'inline-flex' : 'none';
}

// Reset/prep state for one Ideate CN slot (1 or 2)
function resetIdeateCn(slot){
  if (slot === 1){
    iC1Preprocessed = false; iC1ProcRel = "";
    const n1 = document.getElementById('i_c1pre_note'); if (n1) n1.textContent = '';
    const p1 = document.getElementById('i_c1proc_prev'); if (p1) p1.innerHTML = '';
  } else {
    iC2Preprocessed = false; iC2ProcRel = "";
    const n2 = document.getElementById('i_c2pre_note'); if (n2) n2.textContent = '';
    const p2 = document.getElementById('i_c2proc_prev'); if (p2) p2.innerHTML = '';
  }
  updateICannyVisibility();
}

// tiny helper: add listener only if the element exists
const on = (id, ev, cb) => { const el = document.getElementById(id); if (el) el.addEventListener(ev, cb); };

// --- Upscale upload
on('u_file', 'change', async ()=>{
  const r = await uploadTo('u_path','u_file','u_preview','u_file_note');
  if (r) uploadedUpscale = r;
});

// --- Ideate: mode changes invalidate preprocessing + toggle canny rows
on('i_c1mode', 'change', ()=>resetIdeateCn(1));
on('i_c2mode', 'change', ()=>resetIdeateCn(2));

// --- Ideate: changing image path invalidates preprocessing
on('i_c1img', 'input', ()=>resetIdeateCn(1));
on('i_c2img', 'input', ()=>resetIdeateCn(2));

// --- Ideate: preprocess buttons
on('i_c1pre_btn', 'click', ()=>preprocessIdeateCN(1));
on('i_c2pre_btn', 'click', ()=>preprocessIdeateCN(2));

// --- Ideate: file pickers → preview + wire path, then reset state
on('i_c1file', 'change', async ()=>{
  await onFileChange_map('i_c1img','i_c1file','i_c1prev','i_c1note');
  resetIdeateCn(1);
});
on('i_c2file', 'change', async ()=>{
  await onFileChange_map('i_c2img','i_c2file','i_c2prev','i_c2note');
  resetIdeateCn(2);
});

// --- Render: file pickers
on('r_in_file', 'change', ()=>onFileChange_map('r_in','r_in_file','r_in_prev','r_in_note'));
on('r_cn_file', 'change', async ()=>{
  await onFileChange_map('r_cn_image','r_cn_file','r_cn_prev','r_cn_note');
  // invalidate CN preprocess state
  rCnPreprocessed = false; rCnProcessedRel = "";
  const note = document.getElementById('r_cn_pre_note'); if (note) note.textContent = '';
  const box = document.getElementById('r_cn_processed'); if (box) box.innerHTML = '';
});

// initial sync for the Ideate Canny rows
updateICannyVisibility();

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
updateCannyRowVisibility();
updateICannyVisibility();
log('UI ready');
</script>
</main>
</body></html>
'''

# Serve the HTML UI that’s already stored in the HTML variable above
@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse(HTML)
PY

# --- Start ComfyUI ---
cd "$BASE"
nohup "$VENVPY" main.py --listen 0.0.0.0 --port 8188 >"$LOG" 2>&1 &

# --- Start FastAPI UI ---
cd "$BASE/simple-ui"
"$VENV/bin/pip" show uvicorn fastapi >/dev/null || \
  "$VENV/bin/pip" install --no-input fastapi uvicorn python-multipart
nohup "$VENVPY" -m uvicorn app:app --host 0.0.0.0 --port 9999 >/workspace/runpod-slim/ui.log 2>&1 &

echo "✅ ComfyUI running on :8188, UI running on :9999"
