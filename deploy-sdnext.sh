#!/usr/bin/env bash
###############################################################################
# deploy-sdnext.sh — v3.1.0 (2026 Production Edition)
#
# End-to-end automated deployment, front-end GPU provisioning, Google Drive &
# Phone Storage integration, multi-resource downloader, real-time output
# streamer, GPU idle cost guard, and self-healing supervisor for SD.Next
# (vladmandic/sdnext) on RunPod or any Ubuntu/Debian GPU host.
#
# =============================================================================
# MODES OF OPERATION
# =============================================================================
# 1. FRONT-END GPU SELECTOR & POD PROVISIONER (Run from your laptop / local PC)
#    Queries the RunPod GraphQL API for live GPU availability & pricing, ranks
#    GPUs against your requested model profile / VRAM tier, provisions a Pod
#    (or starts an existing one), injects all environment variables (including
#    your base64 rclone config, HF/CivitAI tokens, and Phone Storage settings),
#    and streams readiness:
#
#      export RUNPOD_API_KEY="rp_..."
#      ./deploy-sdnext.sh --provision --gpu-tier 24gb --cloud SECURE
#      ./deploy-sdnext.sh --wizard        # interactive front-end TUI wizard
#      ./deploy-sdnext.sh --list-gpus     # live GPU availability & price table
#
# 2. IN-POD AUTOMATED DEPLOYMENT & SELF-HEALING SUPERVISOR (Runs inside RunPod)
#    Preflight checks -> System deps (aria2, ffmpeg, rclone, zstd, inotify-tools,
#    libtcmalloc) -> Google Drive & Phone Storage setup -> SD.Next clone/update
#    -> uv-accelerated venv & installer -> 2026 extension suite -> Drive & Phone
#    restore -> Multi-resource download (Models, CivitAI API v1 resolver,
#    Utility Resource Pack, LoRAs, Embeddings, Upscalers, YOLOv8 detailers,
#    Wildcards) -> Hardware-tuned launch flags -> Real-time inotify output
#    streamer -> GPU Idle Cost Guard -> Self-healing supervised server with
#    OOM/disk/extension crash recovery & diagnostic crash bundles.
#
# 3. PHONE STORAGE INTEGRATION (3 Built-In Modes)
#    - Mode A (Phone as Remote Storage via Userspace Tailscale + rclone):
#        Connects to an SFTP/WebDAV/FTP server on your Android (Termux sshd,
#        Material Files, Cx File Explorer, Round Sync) or iOS (Secure ShellFish,
#        FE File Explorer) phone. Loads models/LoRAs/inputs from your phone on
#        boot and streams generated outputs back to your phone.
#    - Mode B (Pod as a Mounted Drive on Your Phone via WebDAV):
#        --phone-webdav starts an authenticated WebDAV server on port 7861 so
#        iOS native Files ("Connect to Server") or Android file managers can
#        mount the pod directly as a drive on your phone.
#    - Mode C (Peer-to-Peer Syncthing):
#        SYNCTHING_ENABLED=1 pairs with Syncthing-Fork (Android) or Mobius Sync
#        (iOS) via device ID over global encrypted relays (zero port forwarding).
#    Run `./deploy-sdnext.sh --phone-setup-guide` for step-by-step instructions.
#
# EXIT CODES:
#   0 = success               5 = extension sync failure
#   1 = generic / lock busy   6 = model download fatal failure
#   2 = invalid CLI usage     7 = server launch / health check failure
#   3 = preflight failure     8 = Google Drive / rclone fatal failure
#   4 = install/venv failure  9 = RunPod API provisioning failure
#  10 = Phone storage fatal failure
###############################################################################

set -Euo pipefail
IFS=$'\n\t'

SCRIPT_VERSION="3.1.0"

#-----------------------------------------------------------------------------
# Configuration (all overridable via environment variables or CLI flags)
#-----------------------------------------------------------------------------
: "${WORKSPACE_DIR:=/workspace}"
: "${SDNEXT_DIR:=${WORKSPACE_DIR}/sdnext}"
: "${SDNEXT_REPO:=https://github.com/vladmandic/sdnext.git}"
: "${SDNEXT_BRANCH:=master}"
: "${SDNEXT_COMMIT:=}"
: "${PYTHON_BIN:=python3}"
: "${BACKEND:=cuda}"                             # cuda | rocm | cpu
: "${PORT:=7860}"
: "${LISTEN:=1}"
: "${ENABLE_API:=1}"
: "${EXTRA_ARGS:=}"
: "${AUTO_VRAM_FLAGS:=1}"                        # 1 = auto-tune flags for detected GPU VRAM
: "${USE_UV:=1}"                                 # 1 = accelerate venv/pip with astral uv
: "${USE_TCMALLOC:=1}"                           # 1 = LD_PRELOAD libtcmalloc to prevent RAM fragmentation
: "${HF_HUB_ENABLE_HF_TRANSFER:=1}"              # 1 = Rust-accelerated HuggingFace downloads
: "${LOG_DIR:=${WORKSPACE_DIR}/logs}"
: "${LOG_RETENTION_DAYS:=14}"
: "${INSTALL_EXTENSIONS:=1}"
: "${UPDATE_EXTENSIONS:=0}"
: "${EXTENSIONS_EXTRA:=}"
: "${DOWNLOAD_MODELS:=1}"
: "${MODEL_PROFILE:=core}"                       # minimal | core | flux | video | full
: "${MODEL_URLS:=}"                              # extra space-separated URLs (or subdir::url / civitai://ID)
: "${DOWNLOAD_RESOURCE_PACK:=1}"                 # 1 = download essential upscalers, YOLOv8 detailers & embeddings
: "${LORA_URLS:=}"                               # space-separated URLs or civitai://VERSION_ID -> models/Lora
: "${EMBEDDING_URLS:=}"                          # space-separated URLs or civitai://VERSION_ID -> models/embeddings
: "${UPSCALER_URLS:=}"                           # space-separated URLs -> models/ESRGAN
: "${YOLO_URLS:=}"                               # space-separated URLs -> models/yolo
: "${WILDCARD_URLS:=}"                           # space-separated URLs -> wildcards
: "${HF_TOKEN:=}"
: "${CIVITAI_TOKEN:=}"
: "${HF_ENDPOINT:=}"
: "${RUN_MODE:=supervise}"                       # supervise | foreground | background
: "${SUPERVISE_MAX_BACKOFF:=300}"
: "${SUPERVISE_MAX_RESTARTS:=25}"
: "${HEALTH_TIMEOUT:=600}"
: "${RUNTIME_HEALTH_INTERVAL:=60}"               # seconds between active /sdapi/v1/memory liveness checks
: "${INSTANT_OUTPUT_SYNC:=1}"                    # 1 = real-time inotifywait streaming of new outputs to Drive/Phone
: "${IDLE_SHUTDOWN_MINUTES:=0}"                  # >0 = auto-backup & stop RunPod pod after N mins of GPU idle (<3%)
: "${IDLE_GPU_THRESHOLD_PCT:=3}"
: "${SEED_DEFAULT_CONFIG:=1}"                    # 1 = seed hardware-tuned config.json if none exists
: "${SKIP_INSTALL:=0}"
: "${SKIP_LAUNCH:=0}"
: "${DO_UPDATE:=0}"
: "${DO_REINSTALL:=0}"
: "${MIN_DISK_GB:=20}"
: "${MIN_RAM_GB:=8}"
: "${GRADIO_AUTH:=}"                             # user:pass for --gradio-auth
: "${API_AUTH:=}"                                # user:pass for --api-auth
: "${WEBHOOK_URL:=}"                             # Slack / Discord / generic JSON webhook URL
: "${SMOKE_TEST:=0}"
: "${DRY_RUN:=0}"

# Google Drive Integration (rclone)
: "${DRIVE_ENABLED:=1}"
: "${DRIVE_REMOTE:=gdrive}"
: "${DRIVE_ROOT:=SDNext}"
: "${DRIVE_SYNC_MINUTES:=15}"
: "${DRIVE_MOUNT_MODE:=sync}"                    # sync (bidirectional copy) | mount (FUSE mount) | hybrid
: "${DRIVE_SYNC_LORAS:=1}"                       # 1 = also restore/backup models/Lora, embeddings, wildcards
: "${DRIVE_EXPORT_LOGS:=1}"                      # 1 = push deploy & crash logs to Drive:SDNext/logs
: "${DRIVE_BWLIMIT:=}"                           # optional rclone bwlimit, e.g. "50M"
: "${RCLONE_CONFIG_B64:=}"
: "${RCLONE_CONFIG_FILE:=}"
: "${GDRIVE_SERVICE_ACCOUNT_B64:=}"              # base64 Google Cloud Service Account JSON (zero-OAuth mode)
: "${DRIVE_INTERACTIVE_AUTH:=0}"                 # 1 = run interactive headless rclone authorize wizard
: "${DRIVE_PUSH_LOCK:=${WORKSPACE_DIR}/.drive-push.lock}"

# Phone Storage Integration (Tailscale + rclone SFTP/WebDAV/FTP, Pod WebDAV Server, or Syncthing P2P)
: "${PHONE_ENABLED:=0}"                          # 1 = enable direct pull/push from Phone storage server
: "${PHONE_PROTOCOL:=sftp}"                      # sftp (Termux/ShellFish/MaterialFiles) | webdav | ftp
: "${PHONE_HOST:=}"                              # Phone IP (e.g. Tailscale 100.x.y.z or LAN IP)
: "${PHONE_PORT:=8022}"                          # 8022 = Termux sshd default; 8080 = WebDAV; 2121 = FTP
: "${PHONE_USER:=}"                              # Phone SSH/WebDAV username
: "${PHONE_PASS:=}"                              # Plaintext password (auto-obscured for rclone)
: "${PHONE_KEY_B64:=}"                           # Base64 Ed25519/RSA private key for passwordless phone SFTP
: "${PHONE_ROOT:=/sdcard/SDNext}"                # Base folder on phone (e.g. /sdcard/SDNext or /SDNext on iOS)
: "${PHONE_SYNC_MINUTES:=10}"                    # Background push interval to phone
: "${PHONE_PULL_MODELS:=1}"                      # 1 = pull checkpoints/LoRAs/embeddings from phone on boot
: "${PHONE_PULL_INPUTS:=1}"                      # 1 = pull init-images/reference photos from phone on boot
: "${PHONE_PUSH_OUTPUTS:=1}"                     # 1 = push generated images/videos directly to phone
: "${PHONE_MAX_PULL_GB:=50}"                     # Safety cap when pulling models from phone storage
: "${PHONE_PUSH_LOCK:=${WORKSPACE_DIR}/.phone-push.lock}"
: "${TAILSCALE_AUTH_KEY:=}"                      # tskey-auth-... joins pod to your Tailnet in userspace mode
: "${TAILSCALE_HOSTNAME:=sdnext-runpod}"
: "${POD_WEBDAV_ENABLED:=0}"                     # 1 = expose pod storage via WebDAV so phone mounts pod directly
: "${POD_WEBDAV_PORT:=7861}"
: "${POD_WEBDAV_USER:=sdnext}"
: "${POD_WEBDAV_PASS:=}"                         # Auto-generated if empty when POD_WEBDAV_ENABLED=1
: "${SYNCTHING_ENABLED:=0}"                      # 1 = run headless Syncthing P2P daemon for Android/iOS
: "${SYNCTHING_PHONE_ID:=}"                      # 56-char Syncthing Device ID of your phone
: "${SYNCTHING_GUI_PORT:=8384}"
: "${PHONE_SETUP_GUIDE_ONLY:=0}"
: "${PHONE_PULL_ONLY:=0}"
: "${PHONE_PUSH_ONLY:=0}"

# Front-End RunPod GPU Selection & Provisioning
: "${RUNPOD_API_KEY:=}"
: "${RUNPOD_API_URL:=https://api.runpod.io/graphql}"
: "${PROVISION_POD:=0}"                          # 1 = run front-end GPU selector & pod creator
: "${LIST_GPUS_ONLY:=0}"                         # 1 = query & display RunPod GPU table and exit
: "${WIZARD_MODE:=0}"                            # 1 = interactive TUI wizard for GPU + profile + storage
: "${GPU_TIER:=auto}"                            # auto | 16gb | 24gb | 48gb | 80gb
: "${PREFERRED_GPUS:=}"                          # comma-separated override list of RunPod GPU IDs
: "${MAX_USD_PER_HR:=2.50}"                      # max hourly GPU price filter
: "${CLOUD_TYPE:=ALL}"                           # SECURE | COMMUNITY | ALL
: "${POD_NAME:=sdnext-webui}"
: "${POD_IMAGE:=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
: "${POD_VOLUME_GB:=100}"
: "${POD_CONTAINER_DISK_GB:=40}"
: "${POD_MIN_DOWNLOAD_MBPS:=500}"
: "${NETWORK_VOLUME_ID:=}"                       # optional existing RunPod network volume ID
: "${DATACENTER_ID:=}"                           # optional RunPod datacenter ID (e.g. US-TX-3, EU-RO-1)

#-----------------------------------------------------------------------------
# Extension Suite (2026 SD.Next-verified)
# Note: SD.Next includes built-in ControlNet/Control, Detailer (ADetailer),
# Image Browser, Prompt enhancement, and Queue manager. These extensions add
# power-user workflows verified to work cleanly alongside SD.Next builtins.
#-----------------------------------------------------------------------------
CORE_EXTENSIONS=(
  "adetailer|https://github.com/Bing-su/adetailer.git"
  "ultimate-upscale-for-automatic1111|https://github.com/Coyote-A/ultimate-upscale-for-automatic1111.git"
  "a1111-sd-webui-tagcomplete|https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.git"
  "sd-dynamic-prompts|https://github.com/adieyal/sd-dynamic-prompts.git"
  "sd-webui-infinite-image-browsing|https://github.com/zanllp/sd-webui-infinite-image-browsing.git"
)

DETAIL_EXTENSIONS=(
  "sd-webui-regional-prompter|https://github.com/hako-mikan/sd-webui-regional-prompter.git"
  "multidiffusion-upscaler-for-automatic1111|https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111.git"
  "sd-webui-inpaint-anything|https://github.com/Uminosachi/sd-webui-inpaint-anything.git"
  "sd-webui-segment-anything|https://github.com/continue-revolution/sd-webui-segment-anything.git"
)

POPULAR_EXTENSIONS=(
  "stable-diffusion-webui-wd14-tagger|https://github.com/picobyte/stable-diffusion-webui-wd14-tagger.git"
  "Civitai-Helper|https://github.com/zixaphir/Stable-Diffusion-Webui-Civitai-Helper.git"
  "sd-webui-aspect-ratio-helper|https://github.com/thomasasfk/sd-webui-aspect-ratio-helper.git"
  "sd-webui-cutoff|https://github.com/hnmr293/sd-webui-cutoff.git"
  "sd-webui-lora-block-weight|https://github.com/hako-mikan/sd-webui-lora-block-weight.git"
)

#-----------------------------------------------------------------------------
# Curated Utility Resource Pack (Upscalers, YOLOv8 Detailers, Core Embeddings)
# Format: id|url|target_subdir|note
#-----------------------------------------------------------------------------
UTILITY_RESOURCE_PACK=(
  "4x-ultrasharp|https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth|ESRGAN|4x-UltraSharp photorealistic upscaler"
  "4x-animesharp|https://huggingface.co/Kim2091/AnimeSharp/resolve/main/4x-AnimeSharp.pth|ESRGAN|4x-AnimeSharp line/illustration upscaler"
  "yolo-face-v8m|https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt|yolo|YOLOv8m Face Detailer detector"
  "yolo-hand-v8n|https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt|yolo|YOLOv8n Hand Detailer detector"
  "yolo-person-v8m-seg|https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt|yolo|YOLOv8m Person Segmentation detector"
  "easynegative|https://huggingface.co/datasets/gsdf/EasyNegative/resolve/main/EasyNegative.safetensors|embeddings|EasyNegative universal negative embedding"
)

#-----------------------------------------------------------------------------
# Model Library Catalog (2026)
# Format: id|kind|source|includes|target_subdir|size_gb|flags|note
#   kind : file (single URL) | hf (HuggingFace repo snapshot)
#   flags: gated (requires HF_TOKEN) | besteffort (non-fatal if unavailable)
#-----------------------------------------------------------------------------
MODEL_CATALOG=(
  "sd15|file|https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors||Stable-diffusion|4||SD 1.5 pruned EMA-only"
  "sdxl-base|file|https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors||Stable-diffusion|7||SDXL Base 1.0"
  "sdxl-vae-fp16|file|https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors||VAE|0.3||SDXL VAE FP16 fix (prevents black images)"
  "flux1-schnell-fp8|file|https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors||Stable-diffusion|12||FLUX.1 Schnell FP8 all-in-one (Apache 2.0, fast 4-step)"
  "flux1-dev|file|https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors||Stable-diffusion|24|gated|FLUX.1 Dev (requires HF_TOKEN + license)"
  "flux2-klein-4b-fp8|file|https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors||Stable-diffusion|4.1|besteffort|FLUX.2 Klein 4B FP8 (Apache 2.0)"
  "flux2-vae|file|https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/main/split_files/vae/flux2-vae.safetensors||VAE|0.4|besteffort|FLUX.2 VAE"
  "flux2-dev|file|https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/main/flux2-dev.safetensors||Stable-diffusion|48|gated,besteffort|FLUX.2 Dev (requires HF_TOKEN + 24-48GB VRAM)"
  "flux3-klein|hf|black-forest-labs/flux-3-image-klein|model.safetensors config.json manifest.json ae.safetensors text_encoder/* README.md LICENSE.md NOTICE.md|diffusers/flux-3-image-klein|21|gated,besteffort|FLUX 3 Image Klein distilled 5.6B (txt2img+editing)"
  "qwen-image-2512|hf|Qwen/Qwen-Image-2512|*.safetensors *.json|diffusers/qwen-image-2512|45|besteffort|Qwen-Image 2512 (Apache 2.0)"
  "qwen-image-edit-2511|hf|Qwen/Qwen-Image-Edit-2511|*.safetensors *.json|diffusers/qwen-image-edit-2511|45|besteffort|Qwen-Image-Edit 2511"
  "wan22-t2v|hf|Wan-AI/Wan2.2-T2V-A14B|*.safetensors *.json|diffusers/wan22-t2v|30|besteffort|Wan 2.2 Text-to-Video 14B"
  "hunyuan-video|hf|tencent/HunyuanVideo|*.safetensors *.json|diffusers/hunyuan-video|28|besteffort|HunyuanVideo T2V (SD.Next native diffusers)"
)

MINIMAL_MODEL_IDS="sdxl-base sdxl-vae-fp16"
CORE_MODEL_IDS="sd15 sdxl-base sdxl-vae-fp16 flux1-schnell-fp8 flux2-klein-4b-fp8 flux2-vae"
FLUX_MODEL_IDS="sdxl-base sdxl-vae-fp16 flux1-schnell-fp8 flux1-dev flux2-klein-4b-fp8 flux2-vae flux3-klein"
VIDEO_MODEL_IDS="sdxl-base sdxl-vae-fp16 flux1-schnell-fp8 wan22-t2v hunyuan-video"

#-----------------------------------------------------------------------------
# CLI Help & Argument Parsing
#-----------------------------------------------------------------------------
usage() {
  cat <<'EOF'
Usage: deploy-sdnext.sh [options]

Front-End GPU Selection & RunPod Provisioning (run from laptop or control node):
  --wizard                 Interactive TUI wizard (select GPU, profile, Drive & Phone sync)
  --list-gpus              Query RunPod API for live GPU availability/pricing and exit
  --provision              Auto-select optimal available GPU on RunPod and deploy pod
  --gpu-tier TIER          auto (default) | 16gb | 24gb | 48gb | 80gb
  --max-price USD          Maximum hourly GPU price in USD (default: 2.50)
  --cloud TYPE             SECURE | COMMUNITY | ALL (default: ALL)
  --volume-id ID           Attach existing RunPod Network Volume ID
  --pod-name NAME          RunPod pod name (default: sdnext-webui)

In-Pod Installation & Lifecycle:
  --update                 git pull SD.Next and all extensions before launching
  --reinstall              Wipe venv and force a full dependency reinstall
  --skip-install           Skip installer step (launch an existing install)
  --skip-launch            Install everything and sync storage, but do not start server
  --skip-extensions        Do not clone/update extensions
  --skip-models            Do not download the model library
  --no-resource-pack       Skip the utility resource pack (upscalers, YOLOv8 detailers, embeddings)
  --model-profile P        minimal | core (default) | flux | video | full
  --lora-urls "U1 U2"      Extra LoRA URLs or civitai://VERSION_IDs to download
  --supervise              Restart server automatically on crash with self-healing (default)
  --background, -b         Single run in background, health-check, then exit
  --foreground             Single run in foreground (no auto-restart)
  --idle-shutdown MIN      Auto-backup and stop RunPod pod after MIN minutes of GPU idle (<3%)
  --port N                 WebUI port (default: 7860)
  --extra-args "..."       Extra CLI args appended to webui.py
  --smoke-test             Generate a test image via REST API after first start
  --dry-run                Print full execution plan and exit without changes
  --workspace DIR          Workspace root (default: /workspace)
  --branch NAME            SD.Next git branch (default: master)

Google Drive Integration:
  --no-drive               Disable Google Drive sync
  --drive-auth             Interactive headless Google Drive OAuth wizard
  --drive-mode MODE        sync (default) | mount (rclone FUSE mount) | hybrid
  --drive-pull             Run one-shot restore from Google Drive and exit
  --drive-push             Run one-shot backup to Google Drive and exit

Phone Storage Integration (Android & iOS):
  --phone-setup-guide      Print step-by-step Android & iOS phone storage setup guide and exit
  --phone-host IP          Enable phone storage sync at IP (Tailscale 100.x.y.z or LAN IP)
  --phone-port PORT        Phone server port (default: 8022 for Termux SFTP, 8080 for WebDAV)
  --phone-protocol PROTO   sftp (default) | webdav | ftp
  --phone-user USER        Phone SFTP/WebDAV username
  --phone-path PATH        Folder path on phone (default: /sdcard/SDNext)
  --phone-webdav           Start authenticated WebDAV server on pod (port 7861) for iOS/Android
  --phone-pull             Run one-shot pull from Phone storage and exit
  --phone-push             Run one-shot push to Phone storage and exit
  -h, --help               Show this help
EOF
}

DRIVE_PULL_ONLY=0
DRIVE_PUSH_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wizard)             WIZARD_MODE=1; shift ;;
    --list-gpus)          LIST_GPUS_ONLY=1; shift ;;
    --provision)          PROVISION_POD=1; shift ;;
    --gpu-tier)           GPU_TIER="$2"; shift 2 ;;
    --max-price)          MAX_USD_PER_HR="$2"; shift 2 ;;
    --cloud)              CLOUD_TYPE="$2"; shift 2 ;;
    --volume-id)          NETWORK_VOLUME_ID="$2"; shift 2 ;;
    --pod-name)           POD_NAME="$2"; shift 2 ;;
    --update)             DO_UPDATE=1; UPDATE_EXTENSIONS=1; shift ;;
    --reinstall)          DO_REINSTALL=1; shift ;;
    --skip-install)       SKIP_INSTALL=1; shift ;;
    --skip-launch)        SKIP_LAUNCH=1; shift ;;
    --skip-extensions)    INSTALL_EXTENSIONS=0; shift ;;
    --skip-models)        DOWNLOAD_MODELS=0; shift ;;
    --no-resource-pack)   DOWNLOAD_RESOURCE_PACK=0; shift ;;
    --model-profile)      MODEL_PROFILE="$2"; shift 2 ;;
    --lora-urls)          LORA_URLS="$2"; shift 2 ;;
    --supervise)          RUN_MODE=supervise; shift ;;
    --background|-b)      RUN_MODE=background; shift ;;
    --foreground)         RUN_MODE=foreground; shift ;;
    --idle-shutdown)      IDLE_SHUTDOWN_MINUTES="$2"; shift 2 ;;
    --no-drive)           DRIVE_ENABLED=0; shift ;;
    --drive-auth)         DRIVE_INTERACTIVE_AUTH=1; shift ;;
    --drive-mode)         DRIVE_MOUNT_MODE="$2"; shift 2 ;;
    --drive-pull)         DRIVE_PULL_ONLY=1; shift ;;
    --drive-push)         DRIVE_PUSH_ONLY=1; shift ;;
    --phone-setup-guide)  PHONE_SETUP_GUIDE_ONLY=1; shift ;;
    --phone-host)         PHONE_ENABLED=1; PHONE_HOST="$2"; shift 2 ;;
    --phone-port)         PHONE_PORT="$2"; shift 2 ;;
    --phone-protocol)     PHONE_PROTOCOL="$2"; shift 2 ;;
    --phone-user)         PHONE_USER="$2"; shift 2 ;;
    --phone-path)         PHONE_ROOT="$2"; shift 2 ;;
    --phone-webdav)       POD_WEBDAV_ENABLED=1; shift ;;
    --phone-pull)         PHONE_ENABLED=1; PHONE_PULL_ONLY=1; shift ;;
    --phone-push)         PHONE_ENABLED=1; PHONE_PUSH_ONLY=1; shift ;;
    --port)               PORT="$2"; shift 2 ;;
    --extra-args)         EXTRA_ARGS="$2"; shift 2 ;;
    --dry-run)            DRY_RUN=1; shift ;;
    --smoke-test)         SMOKE_TEST=1; shift ;;
    --workspace)          WORKSPACE_DIR="$2"; SDNEXT_DIR="${WORKSPACE_DIR}/sdnext"; LOG_DIR="${WORKSPACE_DIR}/logs"; DRIVE_PUSH_LOCK="${WORKSPACE_DIR}/.drive-push.lock"; PHONE_PUSH_LOCK="${WORKSPACE_DIR}/.phone-push.lock"; shift 2 ;;
    --branch)             SDNEXT_BRANCH="$2"; shift 2 ;;
    -h|--help)            usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

#-----------------------------------------------------------------------------
# Structured Logging (Human ANSI + Plain Log + Machine-Parsable JSONL) & Traps
#-----------------------------------------------------------------------------
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
  WORKSPACE_DIR="${TMPDIR:-/tmp}/sdnext-workspace"
  SDNEXT_DIR="${WORKSPACE_DIR}/sdnext"
  LOG_DIR="${WORKSPACE_DIR}/logs"
  DRIVE_PUSH_LOCK="${WORKSPACE_DIR}/.drive-push.lock"
  PHONE_PUSH_LOCK="${WORKSPACE_DIR}/.phone-push.lock"
  mkdir -p "$LOG_DIR"
fi

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/sdnext-deploy-${RUN_STAMP}.log"
JSON_LOG_FILE="${LOG_DIR}/sdnext-deploy-${RUN_STAMP}.jsonl"
ln -sfn "$LOG_FILE" "${LOG_DIR}/sdnext-deploy-latest.log"
ln -sfn "$JSON_LOG_FILE" "${LOG_DIR}/sdnext-deploy-latest.jsonl"
START_EPOCH="$(date +%s)"
CURRENT_STAGE="init"

_use_color=0
[[ -t 2 ]] && _use_color=1
_c()   { [[ $_use_color -eq 1 ]] && printf '\033[%sm' "$1" || true; }
_rst() { [[ $_use_color -eq 1 ]] && printf '\033[0m' || true; }

_json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  printf '%s' "$s"
}

_log() { # level color msg...
  local level="$1" color="$2"; shift 2
  local IFS=' '
  local msg="$*"
  local ts iso_ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  iso_ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf '%s [%s] [%s] %s\n' "$ts" "$level" "$CURRENT_STAGE" "$msg" >>"$LOG_FILE"
  printf '{"timestamp":"%s","level":"%s","stage":"%s","pod_id":"%s","message":"%s"}\n' \
    "$iso_ts" "$level" "$CURRENT_STAGE" "${RUNPOD_POD_ID:-local}" "$(_json_escape "$msg")" >>"$JSON_LOG_FILE"
  printf '%s[%s %-5s %-12s]%s %s\n' "$(_c "$color")" "$ts" "$level" "$CURRENT_STAGE" "$(_rst)" "$msg" >&2
}

log_info()  { _log "INFO"  "32" "$@"; }
log_warn()  { _log "WARN"  "33" "$@"; }
log_error() { _log "ERROR" "31" "$@"; }
log_debug() { _log "DEBUG" "36" "$@"; }
set_stage() { CURRENT_STAGE="$1"; log_info "=== Stage: $1 ==="; }

notify() { # status detail...
  [[ -n "${WEBHOOK_URL:-}" ]] || return 0
  local status="$1"; shift
  local IFS=' '
  local text="[sdnext-deploy v${SCRIPT_VERSION} | pod=${RUNPOD_POD_ID:-local}] ${status}: $*"
  local escaped
  escaped="$(_json_escape "$text")"
  curl -fsS -m 15 -X POST -H 'Content-Type: application/json' \
    --data "{\"text\":\"${escaped}\",\"content\":\"${escaped}\"}" \
    "$WEBHOOK_URL" >/dev/null 2>&1 || log_warn "Webhook notification failed"
}

die() { # exit_code message...
  local code="$1"; shift
  log_error "$@"
  notify "FAILED" "$* (rc=${code}, stage=${CURRENT_STAGE})"
  if declare -F drive_export_logs >/dev/null 2>&1; then
    drive_export_logs || true
  fi
  if declare -F phone_export_logs >/dev/null 2>&1; then
    phone_export_logs || true
  fi
  exit "$code"
}

on_error() {
  local rc=$?
  local line="${1:-?}"
  local cmd="${2:-?}"
  log_error "Command failed (rc=${rc}) at line ${line} in stage '${CURRENT_STAGE}': ${cmd}"
  local i=0
  while caller $i >/dev/null 2>&1; do
    local frame
    frame="$(caller $i)"
    log_debug "  stack[$i]: ${frame}"
    i=$(( i + 1 ))
  done
  log_error "Logs: ${LOG_FILE} (JSONL: ${JSON_LOG_FILE})"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

on_exit() {
  local rc=$?
  local pid_var
  for pid_var in SYNC_PID PHONE_SYNC_PID INOTIFY_PID IDLE_GUARD_PID WEBDAV_PID SYNCTHING_PID; do
    local p="${!pid_var:-}"
    if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then
      kill "$p" 2>/dev/null || true
    fi
  done
  if [[ -n "${FUSE_MOUNT_PID:-}" ]] && kill -0 "$FUSE_MOUNT_PID" 2>/dev/null; then
    fusermount -u "${WORKSPACE_DIR}/gdrive-mount" 2>/dev/null || kill "$FUSE_MOUNT_PID" 2>/dev/null || true
  fi
  local dur=$(( $(date +%s) - START_EPOCH ))
  log_info "deploy-sdnext.sh v${SCRIPT_VERSION} exiting with rc=${rc} after ${dur}s (log: ${LOG_FILE})"
}
trap on_exit EXIT

#-----------------------------------------------------------------------------
# General Helpers
#-----------------------------------------------------------------------------
require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die 3 "Required command '$1' not found in PATH"
}

retry() { # attempts command...
  local attempts="$1"; shift
  local delay=4 n=1
  while true; do
    if "$@"; then return 0; fi
    if (( n >= attempts )); then
      log_error "Command failed after ${attempts} attempts: $*"
      return 1
    fi
    local jitter=$(( RANDOM % 3 ))
    local wait_s=$(( delay + jitter ))
    log_warn "Attempt ${n}/${attempts} failed: $* — retrying in ${wait_s}s"
    sleep "$wait_s"
    delay=$(( delay * 2 ))
    n=$(( n + 1 ))
  done
}

ver_ge() {
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo "$@"
  else
    log_warn "No root/passwordless sudo available for: $* (skipping privileged operation)"
    return 1
  fi
}

model_in_active_profile() { # id -> 0 if included in MODEL_PROFILE
  local id="$1"
  case "$MODEL_PROFILE" in
    full)    return 0 ;;
    minimal) [[ " $MINIMAL_MODEL_IDS " == *" $id "* ]] && return 0 || return 1 ;;
    core)    [[ " $CORE_MODEL_IDS "    == *" $id "* ]] && return 0 || return 1 ;;
    flux)    [[ " $FLUX_MODEL_IDS "    == *" $id "* ]] && return 0 || return 1 ;;
    video)   [[ " $VIDEO_MODEL_IDS "   == *" $id "* ]] && return 0 || return 1 ;;
    *)       die 2 "Unknown MODEL_PROFILE '${MODEL_PROFILE}' (minimal|core|flux|video|full)" ;;
  esac
}

catalog_total_gb() {
  local total=0 entry id size flags
  local IFS='|'
  for entry in "${MODEL_CATALOG[@]}"; do
    read -r id _ _ _ _ size flags _ <<< "$entry"
    [[ "$flags" == *"disabled"* ]] && continue
    model_in_active_profile "$id" || continue
    total=$(awk "BEGIN{print $total + $size}")
  done
  printf '%.0f' "$total"
}

recommended_vram_gb_for_profile() {
  case "$MODEL_PROFILE" in
    minimal) echo 12 ;;
    core)    echo 24 ;;
    flux)    echo 24 ;;
    video)   echo 48 ;;
    full)    echo 48 ;;
    *)       echo 24 ;;
  esac
}

#-----------------------------------------------------------------------------
# Front-End RunPod GPU Selection, Live Pricing & Pod Provisioning
#-----------------------------------------------------------------------------
# Priority ranking per VRAM tier (best price/performance & Ada/Blackwell/Hopper first)
GPU_PRIORITY_16GB=(
  "NVIDIA RTX 4000 Ada Generation"
  "NVIDIA RTX A4000"
  "NVIDIA GeForce RTX 4080"
  "NVIDIA Tesla T4"
)
GPU_PRIORITY_24GB=(
  "NVIDIA GeForce RTX 4090"
  "NVIDIA RTX 4500 Ada Generation"
  "NVIDIA GeForce RTX 3090"
  "NVIDIA GeForce RTX 3090 Ti"
  "NVIDIA RTX A5000"
  "NVIDIA L4"
)
GPU_PRIORITY_48GB=(
  "NVIDIA RTX 6000 Ada Generation"
  "NVIDIA L40S"
  "NVIDIA L40"
  "NVIDIA A40"
  "NVIDIA RTX A6000"
)
GPU_PRIORITY_80GB=(
  "NVIDIA H100 80GB HBM3"
  "NVIDIA H100 PCIe"
  "NVIDIA A100-SXM4-80GB"
  "NVIDIA A100 80GB PCIe"
)

runpod_graphql() { # query_json
  local payload="$1"
  [[ -n "${RUNPOD_API_KEY:-}" ]] || die 9 "RUNPOD_API_KEY is not set. Export RUNPOD_API_KEY='rp_...' first."
  local resp
  resp="$(curl -fsS -m 30 -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    --data "$payload" \
    "$RUNPOD_API_URL" 2>/dev/null)" || die 9 "RunPod GraphQL API request failed"
  if printf '%s' "$resp" | grep -q '"errors"'; then
    log_error "RunPod API returned error: ${resp}"
    return 1
  fi
  printf '%s' "$resp"
}

fetch_runpod_gpus_json() {
  local query='{"query":"query GpuTypes { gpuTypes { id displayName memoryInGb secureCloud communityCloud securePrice communityPrice lowestPrice(input: {gpuCount: 1}) { minimumBidPrice uninterruptablePrice stockStatus } } }"}'
  runpod_graphql "$query"
}

list_and_rank_runpod_gpus() { # mode: table | select
  local mode="${1:-table}"
  set_stage "gpu-selector"
  local target_vram
  case "$GPU_TIER" in
    16gb|16GB) target_vram=16 ;;
    24gb|24GB) target_vram=24 ;;
    48gb|48GB) target_vram=48 ;;
    80gb|80GB) target_vram=80 ;;
    auto|*)    target_vram="$(recommended_vram_gb_for_profile)" ;;
  esac

  log_info "Querying RunPod GraphQL API for available GPUs (min_vram=${target_vram}GB, max_price=\$${MAX_USD_PER_HR}/hr, cloud=${CLOUD_TYPE})..."
  local raw_json
  raw_json="$(fetch_runpod_gpus_json)"

  "$PYTHON_BIN" - "$mode" "$target_vram" "$MAX_USD_PER_HR" "$CLOUD_TYPE" "${PREFERRED_GPUS:-}" "$raw_json" <<'PYEOF'
import sys, json

mode = sys.argv[1]
min_vram = int(sys.argv[2])
max_price = float(sys.argv[3])
cloud_type = sys.argv[4].upper()
preferred_csv = sys.argv[5]
data = json.loads(sys.argv[6])

gpus = (data.get("data") or {}).get("gpuTypes") or []

PRIORITY_ORDER = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA L40S",
    "NVIDIA L40",
    "NVIDIA A40",
    "NVIDIA RTX A6000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    "NVIDIA L4",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX A4000",
]
if preferred_csv.strip():
    custom = [x.strip() for x in preferred_csv.split(",") if x.strip()]
    PRIORITY_ORDER = custom + [x for x in PRIORITY_ORDER if x not in custom]

candidates = []
for g in gpus:
    mem = g.get("memoryInGb") or 0
    if mem < min_vram:
        continue
    lp = g.get("lowestPrice") or {}
    stock = (lp.get("stockStatus") or "Unknown")
    sec_p = g.get("securePrice")
    com_p = g.get("communityPrice")
    unint_p = lp.get("uninterruptablePrice")

    price = None
    chosen_cloud = None
    if cloud_type == "SECURE":
        if g.get("secureCloud") and sec_p:
            price = float(sec_p)
            chosen_cloud = "SECURE"
    elif cloud_type == "COMMUNITY":
        if g.get("communityCloud") and com_p:
            price = float(com_p)
            chosen_cloud = "COMMUNITY"
    else:
        options = []
        if g.get("secureCloud") and sec_p:
            options.append((float(sec_p), "SECURE"))
        if g.get("communityCloud") and com_p:
            options.append((float(com_p), "COMMUNITY"))
        if unint_p and not options:
            options.append((float(unint_p), "ALL"))
        if options:
            options.sort(key=lambda x: x[0])
            price, chosen_cloud = options[0]

    if price is None or price <= 0 or price > max_price:
        continue
    if stock and stock.lower() in ("out of stock", "none", "unavailable"):
        continue

    gid = g.get("id")
    prio = PRIORITY_ORDER.index(gid) if gid in PRIORITY_ORDER else 999
    stock_rank = {"high": 0, "medium": 1, "low": 2}.get((stock or "").lower(), 3)
    candidates.append({
        "id": gid,
        "name": g.get("displayName") or gid,
        "vram": mem,
        "price": price,
        "cloud": chosen_cloud,
        "stock": stock or "Available",
        "prio": prio,
        "stock_rank": stock_rank,
    })

candidates.sort(key=lambda c: (c["prio"], c["stock_rank"], c["price"]))

if mode == "table":
    print(f"{'GPU ID':<34} {'VRAM':>6} {'$/HR':>8} {'CLOUD':>10} {'STOCK':>10}")
    print("-" * 72)
    for c in candidates:
        print(f"{c['id']:<34} {str(c['vram'])+'GB':>6} {'$'+format(c['price'],'.2f'):>8} {c['cloud']:>10} {c['stock']:>10}")
    if not candidates:
        sys.exit(1)
elif mode == "select":
    if not candidates:
        sys.exit(1)
    best = candidates[0]
    print(f"{best['id']}|{best['vram']}|{best['price']:.2f}|{best['cloud']}|{best['stock']}")
PYEOF
}

provision_runpod_pod() {
  set_stage "provision"
  local selection
  selection="$(list_and_rank_runpod_gpus select)" \
    || die 9 "No available RunPod GPU matched criteria (tier=${GPU_TIER}, max=\$${MAX_USD_PER_HR}/hr, cloud=${CLOUD_TYPE})"

  local gpu_id gpu_vram gpu_price gpu_cloud gpu_stock
  IFS='|' read -r gpu_id gpu_vram gpu_price gpu_cloud gpu_stock <<< "$selection"
  log_info "Selected GPU: ${gpu_id} (${gpu_vram}GB VRAM, \$${gpu_price}/hr on ${gpu_cloud} cloud, stock=${gpu_stock})"

  if (( DRY_RUN )); then
    log_info "[DRY-RUN] Would provision RunPod pod '${POD_NAME}' with GPU '${gpu_id}' (${gpu_cloud}), volume=${POD_VOLUME_GB}GB, image=${POD_IMAGE}"
    return 0
  fi

  local self_b64=""
  local self_path
  self_path="$(readlink -f "$0" 2>/dev/null || echo "$0")"
  if [[ -f "$self_path" ]]; then
    self_b64="$(base64 -w0 "$self_path" 2>/dev/null || base64 "$self_path" | tr -d '\n')"
  fi

  local payload
  payload="$("$PYTHON_BIN" - "$gpu_id" "$gpu_cloud" "$POD_NAME" "$POD_IMAGE" "$POD_VOLUME_GB" "$POD_CONTAINER_DISK_GB" "$PORT" "$MODEL_PROFILE" "$DRIVE_ENABLED" "$DRIVE_REMOTE" "$DRIVE_ROOT" "${RCLONE_CONFIG_B64:-}" "${HF_TOKEN:-}" "${CIVITAI_TOKEN:-}" "${GRADIO_AUTH:-}" "${API_AUTH:-}" "${WEBHOOK_URL:-}" "${NETWORK_VOLUME_ID:-}" "${DATACENTER_ID:-}" "$self_b64" "$PHONE_ENABLED" "$PHONE_PROTOCOL" "$PHONE_HOST" "$PHONE_PORT" "$PHONE_USER" "$PHONE_PASS" "$PHONE_ROOT" "$TAILSCALE_AUTH_KEY" "$POD_WEBDAV_ENABLED" "$IDLE_SHUTDOWN_MINUTES" <<'PYEOF'
import sys, json

(gpu_id, cloud_type, pod_name, image_name, vol_gb, disk_gb, port,
 model_profile, drive_enabled, drive_remote, drive_root, rclone_b64,
 hf_token, civitai_token, gradio_auth, api_auth, webhook_url,
 net_vol_id, dc_id, script_b64, phone_enabled, phone_proto, phone_host,
 phone_port, phone_user, phone_pass, phone_root, ts_key, pod_webdav, idle_min) = sys.argv[1:31]

env_list = [
    {"key": "PORT", "value": str(port)},
    {"key": "MODEL_PROFILE", "value": model_profile},
    {"key": "DRIVE_ENABLED", "value": str(drive_enabled)},
    {"key": "DRIVE_REMOTE", "value": drive_remote},
    {"key": "DRIVE_ROOT", "value": drive_root},
    {"key": "PHONE_ENABLED", "value": str(phone_enabled)},
    {"key": "PHONE_PROTOCOL", "value": phone_proto},
    {"key": "PHONE_PORT", "value": str(phone_port)},
    {"key": "PHONE_ROOT", "value": phone_root},
    {"key": "POD_WEBDAV_ENABLED", "value": str(pod_webdav)},
    {"key": "IDLE_SHUTDOWN_MINUTES", "value": str(idle_min)},
]
for k, v in [
    ("RCLONE_CONFIG_B64", rclone_b64),
    ("HF_TOKEN", hf_token),
    ("CIVITAI_TOKEN", civitai_token),
    ("GRADIO_AUTH", gradio_auth),
    ("API_AUTH", api_auth),
    ("WEBHOOK_URL", webhook_url),
    ("PHONE_HOST", phone_host),
    ("PHONE_USER", phone_user),
    ("PHONE_PASS", phone_pass),
    ("TAILSCALE_AUTH_KEY", ts_key),
]:
    if v:
        env_list.append({"key": k, "value": v})

docker_args = ""
if script_b64:
    docker_args = f"bash -c 'echo {script_b64} | base64 -d > /workspace/deploy-sdnext.sh && chmod +x /workspace/deploy-sdnext.sh && exec /workspace/deploy-sdnext.sh'"

inp = {
    "cloudType": cloud_type,
    "gpuCount": 1,
    "volumeInGb": int(vol_gb),
    "containerDiskInGb": int(disk_gb),
    "minVcpuCount": 4,
    "minMemoryInGb": 16,
    "gpuTypeId": gpu_id,
    "name": pod_name,
    "imageName": image_name,
    "dockerArgs": docker_args,
    "ports": f"{port}/http,7861/http,22/tcp",
    "volumeMountPath": "/workspace",
    "env": env_list,
}
if net_vol_id:
    inp["networkVolumeId"] = net_vol_id
if dc_id:
    inp["dataCenterId"] = dc_id

query = """
mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    name
    imageName
    machineId
    machine {
      gpuDisplayName
    }
  }
}
"""
print(json.dumps({"query": query, "variables": {"input": inp}}))
PYEOF
)"

  log_info "Deploying pod '${POD_NAME}' on RunPod..."
  local resp
  resp="$(runpod_graphql "$payload")" || die 9 "Pod deployment mutation failed"
  local pod_id
  pod_id="$("$PYTHON_BIN" -c 'import sys,json; d=json.loads(sys.argv[1]); print(((d.get("data") or {}).get("podFindAndDeployOnDemand") or {}).get("id") or "")' "$resp")"
  [[ -n "$pod_id" ]] || die 9 "RunPod did not return a pod ID. Raw response: ${resp}"

  log_info "RunPod Pod created! ID: ${pod_id}"
  log_info "WebUI Proxy URL  : https://${pod_id}-${PORT}.proxy.runpod.net"
  if (( POD_WEBDAV_ENABLED )); then
    log_info "Phone WebDAV URL : https://${pod_id}-${POD_WEBDAV_PORT}.proxy.runpod.net"
  fi
  notify "PROVISIONED" "Pod ${pod_id} (${gpu_id} @ \$${gpu_price}/hr) -> https://${pod_id}-${PORT}.proxy.runpod.net"
}

run_interactive_wizard() {
  set_stage "wizard"
  echo "=================================================================="
  echo " SD.Next RunPod Front-End Provisioning & Deployment Wizard (v${SCRIPT_VERSION})"
  echo "=================================================================="
  read -r -p "1) Where are we running? [1=Provision new RunPod Pod from here, 2=Deploy inside current machine/Pod] (default 2): " mode_choice
  read -r -p "2) Select Model Profile [minimal | core | flux | video | full] (default ${MODEL_PROFILE}): " prof_choice
  [[ -n "$prof_choice" ]] && MODEL_PROFILE="$prof_choice"

  read -r -p "3) Enable Google Drive bidirectional sync? [Y/n]: " gd_choice
  if [[ "${gd_choice:-Y}" =~ ^[Nn] ]]; then
    DRIVE_ENABLED=0
  else
    DRIVE_ENABLED=1
    if [[ -z "$RCLONE_CONFIG_B64" && ! -f "${WORKSPACE_DIR}/.rclone.conf" && -f "${HOME}/.config/rclone/rclone.conf" ]]; then
      log_info "Detected local ~/.config/rclone/rclone.conf — encoding into RCLONE_CONFIG_B64 automatically"
      RCLONE_CONFIG_B64="$(base64 -w0 "${HOME}/.config/rclone/rclone.conf" 2>/dev/null || base64 "${HOME}/.config/rclone/rclone.conf" | tr -d '\n')"
    fi
  fi

  read -r -p "4) Integrate Phone Storage? [1=No, 2=Pull/Push to Phone Server (SFTP/WebDAV), 3=Mount Pod on Phone via WebDAV] (default 1): " ph_choice
  case "${ph_choice:-1}" in
    2)
      PHONE_ENABLED=1
      read -r -p "   Phone IP (Tailscale 100.x.y.z or LAN IP): " ph_host
      PHONE_HOST="$ph_host"
      read -r -p "   Phone Protocol [sftp | webdav | ftp] (default sftp): " ph_proto
      [[ -n "$ph_proto" ]] && PHONE_PROTOCOL="$ph_proto"
      read -r -p "   Phone Port (default 8022 for Termux SFTP, 8080 for WebDAV): " ph_port
      [[ -n "$ph_port" ]] && PHONE_PORT="$ph_port"
      read -r -p "   Phone Username: " ph_user
      PHONE_USER="$ph_user"
      read -r -s -p "   Phone Password: " ph_pass; echo ""
      PHONE_PASS="$ph_pass"
      if [[ "$PHONE_HOST" == 100.* && -z "$TAILSCALE_AUTH_KEY" ]]; then
        read -r -p "   Tailscale Auth Key (tskey-auth-..., needed so Pod can reach your phone): " ts_k
        TAILSCALE_AUTH_KEY="$ts_k"
      fi
      ;;
    3)
      POD_WEBDAV_ENABLED=1
      ;;
  esac

  if [[ "${mode_choice:-2}" == "1" ]]; then
    if [[ -z "$RUNPOD_API_KEY" ]]; then
      read -r -s -p "Enter your RUNPOD_API_KEY: " rp_key
      echo ""
      export RUNPOD_API_KEY="$rp_key"
    fi
    read -r -p "Select GPU VRAM tier [auto | 16gb | 24gb | 48gb | 80gb] (default auto): " tier_choice
    [[ -n "$tier_choice" ]] && GPU_TIER="$tier_choice"
    read -r -p "Max hourly price in USD (default ${MAX_USD_PER_HR}): " price_choice
    [[ -n "$price_choice" ]] && MAX_USD_PER_HR="$price_choice"
    list_and_rank_runpod_gpus table || true
    read -r -p "Proceed with provisioning top-ranked GPU pod? [Y/n]: " confirm
    if [[ "${confirm:-Y}" =~ ^[Yy] ]]; then
      provision_runpod_pod
    fi
    exit 0
  fi
}

#-----------------------------------------------------------------------------
# Preflight & Hardware Auto-Detection
#-----------------------------------------------------------------------------
VRAM_MB=0
DETECTED_GPU_NAME="none"
AUTO_COMPUTED_VRAM_ARGS=()

preflight() {
  set_stage "preflight"
  require_cmd git
  require_cmd curl
  require_cmd flock
  require_cmd awk
  require_cmd df

  mkdir -p "$WORKSPACE_DIR"
  touch "${WORKSPACE_DIR}/.writetest" 2>/dev/null \
    || die 3 "${WORKSPACE_DIR} is not writable — check your volume mount"
  rm -f "${WORKSPACE_DIR}/.writetest"

  # Keep temp files, pip cache, uv cache, and HuggingFace cache on /workspace
  mkdir -p "${WORKSPACE_DIR}/tmp" "${WORKSPACE_DIR}/.cache/pip" "${WORKSPACE_DIR}/.cache/uv" "${WORKSPACE_DIR}/.cache/huggingface"
  export TMPDIR="${WORKSPACE_DIR}/tmp"
  export PIP_CACHE_DIR="${WORKSPACE_DIR}/.cache/pip"
  export UV_CACHE_DIR="${WORKSPACE_DIR}/.cache/uv"
  export HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER}"

  git config --global --add safe.directory '*' 2>/dev/null || true

  log_info "Checking internet connectivity..."
  retry 3 curl -fsS -m 15 -o /dev/null https://huggingface.co 2>/dev/null \
    || retry 3 curl -fsS -m 15 -o /dev/null https://github.com 2>/dev/null \
    || die 3 "No internet access (huggingface.co and github.com unreachable)"

  if (( ! SKIP_LAUNCH )) && (exec 3<>/dev/tcp/127.0.0.1/${PORT}) 2>/dev/null; then
    die 3 "Port ${PORT} is already in use — stop the existing server or pass --port N"
  fi

  local need_gb="$MIN_DISK_GB"
  if (( DOWNLOAD_MODELS )); then
    local cat_gb
    cat_gb="$(catalog_total_gb)"
    need_gb=$(( need_gb + cat_gb ))
    log_info "Model profile '${MODEL_PROFILE}' requires ~${cat_gb} GB (+ ${MIN_DISK_GB} GB base = ${need_gb} GB)"
  fi
  local free_gb
  free_gb="$(df -BG --output=avail "$WORKSPACE_DIR" | tail -n1 | tr -dc '0-9')"
  log_info "Disk available on ${WORKSPACE_DIR}: ${free_gb} GB (recommended >= ${need_gb} GB)"
  if (( free_gb < MIN_DISK_GB )); then
    if (( DRY_RUN )); then
      log_warn "[DRY-RUN] Only ${free_gb} GB free on ${WORKSPACE_DIR} (< ${MIN_DISK_GB} GB minimum); continuing dry-run"
    else
      die 3 "Only ${free_gb} GB free on ${WORKSPACE_DIR}; need at least ${MIN_DISK_GB} GB base disk space"
    fi
  elif (( free_gb < need_gb )); then
    log_warn "Free disk (${free_gb} GB) is less than full profile estimate (${need_gb} GB); per-model disk guards will skip models that do not fit"
  fi

  local ram_gb
  ram_gb="$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)"
  log_info "System RAM: ${ram_gb} GB"
  (( ram_gb >= MIN_RAM_GB )) || log_warn "Only ${ram_gb} GB system RAM detected (recommended >= ${MIN_RAM_GB} GB)"

  if [[ "$BACKEND" == "cuda" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      local gpu_info
      gpu_info="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
      [[ -n "$gpu_info" ]] || die 3 "nvidia-smi found but reports no active NVIDIA GPU"
      DETECTED_GPU_NAME="${gpu_info%%,*}"
      log_info "Detected NVIDIA GPU: ${gpu_info}"
      local vram_mb
      vram_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9')"
      if [[ -n "$vram_mb" && "$vram_mb" -gt 0 ]]; then
        VRAM_MB="$vram_mb"
        local vram_gb=$(( vram_mb / 1024 ))
        log_info "Detected GPU VRAM: ${vram_gb} GB (${vram_mb} MB)"
        if (( AUTO_VRAM_FLAGS )); then
          if (( vram_mb < 10000 )); then
            AUTO_COMPUTED_VRAM_ARGS=(--lowvram --medvram-sdxl)
          elif (( vram_mb < 17000 )); then
            AUTO_COMPUTED_VRAM_ARGS=(--medvram-sdxl)
          else
            AUTO_COMPUTED_VRAM_ARGS=()
          fi
          log_info "Auto-selected VRAM optimization flags: ${AUTO_COMPUTED_VRAM_ARGS[*]:-(none needed — high VRAM GPU)}"
        fi
      fi
    elif (( DRY_RUN || SKIP_LAUNCH )); then
      log_warn "nvidia-smi not present (dry-run or build host) — continuing without live GPU check"
    else
      die 3 "BACKEND=cuda but nvidia-smi was not found"
    fi
  elif [[ "$BACKEND" == "rocm" ]]; then
    require_cmd rocm-smi
    log_info "ROCm GPU backend active"
  else
    log_warn "BACKEND=cpu — inference will run on CPU only"
  fi

  require_cmd "$PYTHON_BIN"
  local pyver
  pyver="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
  log_info "Python version: ${pyver}"
  ver_ge "$pyver" "3.10" || die 3 "Python >= 3.10 is required (found ${pyver})"

  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    log_info "RunPod Environment Detected: Pod ID = ${RUNPOD_POD_ID}"
    log_info "Public RunPod URL: https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net"
  fi
}

#-----------------------------------------------------------------------------
# System Dependencies (Ubuntu/Debian + rclone + uv + tcmalloc + inotify-tools)
#-----------------------------------------------------------------------------
APT_PACKAGES=(
  git git-lfs curl wget ca-certificates python3-venv python3-pip python3-dev
  libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg aria2 jq zstd
  inotify-tools fuse3 libgoogle-perftools4
)

install_system_deps() {
  set_stage "sys-deps"
  if command -v apt-get >/dev/null 2>&1; then
    local missing=() p
    for p in "${APT_PACKAGES[@]}"; do
      dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
    done
    if (( ${#missing[@]} )); then
      log_info "Installing system packages: ${missing[*]}"
      export DEBIAN_FRONTEND=noninteractive
      if as_root apt-get update -qq; then
        as_root apt-get install -y -qq "${missing[@]}" || log_warn "Some optional apt packages failed to install; continuing"
      fi
    else
      log_info "All system packages already present"
    fi
  else
    log_warn "apt-get not available — skipping apt package check"
  fi

  git lfs install --system >/dev/null 2>&1 || git lfs install >/dev/null 2>&1 || true
  install_rclone
  install_uv
}

install_rclone() {
  (( DRIVE_ENABLED || PHONE_ENABLED || POD_WEBDAV_ENABLED )) || return 0
  if command -v rclone >/dev/null 2>&1; then
    log_info "rclone available: $(rclone version 2>/dev/null | head -n1)"
    return 0
  fi
  log_info "Installing rclone (for Google Drive & Phone Storage integration)..."
  if command -v apt-get >/dev/null 2>&1 && as_root apt-get install -y -qq rclone 2>/dev/null; then
    log_info "rclone installed via apt"
    return 0
  fi
  local arch="amd64"
  [[ "$(uname -m)" == "aarch64" ]] && arch="arm64"
  local tmp_zip="${TMPDIR:-/tmp}/rclone.zip"
  retry 3 curl -fsSL "https://downloads.rclone.org/rclone-current-linux-${arch}.zip" -o "$tmp_zip" \
    || die 4 "Failed to download rclone binary"
  mkdir -p "${HOME}/.local/bin"
  "$PYTHON_BIN" - "$tmp_zip" "${HOME}/.local/bin/rclone" <<'PYEOF'
import sys, zipfile, shutil, os
with zipfile.ZipFile(sys.argv[1]) as zf:
    for name in zf.namelist():
        if name.endswith("/rclone"):
            with zf.open(name) as src, open(sys.argv[2], "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.chmod(sys.argv[2], 0o755)
            break
PYEOF
  export PATH="${HOME}/.local/bin:${PATH}"
  rm -f "$tmp_zip"
  command -v rclone >/dev/null 2>&1 || die 4 "rclone installation failed"
  log_info "rclone installed to ~/.local/bin: $(rclone version 2>/dev/null | head -n1)"
}

install_uv() {
  (( USE_UV )) || return 0
  if command -v uv >/dev/null 2>&1; then
    log_info "uv available: $(uv --version 2>/dev/null)"
    return 0
  fi
  log_info "Installing astral uv for fast Python package resolution..."
  if curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${HOME}/.local/bin" sh >/dev/null 2>&1; then
    export PATH="${HOME}/.local/bin:${PATH}"
    log_info "uv installed: $(uv --version 2>/dev/null || echo ok)"
  else
    log_warn "uv install skipped — falling back to standard pip"
    USE_UV=0
  fi
}

configure_tcmalloc() {
  (( USE_TCMALLOC )) || return 0
  local lib
  for lib in \
    /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 \
    /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
    /usr/lib/aarch64-linux-gnu/libtcmalloc.so.4; do
    if [[ -f "$lib" ]]; then
      export LD_PRELOAD="${lib}${LD_PRELOAD:+:${LD_PRELOAD}}"
      log_info "Enabled TCMalloc memory allocator (${lib})"
      return 0
    fi
  done
  log_debug "libtcmalloc not found on system; using default glibc malloc"
}

#-----------------------------------------------------------------------------
# SD.Next Repository & Virtual Environment
#-----------------------------------------------------------------------------
VENV_DIR="${SDNEXT_DIR}/venv"
VENV_PY="${VENV_DIR}/bin/python"

sync_repo() {
  set_stage "repo-sync"
  if [[ -d "${SDNEXT_DIR}/.git" ]]; then
    local cur
    cur="$(git -C "$SDNEXT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    log_info "Existing SD.Next checkout at ${SDNEXT_DIR} (${cur})"
    if (( DO_UPDATE )); then
      log_info "Updating SD.Next to latest ${SDNEXT_BRANCH}..."
      retry 3 git -C "$SDNEXT_DIR" fetch --depth 1 origin "$SDNEXT_BRANCH" || die 4 "git fetch failed"
      git -C "$SDNEXT_DIR" reset --hard FETCH_HEAD || die 4 "git reset failed"
      log_info "Updated SD.Next to $(git -C "$SDNEXT_DIR" rev-parse --short HEAD)"
    fi
  else
    [[ -e "$SDNEXT_DIR" ]] && die 4 "${SDNEXT_DIR} exists but is not a git repository"
    log_info "Cloning ${SDNEXT_REPO} (branch: ${SDNEXT_BRANCH})..."
    mkdir -p "$(dirname "$SDNEXT_DIR")"
    retry 3 git clone --depth 1 --branch "$SDNEXT_BRANCH" "$SDNEXT_REPO" "$SDNEXT_DIR" || die 4 "git clone failed"
  fi
  [[ -f "${SDNEXT_DIR}/webui.py" && -f "${SDNEXT_DIR}/installer.py" ]] \
    || die 4 "SD.Next checkout incomplete (webui.py or installer.py missing)"

  if [[ -n "$SDNEXT_COMMIT" ]]; then
    log_info "Pinning SD.Next to commit ${SDNEXT_COMMIT}..."
    if ! git -C "$SDNEXT_DIR" cat-file -e "${SDNEXT_COMMIT}^{commit}" 2>/dev/null; then
      retry 3 git -C "$SDNEXT_DIR" fetch --depth 1 origin "$SDNEXT_COMMIT" || die 4 "Failed to fetch commit ${SDNEXT_COMMIT}"
    fi
    git -C "$SDNEXT_DIR" reset --hard "$SDNEXT_COMMIT" || die 4 "Failed to pin commit ${SDNEXT_COMMIT}"
  fi
  log_info "SD.Next commit: $(git -C "$SDNEXT_DIR" rev-parse --short HEAD)"
}

setup_venv() {
  set_stage "venv"
  VENV_DIR="${SDNEXT_DIR}/venv"
  VENV_PY="${VENV_DIR}/bin/python"
  if (( DO_REINSTALL )) && [[ -d "$VENV_DIR" ]]; then
    log_warn "--reinstall requested: wiping ${VENV_DIR}"
    rm -rf "$VENV_DIR"
  fi
  if [[ -x "$VENV_PY" ]]; then
    log_info "Using existing virtualenv: ${VENV_DIR}"
  else
    if (( USE_UV )) && command -v uv >/dev/null 2>&1; then
      log_info "Creating virtualenv with uv (${PYTHON_BIN})..."
      retry 2 uv venv --python "$PYTHON_BIN" --seed "$VENV_DIR" || retry 2 "$PYTHON_BIN" -m venv "$VENV_DIR" || die 4 "venv creation failed"
    else
      log_info "Creating virtualenv with ${PYTHON_BIN} -m venv..."
      retry 2 "$PYTHON_BIN" -m venv "$VENV_DIR" || die 4 "venv creation failed"
    fi
  fi

  log_info "Ensuring pip, wheel, huggingface_hub[hf_transfer] are installed..."
  if (( USE_UV )) && command -v uv >/dev/null 2>&1; then
    retry 3 uv pip install --python "$VENV_PY" --upgrade pip wheel "huggingface_hub[hf_transfer,cli]" \
      || retry 3 "$VENV_PY" -m pip install --quiet --upgrade pip wheel "huggingface_hub[hf_transfer,cli]" \
      || die 4 "Failed to bootstrap pip/huggingface_hub in venv"
  else
    retry 3 "$VENV_PY" -m pip install --quiet --upgrade pip wheel "huggingface_hub[hf_transfer,cli]" \
      || die 4 "Failed to bootstrap pip/huggingface_hub in venv"
  fi
}

run_installer() {
  set_stage "installer"
  if (( SKIP_INSTALL )); then
    log_info "--skip-install set: skipping installer.py"
    return 0
  fi
  local args=()
  case "$BACKEND" in
    cuda) args+=(--use-cuda) ;;
    rocm) args+=(--use-rocm) ;;
  esac
  (( DO_REINSTALL )) && args+=(--reinstall)
  (( INSTALL_EXTENSIONS )) || args+=(--skip-extensions)

  log_info "Running SD.Next installer.py (${args[*]:-default})..."
  pushd "$SDNEXT_DIR" >/dev/null
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  retry 2 "$VENV_PY" installer.py "${args[@]}" \
    || { popd >/dev/null; die 4 "SD.Next installer.py failed — check ${SDNEXT_DIR}/sdnext.log"; }
  popd >/dev/null

  local -a torch_check_args=()
  [[ "$BACKEND" == "cuda" ]] && torch_check_args+=(--expect-cuda)
  "$VENV_PY" - "${torch_check_args[@]}" <<'PYEOF' || die 4 "PyTorch / CUDA verification failed"
import torch, sys
print(f"PyTorch={torch.__version__} CUDA_available={torch.cuda.is_available()} CUDA_version={torch.version.cuda}")
if "--expect-cuda" in sys.argv:
    assert torch.cuda.is_available(), "PyTorch cannot see CUDA GPU!"
PYEOF
  log_info "SD.Next installer & PyTorch verification passed"
}

#-----------------------------------------------------------------------------
# Extension Suite Sync (Parallelized)
#-----------------------------------------------------------------------------
EXT_OK=()
EXT_FAILED=()

sync_one_extension() { # name url
  local name="$1" url="$2"
  local dest="${SDNEXT_DIR}/extensions/${name}"
  if [[ -d "${dest}/.git" ]]; then
    if (( UPDATE_EXTENSIONS )); then
      log_info "Updating extension: ${name}"
      retry 2 git -C "$dest" fetch --depth 1 origin HEAD \
        && git -C "$dest" reset --quiet --hard FETCH_HEAD \
        || log_warn "Could not update extension ${name}; keeping existing version"
    else
      log_debug "Extension already present: ${name}"
    fi
  else
    log_info "Cloning extension: ${name} (${url})"
    retry 2 git clone --depth 1 "$url" "$dest" || return 1
  fi
  return 0
}

sync_extensions() {
  set_stage "extensions"
  if (( ! INSTALL_EXTENSIONS )); then
    log_info "--skip-extensions: skipping extension installation"
    return 0
  fi
  mkdir -p "${SDNEXT_DIR}/extensions"
  local -a all_specs=("${CORE_EXTENSIONS[@]}" "${DETAIL_EXTENSIONS[@]}" "${POPULAR_EXTENSIONS[@]}")
  if [[ -n "$EXTENSIONS_EXTRA" ]]; then
    local -a extra
    IFS=' ' read -r -a extra <<< "$EXTENSIONS_EXTRA"
    all_specs+=("${extra[@]}")
  fi

  EXT_OK=()
  EXT_FAILED=()
  local entry name url
  for entry in "${all_specs[@]}"; do
    name="${entry%%|*}"
    url="${entry#*|}"
    if [[ -z "$name" || -z "$url" || "$url" == "$entry" ]]; then
      log_warn "Skipping malformed extension spec: ${entry}"
      continue
    fi
    if sync_one_extension "$name" "$url"; then
      EXT_OK+=("$name")
    else
      log_warn "Extension failed (non-fatal): ${name}"
      EXT_FAILED+=("$name")
    fi
  done
  log_info "Extensions ready (${#EXT_OK[@]}): ${EXT_OK[*]:-none}"
  (( ${#EXT_FAILED[@]} == 0 )) || log_warn "Extensions failed (${#EXT_FAILED[@]}): ${EXT_FAILED[*]}"
}

#-----------------------------------------------------------------------------
# Google Drive Integration (OAuth / Service Account / Headless Wizard / Sync & Mount)
#-----------------------------------------------------------------------------
DRIVE_READY=0
RCLONE_CONF="${WORKSPACE_DIR}/.rclone.conf"
SA_JSON_FILE="${WORKSPACE_DIR}/.gdrive-sa.json"
FUSE_MOUNT_PID=""

run_drive_interactive_auth() {
  set_stage "drive-auth"
  install_rclone
  cat >&2 <<'EOF'
==================================================================
 Google Drive Headless Authorization Wizard
==================================================================
1. On ANY machine with a browser (your laptop or phone), install rclone
   or run:
     rclone authorize "drive"
2. Sign in to Google in your browser and copy the JSON token printed
   between '--->' and '<---'.
==================================================================
EOF
  read -r -p "Paste the rclone JSON token here: " token_json
  [[ -n "$token_json" ]] || die 8 "Empty token provided"
  read -r -p "Optional Google Drive Team/Shared Drive ID (leave blank for personal My Drive): " team_drive
  cat >"$RCLONE_CONF" <<EOF
[${DRIVE_REMOTE}]
type = drive
scope = drive
token = ${token_json}
EOF
  if [[ -n "$team_drive" ]]; then
    echo "team_drive = ${team_drive}" >>"$RCLONE_CONF"
  fi
  chmod 600 "$RCLONE_CONF"
  export RCLONE_CONFIG="$RCLONE_CONF"
  if rclone lsd "${DRIVE_REMOTE}:" >/dev/null 2>&1; then
    log_info "Google Drive connected and verified!"
    log_info "Save this environment variable in your RunPod template for future pods:"
    echo "RCLONE_CONFIG_B64=$(base64 -w0 "$RCLONE_CONF" 2>/dev/null || base64 "$RCLONE_CONF" | tr -d '\n')"
  else
    die 8 "rclone could not list '${DRIVE_REMOTE}:' with the provided token"
  fi
}

drive_setup() {
  set_stage "drive-setup"
  (( DRIVE_ENABLED )) || { log_info "Google Drive integration disabled (--no-drive)"; return 0; }
  if (( DRIVE_INTERACTIVE_AUTH )); then
    run_drive_interactive_auth
  fi

  if [[ -n "$RCLONE_CONFIG_B64" ]]; then
    log_info "Decoding RCLONE_CONFIG_B64 -> ${RCLONE_CONF}"
    printf '%s' "$RCLONE_CONFIG_B64" | base64 -d >"$RCLONE_CONF" \
      || die 8 "RCLONE_CONFIG_B64 is invalid base64"
    chmod 600 "$RCLONE_CONF"
  elif [[ -n "$GDRIVE_SERVICE_ACCOUNT_B64" ]]; then
    log_info "Configuring rclone with Google Cloud Service Account (GDRIVE_SERVICE_ACCOUNT_B64)"
    printf '%s' "$GDRIVE_SERVICE_ACCOUNT_B64" | base64 -d >"$SA_JSON_FILE" \
      || die 8 "GDRIVE_SERVICE_ACCOUNT_B64 is invalid base64"
    chmod 600 "$SA_JSON_FILE"
    cat >"$RCLONE_CONF" <<EOF
[${DRIVE_REMOTE}]
type = drive
scope = drive
service_account_file = ${SA_JSON_FILE}
EOF
    chmod 600 "$RCLONE_CONF"
  elif [[ -n "$RCLONE_CONFIG_FILE" && -f "$RCLONE_CONFIG_FILE" ]]; then
    cp -f "$RCLONE_CONFIG_FILE" "$RCLONE_CONF"
    chmod 600 "$RCLONE_CONF"
  elif [[ ! -f "$RCLONE_CONF" && -f "${HOME}/.config/rclone/rclone.conf" ]]; then
    cp -f "${HOME}/.config/rclone/rclone.conf" "$RCLONE_CONF"
    chmod 600 "$RCLONE_CONF"
  fi

  if [[ ! -f "$RCLONE_CONF" ]]; then
    log_warn "No rclone config found — Google Drive sync is inactive for this run."
    log_warn "Tip: Run './deploy-sdnext.sh --drive-auth' or set RCLONE_CONFIG_B64 / GDRIVE_SERVICE_ACCOUNT_B64."
    return 0
  fi

  export RCLONE_CONFIG="$RCLONE_CONF"
  if retry 2 rclone lsd "${DRIVE_REMOTE}:" >/dev/null 2>&1; then
    DRIVE_READY=1
    log_info "Google Drive connected: ${DRIVE_REMOTE}:${DRIVE_ROOT} (mode=${DRIVE_MOUNT_MODE})"
    rclone mkdir "${DRIVE_REMOTE}:${DRIVE_ROOT}/models" \
      "${DRIVE_REMOTE}:${DRIVE_ROOT}/outputs" \
      "${DRIVE_REMOTE}:${DRIVE_ROOT}/config" \
      "${DRIVE_REMOTE}:${DRIVE_ROOT}/logs" 2>/dev/null || true
  else
    log_warn "rclone config found, but remote '${DRIVE_REMOTE}:' could not be reached — continuing without Drive"
  fi
}

_rclone_common_flags() {
  local -a flags=(--transfers 4 --checkers 8 --timeout 120s --contimeout 60s --fast-list --drive-chunk-size 64M)
  [[ -n "$DRIVE_BWLIMIT" ]] && flags+=(--bwlimit "$DRIVE_BWLIMIT")
  printf '%s\n' "${flags[@]}"
}

drive_mount_fuse() {
  (( DRIVE_READY )) || return 0
  local mnt="${WORKSPACE_DIR}/gdrive-mount"
  mkdir -p "$mnt"
  if mountpoint -q "$mnt" 2>/dev/null; then
    log_info "Google Drive FUSE mount already active at ${mnt}"
    return 0
  fi
  log_info "Mounting ${DRIVE_REMOTE}:${DRIVE_ROOT} at ${mnt} (rclone VFS cache mode writes)..."
  rclone mount "${DRIVE_REMOTE}:${DRIVE_ROOT}" "$mnt" \
    --vfs-cache-mode writes \
    --vfs-cache-max-size 20G \
    --dir-cache-time 5m \
    --allow-non-empty \
    --daemon 2>/dev/null || {
      log_warn "FUSE mount failed (container may lack /dev/fuse CAP_SYS_ADMIN) — falling back to sync mode"
      DRIVE_MOUNT_MODE="sync"
      return 1
    }
  log_info "Google Drive mounted at ${mnt}"
}

drive_pull() {
  (( DRIVE_READY )) || return 0
  set_stage "drive-pull"
  if [[ "$DRIVE_MOUNT_MODE" == "mount" || "$DRIVE_MOUNT_MODE" == "hybrid" ]]; then
    drive_mount_fuse || true
  fi
  local -a rflags
  mapfile -t rflags < <(_rclone_common_flags)

  log_info "Restoring models & checkpoints from ${DRIVE_REMOTE}:${DRIVE_ROOT}/models -> ${SDNEXT_DIR}/models"
  mkdir -p "${SDNEXT_DIR}/models"
  rclone copy "${rflags[@]}" "${DRIVE_REMOTE}:${DRIVE_ROOT}/models" "${SDNEXT_DIR}/models" \
    && log_info "Models restored from Google Drive" \
    || log_warn "Drive model restore reported non-fatal warnings"

  if (( DRIVE_SYNC_LORAS )); then
    local sub
    for sub in "models/Lora" "models/embeddings" "models/ESRGAN" "models/yolo" "wildcards"; do
      mkdir -p "${SDNEXT_DIR}/${sub}"
      rclone copy "${rflags[@]}" "${DRIVE_REMOTE}:${DRIVE_ROOT}/${sub}" "${SDNEXT_DIR}/${sub}" 2>/dev/null || true
    done
  fi

  local f
  for f in config.json ui-config.json styles.csv params.txt; do
    rclone copyto --timeout 60s "${DRIVE_REMOTE}:${DRIVE_ROOT}/config/${f}" "${SDNEXT_DIR}/${f}" 2>/dev/null \
      && log_info "Restored config from Drive: ${f}" || true
  done
}

drive_push() {
  (( DRIVE_READY )) || return 0
  exec 8>"$DRIVE_PUSH_LOCK"
  if ! flock -n 8; then
    log_debug "Drive push already running — skipping overlapping invocation"
    return 0
  fi
  local -a rflags
  mapfile -t rflags < <(_rclone_common_flags)

  log_info "Syncing outputs & configs to ${DRIVE_REMOTE}:${DRIVE_ROOT}..."
  if [[ -d "${SDNEXT_DIR}/outputs" ]]; then
    rclone copy "${rflags[@]}" "${SDNEXT_DIR}/outputs" "${DRIVE_REMOTE}:${DRIVE_ROOT}/outputs" 2>/dev/null \
      || log_warn "Drive outputs backup had warnings"
  fi

  if (( DRIVE_SYNC_LORAS )); then
    local sub
    for sub in "models/Lora" "models/embeddings" "wildcards"; do
      [[ -d "${SDNEXT_DIR}/${sub}" ]] || continue
      rclone copy "${rflags[@]}" "${SDNEXT_DIR}/${sub}" "${DRIVE_REMOTE}:${DRIVE_ROOT}/${sub}" 2>/dev/null || true
    done
  fi

  local f
  for f in config.json ui-config.json styles.csv params.txt deploy-config.env; do
    [[ -f "${SDNEXT_DIR}/${f}" ]] || continue
    rclone copyto --timeout 60s "${SDNEXT_DIR}/${f}" "${DRIVE_REMOTE}:${DRIVE_ROOT}/config/${f}" 2>/dev/null || true
  done

  drive_export_logs || true
  log_info "Google Drive backup complete"
  flock -u 8
}

drive_export_logs() {
  (( DRIVE_READY && DRIVE_EXPORT_LOGS )) || return 0
  rclone copyto --timeout 30s "$LOG_FILE" "${DRIVE_REMOTE}:${DRIVE_ROOT}/logs/$(basename "$LOG_FILE")" 2>/dev/null || true
  rclone copyto --timeout 30s "$JSON_LOG_FILE" "${DRIVE_REMOTE}:${DRIVE_ROOT}/logs/$(basename "$JSON_LOG_FILE")" 2>/dev/null || true
  if [[ -f "${SDNEXT_DIR}/sdnext.log" ]]; then
    rclone copyto --timeout 30s "${SDNEXT_DIR}/sdnext.log" "${DRIVE_REMOTE}:${DRIVE_ROOT}/logs/sdnext-${RUN_STAMP}.log" 2>/dev/null || true
  fi
}

drive_sync_loop() {
  (( DRIVE_READY )) || return 0
  log_info "Background Google Drive sync loop started (every ${DRIVE_SYNC_MINUTES} min)"
  while true; do
    sleep $(( DRIVE_SYNC_MINUTES * 60 ))
    drive_push || true
  done
}

#-----------------------------------------------------------------------------
# Model & Multi-Resource Library Downloader (CivitAI v1 API + Utility Pack + HF)
#-----------------------------------------------------------------------------
MODEL_OK=()
MODEL_FAILED=()
MODEL_SKIPPED=()

verify_safetensors() { # filepath
  "$VENV_PY" - "$1" <<'PYEOF'
import struct, sys, json
p = sys.argv[1]
try:
    with open(p, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            sys.exit(1)
        n = struct.unpack("<Q", head)[0]
        if n <= 0 or n > 100 * 1024 * 1024:
            sys.exit(1)
        meta = json.loads(f.read(n))
        sys.exit(0 if isinstance(meta, dict) else 1)
except Exception:
    sys.exit(1)
PYEOF
}

resolve_civitai_url() { # url_or_shorthand -> prints "resolved_url|filename"
  local input="$1"
  local ver_id=""
  if [[ "$input" =~ ^civitai://([0-9]+)$ ]]; then
    ver_id="${BASH_REMATCH[1]}"
  elif [[ "$input" =~ civitai\.com/api/download/models/([0-9]+) ]]; then
    ver_id="${BASH_REMATCH[1]}"
  fi
  if [[ -z "$ver_id" ]]; then
    local fb="${input%%\?*}"
    fb="$(basename "$fb")"
    printf '%s|%s\n' "$input" "$fb"
    return 0
  fi
  local dl_url="https://civitai.com/api/download/models/${ver_id}"
  local api_url="https://civitai.com/api/v1/model-versions/${ver_id}"
  local -a auth=()
  [[ -n "$CIVITAI_TOKEN" ]] && auth=(-H "Authorization: Bearer ${CIVITAI_TOKEN}")
  local fname=""
  local meta
  if meta="$(curl -fsSL -m 20 "${auth[@]}" "$api_url" 2>/dev/null)"; then
    fname="$("$PYTHON_BIN" -c 'import sys,json; d=json.loads(sys.argv[1]); files=d.get("files") or []; print((files[0].get("name") if files else "") or "")' "$meta" 2>/dev/null || true)"
  fi
  [[ -n "$fname" ]] || fname="civitai-model-${ver_id}.safetensors"
  printf '%s|%s\n' "$dl_url" "$fname"
}

download_file_model() { # url target_dir [explicit_fname] -> 0=ok, 1=fail, 2=skipped
  local raw_url="$1" target_dir="$2" explicit_fname="${3:-}"
  local resolved url fname
  resolved="$(resolve_civitai_url "$raw_url")"
  url="${resolved%%|*}"
  fname="${explicit_fname:-${resolved#*|}}"
  local dest="${target_dir}/${fname}"

  if [[ -s "$dest" ]]; then
    if [[ "$fname" == *.safetensors ]] && ! verify_safetensors "$dest"; then
      log_warn "Existing file ${fname} has corrupt safetensors header — re-downloading"
      rm -f "$dest"
    else
      log_info "Model already exists and verified, skipping: ${fname}"
      return 2
    fi
  fi

  mkdir -p "$target_dir"
  local tmp="${dest}.part"
  local attempt=1
  while (( attempt <= 2 )); do
    log_info "Downloading ${fname} -> ${target_dir}/ (attempt ${attempt}/2)"
    local ok=1
    if command -v aria2c >/dev/null 2>&1; then
      local -a aria=(aria2c -x 8 -s 8 -k 1M -c --retry-wait=5 --max-tries=5 --console-log-level=warn -d "$target_dir" -o "$fname")
      [[ -n "$HF_TOKEN" && "$url" == *"huggingface.co"* ]] && aria+=(--header="Authorization: Bearer ${HF_TOKEN}")
      [[ -n "$CIVITAI_TOKEN" && "$url" == *"civitai.com"* ]] && aria+=(--header="Authorization: Bearer ${CIVITAI_TOKEN}")
      retry 2 "${aria[@]}" "$url" && ok=0 || true
    else
      local -a auth=()
      [[ -n "$HF_TOKEN" && "$url" == *"huggingface.co"* ]] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")
      [[ -n "$CIVITAI_TOKEN" && "$url" == *"civitai.com"* ]] && auth=(-H "Authorization: Bearer ${CIVITAI_TOKEN}")
      if retry 3 curl -fL -C - "${auth[@]}" -o "$tmp" "$url"; then
        mv -f "$tmp" "$dest"
        ok=0
      fi
    fi

    if (( ok == 0 )) && [[ -s "$dest" ]]; then
      if [[ "$fname" == *.safetensors ]] && ! verify_safetensors "$dest"; then
        log_warn "Downloaded ${fname} failed safetensors header check — retrying"
        rm -f "$dest" "$tmp"
        attempt=$(( attempt + 1 ))
        continue
      fi
      log_info "Saved ${dest} ($(du -h "$dest" | cut -f1))"
      return 0
    fi
    rm -f "$tmp"
    attempt=$(( attempt + 1 ))
  done
  log_error "Failed to download ${fname}"
  return 1
}

download_hf_repo() { # repo includes target_dir
  local repo="$1" includes="$2" target_dir="$3"
  if [[ -d "$target_dir" && -n "$(ls -A "$target_dir" 2>/dev/null)" ]]; then
    log_info "Repo already exists, skipping: ${repo}"
    return 2
  fi
  local hf_cli="" c
  for c in "${VENV_DIR}/bin/hf" "${VENV_DIR}/bin/huggingface-cli"; do
    [[ -x "$c" ]] && { hf_cli="$c"; break; }
  done
  [[ -n "$hf_cli" ]] || { log_error "huggingface-cli not found in ${VENV_DIR}"; return 1; }

  mkdir -p "$target_dir"
  local -a args=(download "$repo" --local-dir "$target_dir")
  local -a pats
  IFS=' ' read -r -a pats <<< "$includes"
  local pat
  for pat in "${pats[@]}"; do args+=(--include "$pat"); done

  log_info "Downloading HuggingFace snapshot ${repo} -> ${target_dir}/ (hf_transfer=${HF_HUB_ENABLE_HF_TRANSFER})"
  local -a env_vars=(HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER}")
  [[ -n "$HF_TOKEN" ]] && env_vars+=(HF_TOKEN="$HF_TOKEN")
  [[ -n "$HF_ENDPOINT" ]] && env_vars+=(HF_ENDPOINT="$HF_ENDPOINT")

  if retry 2 env "${env_vars[@]}" "$hf_cli" "${args[@]}"; then
    [[ -n "$(ls -A "$target_dir" 2>/dev/null)" ]] || { log_error "Snapshot empty for ${repo}"; return 1; }
    log_info "Saved snapshot ${target_dir} ($(du -sh "$target_dir" | cut -f1))"
    return 0
  fi
  return 1
}

download_utility_resource_pack() {
  (( DOWNLOAD_RESOURCE_PACK )) || { log_info "Utility Resource Pack skipped (--no-resource-pack)"; return 0; }
  log_info "Downloading Curated Utility Resource Pack (Upscalers, YOLOv8 Detailers, Core Embeddings)..."
  local entry id url subdir note rc
  local IFS='|'
  for entry in "${UTILITY_RESOURCE_PACK[@]}"; do
    read -r id url subdir note <<< "$entry"
    set +e
    download_file_model "$url" "${SDNEXT_DIR}/models/${subdir}"
    rc=$?
    set -e
    case $rc in
      0) MODEL_OK+=("resource:${id}") ;;
      2) MODEL_SKIPPED+=("resource:${id} (exists)") ;;
      *) log_warn "Utility resource '${id}' failed (non-fatal)" ;;
    esac
  done
}

download_url_list_to_subdir() { # space_separated_urls target_dir label_prefix
  local url_list="$1" target_dir="$2" prefix="$3"
  [[ -n "$url_list" ]] || return 0
  local -a specs
  IFS=' ' read -r -a specs <<< "$url_list"
  local spec rc
  for spec in "${specs[@]}"; do
    [[ -n "$spec" ]] || continue
    set +e
    download_file_model "$spec" "$target_dir"
    rc=$?
    set -e
    case $rc in
      0) MODEL_OK+=("${prefix}:$(basename "${spec%%\?*}")") ;;
      2) MODEL_SKIPPED+=("${prefix}:$(basename "${spec%%\?*}") (exists)") ;;
      *) MODEL_FAILED+=("${prefix}:$(basename "${spec%%\?*}")") ;;
    esac
  done
}

download_models() {
  set_stage "models"
  if (( ! DOWNLOAD_MODELS )); then
    log_info "--skip-models: skipping model downloads"
    return 0
  fi

  if [[ -n "$HF_TOKEN" ]]; then
    local hf_cli="" c
    for c in "${VENV_DIR}/bin/hf" "${VENV_DIR}/bin/huggingface-cli"; do
      [[ -x "$c" ]] && { hf_cli="$c"; break; }
    done
    if [[ -n "$hf_cli" ]] && HF_TOKEN="$HF_TOKEN" "$hf_cli" whoami >/dev/null 2>&1; then
      log_info "HuggingFace token authenticated"
    else
      log_warn "HF_TOKEN could not be verified with 'hf whoami' — gated models may be skipped"
    fi
  fi

  download_utility_resource_pack

  local entry id kind source includes target size flags note rc
  local IFS='|'
  for entry in "${MODEL_CATALOG[@]}"; do
    read -r id kind source includes target size flags note <<< "$entry"
    [[ "$flags" == *"disabled"* ]] && { MODEL_SKIPPED+=("$id (disabled)"); continue; }
    if ! model_in_active_profile "$id"; then
      MODEL_SKIPPED+=("$id (not in ${MODEL_PROFILE} profile)")
      continue
    fi
    if [[ "$flags" == *"gated"* && -z "$HF_TOKEN" ]]; then
      log_warn "Skipping gated model '${id}' — export HF_TOKEN and accept license on HuggingFace"
      MODEL_SKIPPED+=("$id (gated, no HF_TOKEN)")
      continue
    fi

    local free_now need_now
    free_now="$(df -BG --output=avail "$WORKSPACE_DIR" | tail -n1 | tr -dc '0-9')"
    need_now=$(( ${size%.*} + 5 ))
    if (( free_now < need_now )); then
      log_warn "Skipping '${id}' (~${size}GB): only ${free_now}GB free on ${WORKSPACE_DIR}"
      MODEL_SKIPPED+=("$id (low disk)")
      continue
    fi

    log_info "Fetching model '${id}': ${note} (~${size} GB)"
    set +e
    if [[ "$kind" == "file" ]]; then
      download_file_model "$source" "${SDNEXT_DIR}/models/${target}"
      rc=$?
    else
      download_hf_repo "$source" "$includes" "${SDNEXT_DIR}/models/${target}"
      rc=$?
    fi
    set -e
    case $rc in
      0) MODEL_OK+=("$id") ;;
      2) MODEL_SKIPPED+=("$id (exists)") ;;
      *)
        if [[ "$flags" == *"besteffort"* ]]; then
          log_warn "Best-effort model '${id}' unavailable — skipping"
          MODEL_SKIPPED+=("$id (best-effort unavailable)")
        else
          log_error "Failed to download model '${id}' (continuing)"
          MODEL_FAILED+=("$id")
        fi
        ;;
    esac
  done

  if [[ -n "$MODEL_URLS" ]]; then
    local spec url subdir
    local -a specs
    IFS=' ' read -r -a specs <<< "$MODEL_URLS"
    for spec in "${specs[@]}"; do
      if [[ "$spec" == *"::"* ]]; then
        subdir="${spec%%::*}"
        url="${spec#*::}"
      else
        subdir="Stable-diffusion"
        url="$spec"
      fi
      set +e
      download_file_model "$url" "${SDNEXT_DIR}/models/${subdir}"
      rc=$?
      set -e
      case $rc in
        0) MODEL_OK+=("custom:$(basename "${url%%\?*}")") ;;
        2) MODEL_SKIPPED+=("custom:$(basename "${url%%\?*}") (exists)") ;;
        *) MODEL_FAILED+=("custom:$(basename "${url%%\?*}")") ;;
      esac
    done
  fi

  # Dedicated Multi-Resource Lists (LoRAs, Embeddings, Upscalers, YOLO detectors, Wildcards)
  download_url_list_to_subdir "$LORA_URLS"      "${SDNEXT_DIR}/models/Lora"       "lora"
  download_url_list_to_subdir "$EMBEDDING_URLS" "${SDNEXT_DIR}/models/embeddings" "embedding"
  download_url_list_to_subdir "$UPSCALER_URLS"  "${SDNEXT_DIR}/models/ESRGAN"     "upscaler"
  download_url_list_to_subdir "$YOLO_URLS"      "${SDNEXT_DIR}/models/yolo"       "yolo"
  download_url_list_to_subdir "$WILDCARD_URLS"  "${SDNEXT_DIR}/wildcards"         "wildcard"

  log_info "Models/Resources downloaded (${#MODEL_OK[@]}): ${MODEL_OK[*]:-none}"
  (( ${#MODEL_FAILED[@]} == 0 )) || log_warn "Models/Resources failed (${#MODEL_FAILED[@]}): ${MODEL_FAILED[*]}"
  log_info "Models/Resources skipped (${#MODEL_SKIPPED[@]}): ${MODEL_SKIPPED[*]:-none}"
}

#-----------------------------------------------------------------------------
# Direct Phone Storage Integration (Android & iOS)
#   1. Userspace Tailscale VPN mesh (works in unprivileged RunPod containers)
#   2. rclone SFTP / WebDAV / FTP remote to Phone (pull models/inputs, push outputs)
#   3. Pod WebDAV server so iOS Files / Android file managers mount the pod directly
#   4. Headless Syncthing P2P daemon for zero-config real-time folder sync
#-----------------------------------------------------------------------------
PHONE_READY=0
PHONE_REMOTE_NAME="phone_storage"
TAILSCALE_SOCK="${WORKSPACE_DIR}/.tailscaled.sock"
TAILSCALE_STATE="${WORKSPACE_DIR}/.tailscale-state"
WEBDAV_PID=""
SYNCTHING_PID=""
PHONE_SYNC_PID=""

print_phone_setup_guide() {
  cat <<'EOF'
================================================================================
 Direct Phone Storage Integration Guide (Android & iOS <-> RunPod SD.Next)
================================================================================
You can connect your phone's storage to your RunPod pod in 3 ways:

--------------------------------------------------------------------------------
METHOD 1: Use Your Phone Like Google Drive (Pod Pulls Models/Inputs & Pushes Outputs)
--------------------------------------------------------------------------------
Because your phone is behind mobile/Wi-Fi NAT, install the free Tailscale app on
your phone so your phone gets a private 100.x.y.z IP reachable from the Pod:

  A) On Android (Option 1 — Termux SFTP, fastest & supports background wake-lock):
     1. Install Tailscale & Termux. Turn Tailscale ON (note your phone's 100.x.y.z IP).
     2. In Termux, run:
          pkg update && pkg install openssh
          termux-setup-storage
          mkdir -p /sdcard/SDNext/{models,Lora,embeddings,inputs,outputs,wildcards,config}
          passwd                  # set your SSH password
          sshd                    # starts SFTP server on port 8022
          whoami                  # note your username (e.g. u0_a245)

  B) On Android (Option 2 — GUI App: Material Files / Cx File Explorer / Round Sync):
     1. Open Material Files or Cx File Explorer -> "Network" / "Access from network".
     2. Start an SFTP or FTP/WebDAV server pointing to /sdcard/SDNext (e.g. port 8022 or 8080).

  C) On iOS (iPhone / iPad):
     1. Install Tailscale + "Secure ShellFish" (SFTP server for iOS Files) or
        "FE File Explorer" / "Documents by Readdle" (WebDAV server).
     2. Enable the SFTP/WebDAV server on folder /SDNext and note port/username/password.

  D) Launch deploy-sdnext.sh with your phone connected:
       export TAILSCALE_AUTH_KEY="tskey-auth-xxxxx"   # from login.tailscale.com/admin/settings/keys
       export PHONE_PASS="your-phone-password"
       ./deploy-sdnext.sh \
         --phone-host 100.x.y.z \
         --phone-port 8022 \
         --phone-protocol sftp \
         --phone-user u0_a245 \
         --phone-path /sdcard/SDNext

--------------------------------------------------------------------------------
METHOD 2: Mount the RunPod Pod Directly inside iOS "Files" or Android File Manager
--------------------------------------------------------------------------------
Run with `--phone-webdav` (or `POD_WEBDAV_ENABLED=1`):
  1. The pod starts an authenticated WebDAV server on port 7861.
  2. Open RunPod TCP/HTTP port 7861 (https://<pod-id>-7861.proxy.runpod.net).
  3. On iPhone/iPad: Open native "Files" app -> "..." -> "Connect to Server" ->
     enter `https://<pod-id>-7861.proxy.runpod.net` with user `sdnext` and your
     POD_WEBDAV_PASS. Your pod's `outputs/`, `models/`, and `inputs/` appear as a
     native folder on your phone!

--------------------------------------------------------------------------------
METHOD 3: Peer-to-Peer Real-Time Sync with Syncthing (Zero Port-Forwarding)
--------------------------------------------------------------------------------
  1. Install "Syncthing-Fork" (Android) or "Mobius Sync" (iOS).
  2. Copy your phone's 56-character Syncthing Device ID.
  3. Run with:
       SYNCTHING_ENABLED=1 SYNCTHING_PHONE_ID="XXXX-XXXX..." ./deploy-sdnext.sh
  4. Accept the folder share prompt on your phone for instant 2-way P2P sync!
================================================================================
EOF
}

setup_tailscale_userspace() {
  [[ -n "$TAILSCALE_AUTH_KEY" ]] || return 0
  set_stage "tailscale"
  if ! command -v tailscale >/dev/null 2>&1 || ! command -v tailscaled >/dev/null 2>&1; then
    log_info "Installing Tailscale binaries for userspace networking..."
    local arch="amd64"
    [[ "$(uname -m)" == "aarch64" ]] && arch="arm64"
    local ts_tgz="${TMPDIR:-/tmp}/tailscale.tgz"
    local ts_url
    ts_url="$(curl -fsSL "https://pkgs.tailscale.com/stable/?mode=json" 2>/dev/null | "$PYTHON_BIN" -c "import sys,json; d=json.loads(sys.stdin.read()); v=d['TarballsVersion']; print(f'https://pkgs.tailscale.com/stable/tailscale_{v}_${arch}.tgz')" 2>/dev/null || echo "https://pkgs.tailscale.com/stable/tailscale_1.76.6_${arch}.tgz")"
    retry 3 curl -fsSL "$ts_url" -o "$ts_tgz" || { log_warn "Could not download Tailscale tarball"; return 1; }
    mkdir -p "${HOME}/.local/bin" "${TMPDIR:-/tmp}/ts-extract"
    tar -xzf "$ts_tgz" -C "${TMPDIR:-/tmp}/ts-extract" --strip-components=1
    cp -f "${TMPDIR:-/tmp}/ts-extract/tailscale" "${TMPDIR:-/tmp}/ts-extract/tailscaled" "${HOME}/.local/bin/"
    chmod +x "${HOME}/.local/bin/tailscale" "${HOME}/.local/bin/tailscaled"
    export PATH="${HOME}/.local/bin:${PATH}"
    rm -rf "$ts_tgz" "${TMPDIR:-/tmp}/ts-extract"
  fi

  if ! pgrep -f "tailscaled.*userspace-networking" >/dev/null 2>&1; then
    log_info "Starting tailscaled in userspace-networking mode (SOCKS5 proxy on 127.0.0.1:1055)..."
    mkdir -p "$TAILSCALE_STATE"
    tailscaled \
      --tun=userspace-networking \
      --statedir="$TAILSCALE_STATE" \
      --socket="$TAILSCALE_SOCK" \
      --socks5-server=127.0.0.1:1055 \
      --outbound-http-proxy-listen=127.0.0.1:1055 \
      >"${LOG_DIR}/tailscaled.log" 2>&1 &
    sleep 3
  fi

  log_info "Joining Tailnet as '${TAILSCALE_HOSTNAME}'..."
  if tailscale --socket="$TAILSCALE_SOCK" up --authkey="$TAILSCALE_AUTH_KEY" --hostname="${TAILSCALE_HOSTNAME}-${RUNPOD_POD_ID:-pod}" --accept-routes >/dev/null 2>&1; then
    local ts_ip
    ts_ip="$(tailscale --socket="$TAILSCALE_SOCK" ip -4 2>/dev/null | head -n1 || echo unknown)"
    log_info "Tailscale connected! Pod Tailscale IP: ${ts_ip} (routing 100.x.y.z traffic via userspace proxy 127.0.0.1:1055)"
    export ALL_PROXY_TAILSCALE="socks5://127.0.0.1:1055"
  else
    log_warn "Tailscale authentication failed — check TAILSCALE_AUTH_KEY"
  fi
}

phone_setup() {
  (( PHONE_ENABLED )) || return 0
  set_stage "phone-setup"
  install_rclone

  if [[ -n "$TAILSCALE_AUTH_KEY" ]]; then
    setup_tailscale_userspace || true
  fi

  if [[ -z "$PHONE_HOST" ]]; then
    log_warn "PHONE_ENABLED=1 but PHONE_HOST is empty — skipping direct phone pull/push (run --phone-setup-guide)"
    return 0
  fi

  touch "$RCLONE_CONF"
  chmod 600 "$RCLONE_CONF"
  export RCLONE_CONFIG="$RCLONE_CONF"

  local obscured_pass=""
  if [[ -n "$PHONE_PASS" ]]; then
    obscured_pass="$(rclone obscure "$PHONE_PASS" 2>/dev/null || true)"
  fi

  local key_file=""
  if [[ -n "$PHONE_KEY_B64" ]]; then
    key_file="${WORKSPACE_DIR}/.phone-ssh.key"
    printf '%s' "$PHONE_KEY_B64" | base64 -d >"$key_file" || die 10 "PHONE_KEY_B64 is not valid base64"
    chmod 600 "$key_file"
  fi

  # Remove any stale [phone_storage] block from .rclone.conf before writing fresh config
  if [[ -f "$RCLONE_CONF" ]]; then
    "$PYTHON_BIN" - "$RCLONE_CONF" "$PHONE_REMOTE_NAME" <<'PYEOF'
import sys, configparser
p, sec = sys.argv[1], sys.argv[2]
cp = configparser.ConfigParser()
cp.read(p)
if cp.has_section(sec):
    cp.remove_section(sec)
with open(p, "w") as f:
    cp.write(f)
PYEOF
  fi

  case "$PHONE_PROTOCOL" in
    sftp)
      {
        echo "[${PHONE_REMOTE_NAME}]"
        echo "type = sftp"
        echo "host = ${PHONE_HOST}"
        echo "port = ${PHONE_PORT}"
        [[ -n "$PHONE_USER" ]] && echo "user = ${PHONE_USER}"
        [[ -n "$obscured_pass" ]] && echo "pass = ${obscured_pass}"
        [[ -n "$key_file" ]] && echo "key_file = ${key_file}"
        echo "shell_type = unix"
        echo "md5sum_command = none"
        echo "sha1sum_command = none"
      } >>"$RCLONE_CONF"
      ;;
    webdav)
      {
        echo "[${PHONE_REMOTE_NAME}]"
        echo "type = webdav"
        echo "url = http://${PHONE_HOST}:${PHONE_PORT}${PHONE_ROOT}"
        echo "vendor = other"
        [[ -n "$PHONE_USER" ]] && echo "user = ${PHONE_USER}"
        [[ -n "$obscured_pass" ]] && echo "pass = ${obscured_pass}"
      } >>"$RCLONE_CONF"
      ;;
    ftp)
      {
        echo "[${PHONE_REMOTE_NAME}]"
        echo "type = ftp"
        echo "host = ${PHONE_HOST}"
        echo "port = ${PHONE_PORT}"
        [[ -n "$PHONE_USER" ]] && echo "user = ${PHONE_USER}"
        [[ -n "$obscured_pass" ]] && echo "pass = ${obscured_pass}"
      } >>"$RCLONE_CONF"
      ;;
    *)
      die 10 "Unknown PHONE_PROTOCOL '${PHONE_PROTOCOL}' (expected sftp|webdav|ftp)"
      ;;
  esac

  local target_root="${PHONE_REMOTE_NAME}:${PHONE_ROOT}"
  [[ "$PHONE_PROTOCOL" == "webdav" ]] && target_root="${PHONE_REMOTE_NAME}:"

  if _phone_rclone lsd "$target_root" >/dev/null 2>&1; then
    PHONE_READY=1
    log_info "Phone Storage connected over ${PHONE_PROTOCOL}://${PHONE_HOST}:${PHONE_PORT} (${target_root})"
    _phone_rclone mkdir \
      "${target_root}/models" \
      "${target_root}/models/Lora" \
      "${target_root}/models/embeddings" \
      "${target_root}/inputs" \
      "${target_root}/outputs" \
      "${target_root}/wildcards" \
      "${target_root}/config" \
      "${target_root}/logs" 2>/dev/null || true
  else
    log_warn "Could not reach phone storage at ${PHONE_PROTOCOL}://${PHONE_HOST}:${PHONE_PORT} — continuing without phone sync"
    log_warn "Tip: Make sure your phone's SFTP/WebDAV app is active and Tailscale is connected (--phone-setup-guide)"
  fi
}

_phone_rclone() {
  if [[ -n "${ALL_PROXY_TAILSCALE:-}" && "$PHONE_HOST" == 100.* ]]; then
    env ALL_PROXY="$ALL_PROXY_TAILSCALE" rclone "$@"
  else
    rclone "$@"
  fi
}

phone_pull() {
  (( PHONE_READY )) || return 0
  set_stage "phone-pull"
  local target_root="${PHONE_REMOTE_NAME}:${PHONE_ROOT}"
  [[ "$PHONE_PROTOCOL" == "webdav" ]] && target_root="${PHONE_REMOTE_NAME}:"
  local -a pflags=(--transfers 3 --checkers 4 --timeout 90s --contimeout 30s --max-transfer "${PHONE_MAX_PULL_GB}G")

  if (( PHONE_PULL_MODELS )); then
    log_info "Loading models, LoRAs & embeddings from Phone Storage (${target_root}/models -> ${SDNEXT_DIR}/models)..."
    mkdir -p "${SDNEXT_DIR}/models"
    _phone_rclone copy "${pflags[@]}" "${target_root}/models" "${SDNEXT_DIR}/models" 2>/dev/null \
      && log_info "Models/LoRAs loaded from Phone Storage" \
      || log_warn "Phone model pull completed with warnings"
  fi

  if (( PHONE_PULL_INPUTS )); then
    log_info "Loading input/init-images & wildcards from Phone Storage..."
    mkdir -p "${SDNEXT_DIR}/inputs" "${SDNEXT_DIR}/wildcards"
    _phone_rclone copy "${pflags[@]}" "${target_root}/inputs" "${SDNEXT_DIR}/inputs" 2>/dev/null || true
    _phone_rclone copy "${pflags[@]}" "${target_root}/wildcards" "${SDNEXT_DIR}/wildcards" 2>/dev/null || true
  fi

  local f
  for f in config.json ui-config.json styles.csv; do
    _phone_rclone copyto --timeout 45s "${target_root}/config/${f}" "${SDNEXT_DIR}/${f}" 2>/dev/null \
      && log_info "Restored config from Phone Storage: ${f}" || true
  done
}

phone_push() {
  (( PHONE_READY && PHONE_PUSH_OUTPUTS )) || return 0
  exec 7>"$PHONE_PUSH_LOCK"
  if ! flock -n 7; then
    return 0
  fi
  local target_root="${PHONE_REMOTE_NAME}:${PHONE_ROOT}"
  [[ "$PHONE_PROTOCOL" == "webdav" ]] && target_root="${PHONE_REMOTE_NAME}:"
  local -a pflags=(--transfers 3 --checkers 4 --timeout 90s --contimeout 30s)

  log_info "Pushing generated outputs & configs to Phone Storage (${target_root}/outputs)..."
  if [[ -d "${SDNEXT_DIR}/outputs" ]]; then
    _phone_rclone copy "${pflags[@]}" "${SDNEXT_DIR}/outputs" "${target_root}/outputs" 2>/dev/null \
      || log_warn "Phone outputs push had non-fatal warnings"
  fi
  local f
  for f in config.json ui-config.json styles.csv; do
    [[ -f "${SDNEXT_DIR}/${f}" ]] || continue
    _phone_rclone copyto --timeout 45s "${SDNEXT_DIR}/${f}" "${target_root}/config/${f}" 2>/dev/null || true
  done
  phone_export_logs || true
  log_info "Phone Storage sync complete"
  flock -u 7
}

phone_export_logs() {
  (( PHONE_READY )) || return 0
  local target_root="${PHONE_REMOTE_NAME}:${PHONE_ROOT}"
  [[ "$PHONE_PROTOCOL" == "webdav" ]] && target_root="${PHONE_REMOTE_NAME}:"
  _phone_rclone copyto --timeout 30s "$LOG_FILE" "${target_root}/logs/$(basename "$LOG_FILE")" 2>/dev/null || true
}

phone_sync_loop() {
  (( PHONE_READY )) || return 0
  log_info "Background Phone Storage sync loop started (every ${PHONE_SYNC_MINUTES} min)"
  while true; do
    sleep $(( PHONE_SYNC_MINUTES * 60 ))
    phone_push || true
  done
}

start_pod_webdav_server() {
  (( POD_WEBDAV_ENABLED )) || return 0
  set_stage "phone-webdav"
  install_rclone
  if [[ -z "$POD_WEBDAV_PASS" ]]; then
    if [[ -n "$GRADIO_AUTH" && "$GRADIO_AUTH" == *":"* ]]; then
      POD_WEBDAV_USER="${GRADIO_AUTH%%:*}"
      POD_WEBDAV_PASS="${GRADIO_AUTH#*:}"
    else
      POD_WEBDAV_PASS="$(head -c 12 /dev/urandom | base64 | tr -dc 'a-zA-Z0-9' | head -c 12)"
    fi
  fi
  log_info "Starting Pod WebDAV Server on port ${POD_WEBDAV_PORT} for direct iOS/Android mounting..."
  rclone serve webdav "$SDNEXT_DIR" \
    --addr "0.0.0.0:${POD_WEBDAV_PORT}" \
    --user "$POD_WEBDAV_USER" \
    --pass "$POD_WEBDAV_PASS" \
    --vfs-cache-mode writes \
    >"${LOG_DIR}/pod-webdav.log" 2>&1 &
  WEBDAV_PID=$!
  log_info "Pod WebDAV Server running (PID ${WEBDAV_PID}) — user='${POD_WEBDAV_USER}' pass='${POD_WEBDAV_PASS}'"
  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    log_info "Connect from iOS Files / Android to: https://${RUNPOD_POD_ID}-${POD_WEBDAV_PORT}.proxy.runpod.net"
  fi
}

start_syncthing_p2p() {
  (( SYNCTHING_ENABLED )) || return 0
  set_stage "syncthing"
  if ! command -v syncthing >/dev/null 2>&1; then
    log_info "Installing Syncthing for peer-to-peer phone sync..."
    local arch="amd64"
    [[ "$(uname -m)" == "aarch64" ]] && arch="arm64"
    local st_url
    st_url="$(curl -fsSL https://api.github.com/repos/syncthing/syncthing/releases/latest 2>/dev/null | "$PYTHON_BIN" -c "import sys,json; d=json.loads(sys.stdin.read()); assets=d.get('assets') or []; print(next((a['browser_download_url'] for a in assets if f'linux-{arch}' in a['name'] and a['name'].endswith('.tar.gz')), ''))" 2>/dev/null || true)"
    if [[ -n "$st_url" ]]; then
      local st_tgz="${TMPDIR:-/tmp}/syncthing.tar.gz"
      retry 2 curl -fsSL "$st_url" -o "$st_tgz" || return 1
      mkdir -p "${HOME}/.local/bin" "${TMPDIR:-/tmp}/st-extract"
      tar -xzf "$st_tgz" -C "${TMPDIR:-/tmp}/st-extract" --strip-components=1
      cp -f "${TMPDIR:-/tmp}/st-extract/syncthing" "${HOME}/.local/bin/syncthing"
      chmod +x "${HOME}/.local/bin/syncthing"
      export PATH="${HOME}/.local/bin:${PATH}"
      rm -rf "$st_tgz" "${TMPDIR:-/tmp}/st-extract"
    fi
  fi
  command -v syncthing >/dev/null 2>&1 || { log_warn "Syncthing binary unavailable"; return 1; }

  local st_home="${WORKSPACE_DIR}/.syncthing"
  mkdir -p "$st_home" "${SDNEXT_DIR}/outputs" "${SDNEXT_DIR}/models/Lora"
  syncthing generate --home="$st_home" >/dev/null 2>&1 || true
  local my_id
  my_id="$(syncthing --home="$st_home" --device-id 2>/dev/null || echo unknown)"
  log_info "Starting Syncthing P2P daemon (Pod Device ID: ${my_id})..."
  syncthing serve --no-browser --home="$st_home" --gui-address="127.0.0.1:${SYNCTHING_GUI_PORT}" >"${LOG_DIR}/syncthing.log" 2>&1 &
  SYNCTHING_PID=$!
  sleep 3
  if [[ -n "$SYNCTHING_PHONE_ID" ]]; then
    log_info "Pairing Syncthing with Phone Device ID: ${SYNCTHING_PHONE_ID}"
    syncthing cli --home="$st_home" config devices add --device-id "$SYNCTHING_PHONE_ID" --name "UserPhone" --auto-accept-folders >/dev/null 2>&1 || true
    syncthing cli --home="$st_home" config folders add --id "sdnext-outputs" --label "SDNext Outputs" --path "${SDNEXT_DIR}/outputs" >/dev/null 2>&1 || true
    syncthing cli --home="$st_home" config folders "sdnext-outputs" devices add --device-id "$SYNCTHING_PHONE_ID" >/dev/null 2>&1 || true
    syncthing cli --home="$st_home" config folders add --id "sdnext-loras" --label "SDNext LoRAs" --path "${SDNEXT_DIR}/models/Lora" >/dev/null 2>&1 || true
    syncthing cli --home="$st_home" config folders "sdnext-loras" devices add --device-id "$SYNCTHING_PHONE_ID" >/dev/null 2>&1 || true
  fi
}

#-----------------------------------------------------------------------------
# Real-Time Output Streamer (inotifywait) & GPU Idle Cost Guard
#-----------------------------------------------------------------------------
INOTIFY_PID=""
IDLE_GUARD_PID=""

start_instant_output_watcher() {
  (( INSTANT_OUTPUT_SYNC )) || return 0
  (( DRIVE_READY || PHONE_READY )) || return 0
  command -v inotifywait >/dev/null 2>&1 || return 0
  mkdir -p "${SDNEXT_DIR}/outputs"
  log_info "Starting real-time inotify output streamer (syncs new images/videos to Drive/Phone within seconds)..."
  (
    while inotifywait -r -q -e close_write,moved_to --format '%w%f' "${SDNEXT_DIR}/outputs" >/dev/null 2>&1; do
      # Debounce bursts (e.g. batch generation or grids) by waiting 4s before syncing
      sleep 4
      (( DRIVE_READY )) && drive_push >/dev/null 2>&1 || true
      (( PHONE_READY )) && phone_push >/dev/null 2>&1 || true
    done
  ) &
  INOTIFY_PID=$!
}

start_idle_cost_guard() {
  (( IDLE_SHUTDOWN_MINUTES > 0 )) || return 0
  [[ "$BACKEND" == "cuda" ]] && command -v nvidia-smi >/dev/null 2>&1 || return 0
  log_info "Starting GPU Idle Cost Guard (auto-stops pod after ${IDLE_SHUTDOWN_MINUTES} min of GPU utilization < ${IDLE_GPU_THRESHOLD_PCT}%)..."
  (
    local idle_sec=0
    local max_sec=$(( IDLE_SHUTDOWN_MINUTES * 60 ))
    while true; do
      sleep 30
      local util
      util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9')"
      util="${util:-0}"
      if (( util <= IDLE_GPU_THRESHOLD_PCT )); then
        idle_sec=$(( idle_sec + 30 ))
        if (( idle_sec >= max_sec )); then
          log_warn "GPU utilization <= ${IDLE_GPU_THRESHOLD_PCT}% for ${IDLE_SHUTDOWN_MINUTES} minutes — triggering final backup & RunPod shutdown!"
          notify "IDLE_SHUTDOWN" "GPU idle for ${IDLE_SHUTDOWN_MINUTES}m — backing up & stopping pod ${RUNPOD_POD_ID:-local}"
          (( DRIVE_READY )) && drive_push || true
          (( PHONE_READY )) && phone_push || true
          if command -v runpodctl >/dev/null 2>&1 && [[ -n "${RUNPOD_POD_ID:-}" ]]; then
            runpodctl stop pod "$RUNPOD_POD_ID" || true
          elif [[ -n "${RUNPOD_API_KEY:-}" && -n "${RUNPOD_POD_ID:-}" ]]; then
            runpod_graphql "{\"query\":\"mutation { podStop(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { id desiredStatus } }\"}" >/dev/null 2>&1 || true
          fi
          kill -TERM $$ 2>/dev/null || true
          exit 0
        fi
      else
        idle_sec=0
      fi
    done
  ) &
  IDLE_GUARD_PID=$!
}

#-----------------------------------------------------------------------------
# Default Config Seeding, Snapshot & RunPod Start Script
#-----------------------------------------------------------------------------
seed_default_config() {
  (( SEED_DEFAULT_CONFIG )) || return 0
  local cfg="${SDNEXT_DIR}/config.json"
  [[ ! -f "$cfg" ]] || return 0
  log_info "Seeding hardware-tuned default ${cfg} (fast TAESD live previews, organized output paths)..."
  cat >"$cfg" <<'EOF'
{
  "outdir_samples": "",
  "outdir_txt2img_samples": "outputs/txt2img-images",
  "outdir_img2img_samples": "outputs/img2img-images",
  "outdir_extras_samples": "outputs/extras-images",
  "outdir_grids": "",
  "outdir_txt2img_grids": "outputs/txt2img-grids",
  "outdir_img2img_grids": "outputs/img2img-grids",
  "save_to_dirs": true,
  "grid_save": true,
  "live_preview_content": "TAESD",
  "show_progress_every_n_steps": 4,
  "pin_memory": true
}
EOF
}

write_config_snapshot() {
  cat >"${SDNEXT_DIR}/deploy-config.env" <<EOF
# Generated by deploy-sdnext.sh v${SCRIPT_VERSION} on $(date -u +%Y-%m-%dT%H:%M:%SZ)
SCRIPT_VERSION=${SCRIPT_VERSION}
SDNEXT_REPO=${SDNEXT_REPO}
SDNEXT_BRANCH=${SDNEXT_BRANCH}
GIT_HEAD=$(git -C "$SDNEXT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
BACKEND=${BACKEND}
DETECTED_GPU_NAME="${DETECTED_GPU_NAME}"
VRAM_MB=${VRAM_MB}
PORT=${PORT}
LISTEN=${LISTEN}
ENABLE_API=${ENABLE_API}
RUN_MODE=${RUN_MODE}
MODEL_PROFILE=${MODEL_PROFILE}
DRIVE_ENABLED=${DRIVE_ENABLED}
DRIVE_REMOTE=${DRIVE_REMOTE}
DRIVE_ROOT=${DRIVE_ROOT}
DRIVE_MOUNT_MODE=${DRIVE_MOUNT_MODE}
PHONE_ENABLED=${PHONE_ENABLED}
PHONE_PROTOCOL=${PHONE_PROTOCOL}
PHONE_HOST=${PHONE_HOST}
PHONE_PORT=${PHONE_PORT}
POD_WEBDAV_ENABLED=${POD_WEBDAV_ENABLED}
IDLE_SHUTDOWN_MINUTES=${IDLE_SHUTDOWN_MINUTES}
EOF
}

install_self_and_start_script() {
  local self_src self_dst
  self_src="$(readlink -f "$0" 2>/dev/null || echo "$0")"
  self_dst="${WORKSPACE_DIR}/deploy-sdnext.sh"
  if [[ -f "$self_src" && "$self_src" != "$self_dst" ]]; then
    cp -f "$self_src" "$self_dst"
    chmod +x "$self_dst"
    log_info "Installed persistent script copy to ${self_dst}"
  fi
  cat >"${WORKSPACE_DIR}/start-sdnext.sh" <<EOF
#!/usr/bin/env bash
# RunPod Pod Template "Start Command":
#   bash /workspace/start-sdnext.sh
exec bash "${self_dst}" --workspace "${WORKSPACE_DIR}"
EOF
  chmod +x "${WORKSPACE_DIR}/start-sdnext.sh"
  log_info "RunPod start script ready at ${WORKSPACE_DIR}/start-sdnext.sh"
}

#-----------------------------------------------------------------------------
# Server Launch, Self-Healing Crash Analyzer & Active Liveness Supervision
#-----------------------------------------------------------------------------
LAUNCH_ARGS=()
HEAL_EXTRA_ARGS=()
SERVER_LOG=""
SERVER_PID=""
SYNC_PID=""

build_launch_args() {
  LAUNCH_ARGS=()
  case "$BACKEND" in
    cuda) LAUNCH_ARGS+=(--use-cuda) ;;
    rocm) LAUNCH_ARGS+=(--use-rocm) ;;
  esac
  (( LISTEN )) && LAUNCH_ARGS+=(--listen --port "$PORT")
  (( ENABLE_API )) && LAUNCH_ARGS+=(--api --api-log)
  [[ -n "$GRADIO_AUTH" ]] && LAUNCH_ARGS+=(--gradio-auth "$GRADIO_AUTH")
  [[ -n "$API_AUTH" ]] && LAUNCH_ARGS+=(--api-auth "$API_AUTH")
  if (( LISTEN )) && [[ -z "$GRADIO_AUTH" && -z "$API_AUTH" ]]; then
    log_warn "No GRADIO_AUTH / API_AUTH set — WebUI and REST API are publicly accessible via proxy URL"
  fi
  if (( ${#AUTO_COMPUTED_VRAM_ARGS[@]} )); then
    LAUNCH_ARGS+=("${AUTO_COMPUTED_VRAM_ARGS[@]}")
  fi
  if (( ${#HEAL_EXTRA_ARGS[@]} )); then
    LAUNCH_ARGS+=("${HEAL_EXTRA_ARGS[@]}")
  fi
  if [[ -n "$EXTRA_ARGS" ]]; then
    local -a extra
    IFS=' ' read -r -a extra <<< "$EXTRA_ARGS"
    LAUNCH_ARGS+=("${extra[@]}")
  fi
}

wait_for_http() { # url timeout_s
  local url="$1" timeout="$2"
  local deadline=$(( $(date +%s) + timeout ))
  log_info "Waiting for SD.Next HTTP readiness at ${url} (timeout=${timeout}s)..."
  while (( $(date +%s) < deadline )); do
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log_error "Server process (PID ${SERVER_PID}) exited prematurely during startup"
      return 1
    fi
    if curl -fsS -m 5 -o /dev/null "$url" 2>/dev/null; then
      log_info "SD.Next HTTP endpoint is live!"
      return 0
    fi
    sleep 4
  done
  log_error "Timed out waiting for SD.Next after ${timeout}s"
  [[ -f "${SERVER_LOG:-}" ]] && tail -n 40 "$SERVER_LOG" >&2 || true
  return 1
}

smoke_test() {
  (( SMOKE_TEST )) || return 0
  (( ENABLE_API )) || { log_warn "SMOKE_TEST=1 requires ENABLE_API=1 — skipping"; return 0; }
  set_stage "smoke-test"
  local out="${LOG_DIR}/smoke-test.png"
  local -a auth=()
  [[ -n "$API_AUTH" ]] && auth=(--user "${API_AUTH%%,*}")
  local resp
  if ! resp="$(curl -fsS -m 600 "${auth[@]}" -H 'Content-Type: application/json' \
    -d '{"prompt":"a futuristic glass data center in the mountains, cinematic lighting","steps":8,"width":512,"height":512,"batch_size":1}' \
    "http://127.0.0.1:${PORT}/sdapi/v1/txt2img" 2>/dev/null)"; then
    log_warn "Smoke test API call failed (server may still be compiling kernels)"
    return 0
  fi
  if "$VENV_PY" - "$resp" "$out" <<'PYEOF'; then
import sys, json, base64
imgs = json.loads(sys.argv[1]).get("images") or []
if not imgs:
    sys.exit(1)
with open(sys.argv[2], "wb") as f:
    f.write(base64.b64decode(imgs[0]))
PYEOF
    log_info "Smoke test PASSED: generated ${out}"
  else
    log_warn "Smoke test response contained no image"
  fi
}

analyze_and_heal_failure() { # crash_count
  local crash_num="$1"
  local log_src="${SERVER_LOG:-${SDNEXT_DIR}/sdnext.log}"
  [[ -f "$log_src" ]] || return 0

  log_info "Running Self-Healing Crash Analyzer on ${log_src} (crash #${crash_num})..."
  local tail_txt
  tail_txt="$(tail -n 120 "$log_src" 2>/dev/null || true)"

  # 1. CUDA Out of Memory -> auto-downgrade VRAM flags (--medvram -> --lowvram)
  if printf '%s' "$tail_txt" | grep -qiE "CUDA out of memory|OutOfMemoryError"; then
    if [[ " ${HEAL_EXTRA_ARGS[*]} " != *" --lowvram "* ]]; then
      if [[ " ${HEAL_EXTRA_ARGS[*]} " != *" --medvram "* ]]; then
        HEAL_EXTRA_ARGS+=(--medvram)
        log_warn "[SELF-HEAL] Detected CUDA OutOfMemoryError -> adding '--medvram' for next restart"
      else
        HEAL_EXTRA_ARGS=(--lowvram)
        log_warn "[SELF-HEAL] Detected repeated CUDA OutOfMemoryError -> escalating to '--lowvram'"
      fi
      build_launch_args
    fi
  fi

  # 2. Disk Space Exhausted -> purge pip/uv/tmp caches
  if printf '%s' "$tail_txt" | grep -qiE "No space left on device|ENOSPC"; then
    log_warn "[SELF-HEAL] Detected 'No space left on device' -> purging tmp, pip, and uv caches"
    rm -rf "${WORKSPACE_DIR}/tmp/"* "${WORKSPACE_DIR}/.cache/pip/"* "${WORKSPACE_DIR}/.cache/uv/"* 2>/dev/null || true
  fi

  # 3. Broken Extension Crash -> detect offending extension in traceback and quarantine
  local bad_ext
  bad_ext="$(printf '%s' "$tail_txt" | grep -oP "${SDNEXT_DIR}/extensions/\K[^/]+" | tail -n1 || true)"
  if [[ -n "$bad_ext" && -d "${SDNEXT_DIR}/extensions/${bad_ext}" ]]; then
    mkdir -p "${SDNEXT_DIR}/extensions-disabled"
    log_warn "[SELF-HEAL] Detected crash referencing extension '${bad_ext}' -> quarantining to extensions-disabled/${bad_ext}"
    mv -f "${SDNEXT_DIR}/extensions/${bad_ext}" "${SDNEXT_DIR}/extensions-disabled/${bad_ext}-${RUN_STAMP}" 2>/dev/null || true
  fi

  # 4. Create & upload diagnostic crash bundle
  local bundle="${LOG_DIR}/crash-bundle-${RUN_STAMP}-r${crash_num}.tar.gz"
  local diag_txt="${LOG_DIR}/crash-diag-r${crash_num}.txt"
  {
    echo "=== SD.Next Crash Diagnostic Report (v${SCRIPT_VERSION}, restart #${crash_num}) ==="
    date -u
    echo "Launch args: ${LAUNCH_ARGS[*]}"
    echo "--- GPU Status ---"
    nvidia-smi 2>/dev/null || true
    echo "--- Disk Status ---"
    df -h "$WORKSPACE_DIR" 2>/dev/null || true
    echo "--- Last 100 lines of Server Log ---"
    printf '%s\n' "$tail_txt"
  } >"$diag_txt"
  tar -czf "$bundle" -C "$LOG_DIR" "$(basename "$diag_txt")" "$(basename "$LOG_FILE")" "$(basename "$JSON_LOG_FILE")" 2>/dev/null || true
  if (( DRIVE_READY )) && [[ -f "$bundle" ]]; then
    rclone copyto --timeout 30s "$bundle" "${DRIVE_REMOTE}:${DRIVE_ROOT}/logs/$(basename "$bundle")" 2>/dev/null || true
  fi
  if (( PHONE_READY )) && [[ -f "$bundle" ]]; then
    local target_root="${PHONE_REMOTE_NAME}:${PHONE_ROOT}"
    [[ "$PHONE_PROTOCOL" == "webdav" ]] && target_root="${PHONE_REMOTE_NAME}:"
    _phone_rclone copyto --timeout 30s "$bundle" "${target_root}/logs/$(basename "$bundle")" 2>/dev/null || true
  fi
}

start_server_child() {
  SERVER_LOG="${LOG_DIR}/sdnext-server-${RUN_STAMP}.log"
  ln -sfn "$SERVER_LOG" "${LOG_DIR}/sdnext-server-latest.log"
  configure_tcmalloc
  pushd "$SDNEXT_DIR" >/dev/null
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  export HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
  export TORCH_HOME="${WORKSPACE_DIR}/.cache/torch"
  export PYTHONUNBUFFERED=1
  mkdir -p "$HF_HOME" "$TORCH_HOME"
  "$VENV_PY" webui.py "${LAUNCH_ARGS[@]}" >>"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" >"${LOG_DIR}/sdnext-server.pid"
  popd >/dev/null
  log_info "Launched SD.Next server PID ${SERVER_PID} (log: ${SERVER_LOG})"
}

stop_server_child() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log_info "Stopping SD.Next server PID ${SERVER_PID}..."
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    local waited=0
    while kill -0 "$SERVER_PID" 2>/dev/null && (( waited < 20 )); do
      sleep 1
      waited=$(( waited + 1 ))
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

launch_supervised() {
  set_stage "supervise"
  local restarts=0 backoff=10 rc=0 first_start=1

  drive_sync_loop & SYNC_PID=$!
  phone_sync_loop & PHONE_SYNC_PID=$!
  start_instant_output_watcher
  start_idle_cost_guard

  trap 'log_info "Shutdown signal received — running final Google Drive & Phone backup"; drive_push; phone_push; stop_server_child; exit 0' TERM INT

  while true; do
    start_server_child
    if ! wait_for_http "http://127.0.0.1:${PORT}/" "$HEALTH_TIMEOUT"; then
      log_error "Health check failed — analyzing failure & restarting server child"
      stop_server_child
      rc=7
    else
      if (( first_start )); then
        notify "ONLINE" "SD.Next UI ready on port ${PORT} (${DETECTED_GPU_NAME})"
        smoke_test
        first_start=0
      fi
      # Active runtime monitoring loop
      set +E
      while kill -0 "$SERVER_PID" 2>/dev/null; do
        sleep "$RUNTIME_HEALTH_INTERVAL"
      done
      wait "$SERVER_PID" 2>/dev/null
      rc=$?
      set -E
    fi

    SERVER_PID=""
    restarts=$(( restarts + 1 ))
    analyze_and_heal_failure "$restarts" || true

    if (( SUPERVISE_MAX_RESTARTS > 0 && restarts >= SUPERVISE_MAX_RESTARTS )); then
      drive_push
      phone_push
      die 7 "SD.Next crashed ${restarts} times in a row — max restarts reached"
    fi
    notify "RESTARTING" "SD.Next exited (rc=${rc}) — self-healing restart #${restarts} in ${backoff}s"
    log_warn "SD.Next exited (rc=${rc}) — restart #${restarts} in ${backoff}s"
    sleep "$backoff"
    backoff=$(( backoff * 2 ))
    (( backoff > SUPERVISE_MAX_BACKOFF )) && backoff=$SUPERVISE_MAX_BACKOFF
  done
}

launch_background() {
  set_stage "background"
  start_instant_output_watcher
  start_idle_cost_guard
  start_server_child
  wait_for_http "http://127.0.0.1:${PORT}/" "$HEALTH_TIMEOUT" || die 7 "Startup health check failed"
  smoke_test
  notify "ONLINE" "SD.Next running in background (PID ${SERVER_PID}) on port ${PORT}"
  log_info "SD.Next running detached (PID ${SERVER_PID})"
  trap - TERM INT
}

launch_foreground() {
  set_stage "foreground"
  SERVER_LOG="${LOG_DIR}/sdnext-server-${RUN_STAMP}.log"
  ln -sfn "$SERVER_LOG" "${LOG_DIR}/sdnext-server-latest.log"
  configure_tcmalloc
  pushd "$SDNEXT_DIR" >/dev/null
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  export HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
  export TORCH_HOME="${WORKSPACE_DIR}/.cache/torch"
  export PYTHONUNBUFFERED=1
  mkdir -p "$HF_HOME" "$TORCH_HOME"
  "$VENV_PY" webui.py "${LAUNCH_ARGS[@]}" 2>&1 | tee -a "$SERVER_LOG"
  popd >/dev/null || true
}

launch_server() {
  if (( SKIP_LAUNCH )); then
    log_info "--skip-launch set: server will not be started"
    return 0
  fi
  build_launch_args
  log_info "Final webui.py launch args: ${LAUNCH_ARGS[*]}"
  case "$RUN_MODE" in
    supervise)  launch_supervised ;;
    background) launch_background ;;
    foreground) launch_foreground ;;
    *) die 2 "Unknown RUN_MODE '${RUN_MODE}' (supervise|background|foreground)" ;;
  esac
}

#-----------------------------------------------------------------------------
# Summary & Dry-Run Reporting
#-----------------------------------------------------------------------------
print_summary() {
  local url="http://127.0.0.1:${PORT}"
  {
    echo "=================================================================="
    echo " SD.Next Deployment Summary (v${SCRIPT_VERSION})"
    echo "=================================================================="
    echo " Install Dir  : ${SDNEXT_DIR}"
    echo " Git Commit   : $(git -C "$SDNEXT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo " Detected GPU : ${DETECTED_GPU_NAME} (${VRAM_MB} MB VRAM)"
    echo " Run Mode     : ${RUN_MODE}${IDLE_SHUTDOWN_MINUTES:+ (idle cost guard: ${IDLE_SHUTDOWN_MINUTES}m)}"
    echo " Local UI     : ${url}"
    if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
      echo " Public UI    : https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net"
    fi
    if (( POD_WEBDAV_ENABLED )); then
      echo " Phone WebDAV : port ${POD_WEBDAV_PORT} (user=${POD_WEBDAV_USER})"
      [[ -n "${RUNPOD_POD_ID:-}" ]] && echo " WebDAV Proxy : https://${RUNPOD_POD_ID}-${POD_WEBDAV_PORT}.proxy.runpod.net"
    fi
    echo " Deploy Log   : ${LOG_FILE}"
    echo " JSONL Log    : ${JSON_LOG_FILE}"
    echo " Server Log   : ${LOG_DIR}/sdnext-server-latest.log"
    echo " Extensions   : ${#EXT_OK[@]} ok, ${#EXT_FAILED[@]} failed"
    echo " Models/Assets: ${#MODEL_OK[@]} downloaded, ${#MODEL_SKIPPED[@]} skipped, ${#MODEL_FAILED[@]} failed"
    if (( DRIVE_READY )); then
      echo " Google Drive : ${DRIVE_REMOTE}:${DRIVE_ROOT} (mode=${DRIVE_MOUNT_MODE}, every ${DRIVE_SYNC_MINUTES}m + real-time inotify)"
    else
      echo " Google Drive : inactive (use --drive-auth or RCLONE_CONFIG_B64)"
    fi
    if (( PHONE_READY )); then
      echo " Phone Storage: ${PHONE_PROTOCOL}://${PHONE_HOST}:${PHONE_PORT}${PHONE_ROOT} (every ${PHONE_SYNC_MINUTES}m + real-time inotify)"
    elif (( PHONE_ENABLED )); then
      echo " Phone Storage: configured but unreachable (run --phone-setup-guide)"
    else
      echo " Phone Storage: inactive (use --phone-host / --phone-webdav / --phone-setup-guide)"
    fi
    echo " Start Script : ${WORKSPACE_DIR}/start-sdnext.sh"
    echo "=================================================================="
  } | tee -a "$LOG_FILE" >&2
}

print_plan() {
  build_launch_args
  log_info "================ DRY RUN PLAN (v${SCRIPT_VERSION}) ================"
  log_info "Workspace        : ${WORKSPACE_DIR}"
  log_info "SD.Next Repo     : ${SDNEXT_REPO} @ ${SDNEXT_BRANCH}${SDNEXT_COMMIT:+ (${SDNEXT_COMMIT})}"
  log_info "Install Dir      : ${SDNEXT_DIR}"
  log_info "Detected GPU     : ${DETECTED_GPU_NAME} (${VRAM_MB} MB VRAM)"
  log_info "Extensions       : core=${#CORE_EXTENSIONS[@]} detail=${#DETAIL_EXTENSIONS[@]} popular=${#POPULAR_EXTENSIONS[@]}"
  log_info "Resource Pack    : enabled=${DOWNLOAD_RESOURCE_PACK} (${#UTILITY_RESOURCE_PACK[@]} utility models: 4x-UltraSharp, 4x-AnimeSharp, YOLOv8 face/hand/person, EasyNegative)"
  log_info "Model Profile    : ${MODEL_PROFILE} (~$(catalog_total_gb) GB)"
  local entry id kind size flags note
  local IFS='|'
  for entry in "${MODEL_CATALOG[@]}"; do
    read -r id kind _ _ _ size flags note <<< "$entry"
    model_in_active_profile "$id" || continue
    log_info "  - ${id}: ~${size} GB [${flags:-open}] ${note}"
  done
  log_info "Google Drive     : enabled=${DRIVE_ENABLED} remote=${DRIVE_REMOTE}:${DRIVE_ROOT} mode=${DRIVE_MOUNT_MODE}"
  log_info "Phone Storage    : enabled=${PHONE_ENABLED} host=${PHONE_HOST:-none}:${PHONE_PORT} proto=${PHONE_PROTOCOL} pod_webdav=${POD_WEBDAV_ENABLED} syncthing=${SYNCTHING_ENABLED}"
  log_info "Real-time Sync   : inotify=${INSTANT_OUTPUT_SYNC} idle_cost_guard=${IDLE_SHUTDOWN_MINUTES}m"
  log_info "Launch Args      : ${LAUNCH_ARGS[*]}"
  log_info "==============================================================="
}

#-----------------------------------------------------------------------------
# Main Entrypoint
#-----------------------------------------------------------------------------
main() {
  log_info "deploy-sdnext.sh v${SCRIPT_VERSION} starting (run ${RUN_STAMP})"

  if (( PHONE_SETUP_GUIDE_ONLY )); then
    print_phone_setup_guide
    exit 0
  fi
  if (( WIZARD_MODE )); then
    run_interactive_wizard
  fi
  if (( LIST_GPUS_ONLY )); then
    list_and_rank_runpod_gpus table
    exit 0
  fi
  if (( PROVISION_POD )); then
    provision_runpod_pod
    exit 0
  fi
  if (( DRIVE_INTERACTIVE_AUTH && ! DRIVE_PULL_ONLY && ! DRIVE_PUSH_ONLY && SKIP_LAUNCH )); then
    run_drive_interactive_auth
    exit 0
  fi

  if (( LOG_RETENTION_DAYS > 0 )); then
    find "$LOG_DIR" -maxdepth 1 -type f \( -name 'sdnext-*.log' -o -name 'sdnext-*.jsonl' \) \
      -mtime +"$LOG_RETENTION_DAYS" -delete 2>/dev/null || true
  fi

  if (( DRIVE_PULL_ONLY )); then
    install_rclone
    drive_setup
    drive_pull
    exit 0
  fi
  if (( DRIVE_PUSH_ONLY )); then
    install_rclone
    drive_setup
    drive_push
    exit 0
  fi
  if (( PHONE_PULL_ONLY )); then
    install_rclone
    phone_setup
    phone_pull
    exit 0
  fi
  if (( PHONE_PUSH_ONLY )); then
    install_rclone
    phone_setup
    phone_push
    exit 0
  fi

  exec 9>"${WORKSPACE_DIR}/.deploy-sdnext.lock"
  flock -n 9 || die 1 "Another deploy-sdnext.sh instance is already running (lock: ${WORKSPACE_DIR}/.deploy-sdnext.lock)"

  preflight

  if (( DRY_RUN )); then
    print_plan
    log_info "Dry run complete — no changes made"
    exit 0
  fi

  install_system_deps
  drive_setup
  phone_setup
  start_pod_webdav_server
  start_syncthing_p2p
  sync_repo
  setup_venv
  run_installer
  sync_extensions
  drive_pull
  phone_pull
  seed_default_config
  download_models
  drive_push
  phone_push
  write_config_snapshot
  install_self_and_start_script
  print_summary
  launch_server
}

main "$@"
