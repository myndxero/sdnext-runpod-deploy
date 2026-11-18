# SD.Next RunPod Automated Deployment Suite (`v3.1.0`)

End-to-end automated deployment, **Front-End RunPod GPU Selector & Provisioner**, **Google Drive** & **Direct Phone Storage (Android/iOS)** integration, **2026 Model & Extension Suite**, **Real-Time Output Streamer**, **GPU Idle Cost Guard**, and **Self-Healing Supervisor** for [SD.Next (`vladmandic/sdnext`)](https://github.com/vladmandic/sdnext).

---

## Features at a Glance

1. **Front-End RunPod GPU Auto-Selector & Pod Provisioner (`--provision` / `--wizard` / `--list-gpus`)**
   - Queries the RunPod GraphQL API for live GPU stock, Secure vs. Community cloud availability, and hourly pricing.
   - Automatically matches the best-value GPU tier (`16gb`, `24gb`, `48gb`, `80gb`, or `auto` based on your chosen model profile).
   - Creates the Pod, mounts `/workspace`, base64-injects `deploy-sdnext.sh` and your tokens/configs (`RCLONE_CONFIG_B64`, `HF_TOKEN`, `CIVITAI_TOKEN`, `TAILSCALE_AUTH_KEY`), and boots the entire stack unattended.

2. **Google Drive Bidirectional Integration (`rclone`)**
   - **3 Auth Modes**: Base64 config (`RCLONE_CONFIG_B64`), zero-OAuth Google Cloud Service Account (`GDRIVE_SERVICE_ACCOUNT_B64`), or interactive headless wizard (`./deploy-sdnext.sh --drive-auth`).
   - **3 Storage Modes**: `sync` (periodic + real-time delta copy), `mount` (`rclone mount` FUSE VFS cache), or `hybrid`.
   - Restores `models/`, `models/Lora`, `models/embeddings`, `models/ESRGAN`, `models/yolo`, `wildcards/`, and `config.json` / `ui-config.json` / `styles.csv` on boot, and backs up `outputs/`, configs, and crash bundles.

3. **Direct Phone Storage Integration (Android & iOS — 3 Built-In Modes)**
   - **Mode A (`--phone-host 100.x.y.z` — Phone as Remote Storage like Google Drive)**:
     - Runs **Userspace Tailscale** (`tailscaled --tun=userspace-networking`, requiring zero root `/dev/net/tun` privileges inside RunPod containers) + `rclone` over SFTP/WebDAV/FTP.
     - Connects to your Android phone (**Termux `sshd`**, **Material Files**, **Cx File Explorer**, **Round Sync**) or iPhone/iPad (**Secure ShellFish**, **FE File Explorer**, **Documents**).
     - Pulls checkpoints, LoRAs, embeddings, init-images (`inputs/`), and configs from your phone on boot (capped by `PHONE_MAX_PULL_GB`), and pushes generated images/videos directly to your phone.
   - **Mode B (`--phone-webdav` — Mount the RunPod Pod as a Native Drive on Your Phone)**:
     - Starts an authenticated WebDAV server on port `7861` (`https://<pod-id>-7861.proxy.runpod.net`).
     - Mount directly inside iOS **Files** (*Connect to Server*) or Android file managers to browse `outputs/` and drag-and-drop LoRAs or camera photos into the pod.
   - **Mode C (`SYNCTHING_ENABLED=1` — Peer-to-Peer Real-Time Sync)**:
     - Runs a headless Syncthing P2P daemon that pairs with **Syncthing-Fork** (Android) or **Möbius Sync** (iOS) using your phone's 56-character Device ID over encrypted global relays.

4. **2026 Model Profiles, CivitAI API v1 Resolver & Utility Resource Pack**
   - **Profiles**: `minimal` (SDXL + VAE), `core` (SD 1.5, SDXL, FLUX.1 Schnell FP8, FLUX.2 Klein 4B FP8), `flux` (adds FLUX.1 Dev & FLUX 3 Image Klein 5.6B), `video` (adds Wan 2.2 T2V 14B & HunyuanVideo), and `full` (adds Qwen-Image 2512 & Edit 2511).
   - **Utility Resource Pack (`DOWNLOAD_RESOURCE_PACK=1`)**: Automatically installs `4x-UltraSharp.pth`, `4x-AnimeSharp.pth`, YOLOv8 detailer detectors (`face_yolov8m.pt`, `hand_yolov8n.pt`, `person_yolov8m-seg.pt`), and `EasyNegative.safetensors`.
   - **CivitAI API v1 Resolver**: Pass `civitai://VERSION_ID` or `https://civitai.com/api/download/models/VERSION_ID` to `MODEL_URLS`, `LORA_URLS`, or `EMBEDDING_URLS` — resolves the true `.safetensors` filename automatically and verifies header integrity.

5. **Real-Time Output Streamer, GPU Idle Cost Guard & Self-Healing Supervisor**
   - **Instant Output Watcher (`inotifywait`)**: Watches `outputs/` for file-write completion and pushes new images/videos to Google Drive and Phone storage within ~4 seconds.
   - **GPU Idle Cost Guard (`--idle-shutdown MIN`)**: Monitors `nvidia-smi` utilization every 30s; if GPU usage stays below `3%` for `MIN` minutes, runs a final Drive + Phone backup and stops the RunPod pod automatically.
   - **Self-Healing Crash Analyzer**: Inspects `sdnext.log` on crash to auto-remediate `CUDA out of memory` (escalates `--medvram` -> `--lowvram`), `No space left on device` (purges tmp/pip/uv caches), and faulty extensions (quarantines crashing extension to `extensions-disabled/`), then uploads a `.tar.gz` diagnostic crash bundle to Google Drive and Phone storage.

---

## Quick Start

### Option 1: One-Line Bootstrap Inside a RunPod Pod
Paste into your RunPod Pod terminal or Pod Template **Start Command**:
```bash
curl -fsSL https://raw.githubusercontent.com/myndxero/sdnext-runpod-deploy/main/deploy-sdnext.sh -o /workspace/deploy-sdnext.sh && chmod +x /workspace/deploy-sdnext.sh && /workspace/deploy-sdnext.sh
```

### Option 2: Front-End GPU Selection & Pod Provisioning (From Your Local Machine)
```bash
git clone https://github.com/myndxero/sdnext-runpod-deploy.git
cd sdnext-runpod-deploy

# View live RunPod GPU stock & hourly prices:
export RUNPOD_API_KEY="rp_your_api_key"
./deploy-sdnext.sh --list-gpus --gpu-tier 24gb

# Auto-select the best 24GB+ GPU and deploy the pod with your settings:
export RCLONE_CONFIG_B64="$(base64 -w0 ~/.config/rclone/rclone.conf)"
export HF_TOKEN="hf_your_token"
export CIVITAI_TOKEN="your_civitai_token"
./deploy-sdnext.sh --provision --gpu-tier 24gb --model-profile core --idle-shutdown 45

# Or run the interactive TUI wizard:
./deploy-sdnext.sh --wizard
```

### Option 3: Connect Your Phone's Storage (Android & iOS)
View the full built-in setup walkthrough anytime:
```bash
./deploy-sdnext.sh --phone-setup-guide
```

#### Direct Pull/Push from Your Phone (SFTP / WebDAV over Userspace Tailscale)
```bash
export TAILSCALE_AUTH_KEY="tskey-auth-xxxxx"
export PHONE_PASS="your-phone-sftp-password"

./deploy-sdnext.sh \
  --phone-host 100.x.y.z \
  --phone-port 8022 \
  --phone-protocol sftp \
  --phone-user u0_a245 \
  --phone-path /sdcard/SDNext
```

#### Mount the Pod as a Network Drive on iOS "Files" or Android
```bash
POD_WEBDAV_USER="sdnext" POD_WEBDAV_PASS="secret123" ./deploy-sdnext.sh --phone-webdav
# Connect from your phone to: https://<pod-id>-7861.proxy.runpod.net
```

---

## CLI Reference

```text
Usage: deploy-sdnext.sh [options]

Front-End GPU Selection & RunPod Provisioning:
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
```
