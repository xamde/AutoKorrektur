# Frankfurt Cloud SDXL Backend Deployment Guide

This guide provides step-by-step instructions for deploying the **AutoKorrektur SDXL Inpainting Backend** to an ISO 27001-certified German data center (e.g., Hetzner Cloud in Frankfurt/Falkenstein or AWS Frankfurt `eu-central-1`).

---

## 1. Server Requirements

- **Server Location**: Frankfurt am Main, Germany (or Falkenstein/Nuremberg, Germany) for GDPR compliance.
- **Recommended Spec**:
  - **GPU Option (Recommended for speed ~2-3s)**: 1x NVIDIA RTX 4000 / A4000 (16GB VRAM) or RTX 3060/4060 (12GB+ VRAM), 4 vCPUs, 16 GB RAM.
  - **CPU-Only Option (Inference ~15-20s)**: 8 vCPUs, 32 GB RAM (e.g., Hetzner CCX33).
- **OS**: Ubuntu 24.04 LTS x86_64.

---

## 2. Docker & Container Deployment

### A. Clone Repository on Server
```bash
git clone https://github.com/xamde/AutoKorrektur.git
cd AutoKorrektur/backend
```

### B. Configure Environment Variables (`.env`)
```bash
cat << 'EOF' > .env
AUTOKORREKTUR_PORT=8000
AUTOKORREKTUR_HOST=0.0.0.0
AUTOKORREKTUR_DEVICE="cuda" # or "cpu"
AUTOKORREKTUR_DAILY_QUOTA_LIMIT=2
AUTOKORREKTUR_REDIS_HOST=localhost
AUTOKORREKTUR_REDIS_PORT=6379
EOF
```

### C. Build & Run with Docker Compose
```bash
docker compose up -d --build
```

---

## 3. Caddy Reverse Proxy with Automatic SSL

Caddy automatically provisions and renews Let's Encrypt SSL certificates.

### Install Caddy
```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install caddy -y
```

### Configure `/etc/caddy/Caddyfile`
```caddy
api.autokorrektur.org {
    reverse_proxy 127.0.0.1:8000

    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
    }
}
```

### Restart Caddy
```bash
sudo systemctl restart caddy
```

---

## 4. Verification
Test endpoint health and SSL certificate:
```bash
curl -i https://api.autokorrektur.org/health
```
Expected response:
```json
{"status": "ok", "redis_connected": true, "sdxl_loaded": true}
```
