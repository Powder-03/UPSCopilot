#!/bin/bash
# ==============================================================================
# UPSCopilot EC2 Coordinator Setup Script (Free-Tier t3.micro Ready)
# Configures 3 GB Swap File, system optimizations, and environment for running
# the FastAPI web coordinator and SQS worker safely on a 1 GB RAM instance.
# ==============================================================================

set -euo pipefail

echo "=================================================================="
echo "Starting UPSCopilot EC2 Provisioning..."
echo "=================================================================="

# 1. Update system packages
echo "[1/5] Updating system packages..."
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y curl git build-essential htop ufw

# 2. Configure 3 GB Swap File on the 30 GB EBS Root SSD
echo "[2/5] Configuring 3 GB Swap File to prevent memory cliff..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 3G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "Swap file created successfully."
else
    echo "Swap file already exists. Skipping creation."
fi

# Set optimal swappiness (10: only swap under true memory pressure)
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

# 3. Install Python 3.11+, uv, and Docker
echo "[3/5] Installing uv package manager and Docker..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

if ! command -v docker &> /dev/null; then
    sudo apt-get install -y docker.io docker-compose-v2
    sudo usermod -aG docker "$USER"
    sudo systemctl enable docker
    sudo systemctl start docker
fi

# 4. Clone or pull repo (if script executed standalone)
echo "[4/5] Verifying workspace..."
echo "Memory status:"
free -h

# 5. Create Systemd Services for Coordinator and Worker
echo "[5/5] Creating systemd services for UPSCopilot coordinator and worker..."

# A. FastAPI Web Portal & Coordinator Service
cat << 'EOF' | sudo tee /etc/systemd/system/upscopilot.service
[Unit]
Description=UPSCopilot EC2 Coordinator and Web API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/evaluator
EnvironmentFile=/home/ubuntu/evaluator/.env
ExecStart=/home/ubuntu/.cargo/bin/uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

# B. SQS Evaluation Worker Service (Polls SQS, fans out 20 Lambdas)
cat << 'EOF' | sudo tee /etc/systemd/system/upscopilot-worker.service
[Unit]
Description=UPSCopilot EC2 SQS Evaluation Worker (20-Lambda Fan-Out)
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/evaluator
EnvironmentFile=/home/ubuntu/evaluator/.env
ExecStart=/home/ubuntu/.cargo/bin/uv run python -m src.handlers.ec2_worker
Restart=always
RestartSec=5
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo "Systemd services created at /etc/systemd/system/upscopilot.service and upscopilot-worker.service"

echo "=================================================================="
echo "EC2 Setup Complete! Available Memory:"
free -h
echo ""
echo "To start the services on EC2:"
echo "  1. Copy your .env secrets into /home/ubuntu/evaluator/.env"
echo "  2. Start Web Portal: sudo systemctl enable --now upscopilot"
echo "  3. Start SQS Worker: sudo systemctl enable --now upscopilot-worker"
echo "  4. Check logs: journalctl -u upscopilot-worker -f"
echo "=================================================================="

