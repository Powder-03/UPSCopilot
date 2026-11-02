#!/bin/bash
# ==============================================================================
# UPSCopilot EC2 Production Update Script
# Updates repository, configures Google SMTP in .env, and restarts systemd services.
# ==============================================================================

set -euo pipefail

echo "=================================================================="
echo "Updating UPSCopilot on EC2 Production Instance..."
echo "=================================================================="

APP_DIR="${HOME}/evaluator"
cd "$APP_DIR"

# 1. Pull latest git commits
echo "[1/4] Pulling latest code from git..."
git pull origin main

# 2. Update .env to enforce Google SMTP
echo "[2/4] Ensuring Google SMTP is configured in .env..."
ENV_FILE="${APP_DIR}/.env"

if [ -f "$ENV_FILE" ]; then
    # Replace EMAIL_PROVIDER=ses or EMAIL_PROVIDER=google with EMAIL_PROVIDER=smtp
    sed -i 's/^EMAIL_PROVIDER=ses/EMAIL_PROVIDER=smtp/' "$ENV_FILE"
    sed -i 's/^EMAIL_PROVIDER=google/EMAIL_PROVIDER=smtp/' "$ENV_FILE"
    
    # Ensure EMAIL_PROVIDER=smtp exists
    if ! grep -q "^EMAIL_PROVIDER=" "$ENV_FILE"; then
        echo "EMAIL_PROVIDER=smtp" >> "$ENV_FILE"
    fi
    
    # Ensure SMTP settings exist
    if ! grep -q "^SMTP_HOST=" "$ENV_FILE"; then
        echo "SMTP_HOST=smtp.gmail.com" >> "$ENV_FILE"
    fi
    if ! grep -q "^SMTP_PORT=" "$ENV_FILE"; then
        echo "SMTP_PORT=587" >> "$ENV_FILE"
    fi
    if ! grep -q "^SMTP_USER=" "$ENV_FILE"; then
        echo "SMTP_USER=yujitodoyuta@gmail.com" >> "$ENV_FILE"
    fi
    if ! grep -q "^SMTP_PASSWORD=" "$ENV_FILE"; then
        echo "SMTP_PASSWORD=dtkyuvivhwoysdtm" >> "$ENV_FILE"
    fi
    if ! grep -q "^MAIL_FROM_NAME=" "$ENV_FILE"; then
        echo 'MAIL_FROM_NAME="UPSC Evaluator"' >> "$ENV_FILE"
    fi
    if ! grep -q "^MAIL_FROM_EMAIL=" "$ENV_FILE"; then
        echo "MAIL_FROM_EMAIL=yujitodoyuta@gmail.com" >> "$ENV_FILE"
    fi
    echo "Updated $ENV_FILE to use Google SMTP."
else
    echo "Warning: $ENV_FILE not found. Using defaults from src/config.py."
fi

# 3. Sync dependencies with uv
echo "[3/4] Syncing virtual environment with uv..."
/usr/local/bin/uv sync || uv sync

# 4. Restart systemd services
echo "[4/4] Restarting systemd services..."
sudo systemctl restart upscopilot upscopilot-worker

echo "=================================================================="
echo "Update complete! Checking service status:"
sudo systemctl status upscopilot-worker --no-pager -n 10 || true
echo "=================================================================="
