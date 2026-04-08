#!/bin/bash
# Server Setup Script for AML Detection System

set -e

echo "=========================================="
echo "AML Detection System - Server Setup"
echo "=========================================="

# Update system
echo "[1/8] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python and dependencies
echo "[2/8] Installing Python 3.10 and pip..."
sudo apt-get install -y python3.10 python3.10-venv python3-pip nginx

# Create application directory
echo "[3/8] Setting up application directory..."
sudo mkdir -p /opt/aml-detection
sudo chown ubuntu:ubuntu /opt/aml-detection

# Create virtual environment
echo "[4/8] Creating Python virtual environment..."
cd /opt/aml-detection
python3.10 -m venv venv
source venv/bin/activate

# Install Python packages
echo "[5/8] Installing Python dependencies..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric flask flask-cors gunicorn pandas numpy scikit-learn joblib

# Configure Nginx
echo "[6/8] Configuring Nginx..."
sudo tee /etc/nginx/sites-available/aml-detection << 'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/aml-detection /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx

# Create systemd service
echo "[7/8] Creating systemd service..."
sudo tee /etc/systemd/system/aml-detection.service << 'EOF'
[Unit]
Description=AML Detection System
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/aml-detection
Environment="PATH=/opt/aml-detection/venv/bin"
ExecStart=/opt/aml-detection/venv/bin/gunicorn --workers 2 --bind 127.0.0.1:5000 --timeout 120 deployment.app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload

echo ""
echo "=========================================="
echo "Server setup complete!"
echo "=========================================="
echo "Next: Upload project files to /opt/aml-detection"
echo "Then: sudo systemctl start aml-detection"
