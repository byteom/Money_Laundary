# AWS Deployment Guide - AML Detection System

## 🌐 Live Deployment

Your Money Laundering Detection System is now **LIVE** on AWS!

### Access URLs

| Resource | URL |
|----------|-----|
| **Web Interface** | http://13.126.159.27/ |
| **Health Check** | http://13.126.159.27/health |
| **Prediction API** | http://13.126.159.27/predict |

---

## 📊 Testing the System

### 1. Web Interface
Open your browser and go to: **http://13.126.159.27/**

The advanced web interface allows you to:
- Enter transaction details
- Select from pre-defined test scenarios
- View predictions from all models (Baseline, GraphSAGE, TGN, Ensemble)
- Compare model performance
- See risk classification with color-coded results

### 2. API Testing (curl)

**Health Check:**
```bash
curl http://13.126.159.27/health
```

**Prediction Request:**
```bash
curl -X POST http://13.126.159.27/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sender_id": "ACC_0001",
    "receiver_id": "ACC_0002",
    "transaction_amount": 15000,
    "timestamp": "2024-01-15 14:30:00",
    "transaction_type": "transfer"
  }'
```

### 3. API Testing (PowerShell)

```powershell
$body = @{
    sender_id = "ACC_0001"
    receiver_id = "ACC_0002"
    transaction_amount = 15000
    timestamp = "2024-01-15 14:30:00"
    transaction_type = "transfer"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://13.126.159.27/predict" -Method Post -Body $body -ContentType "application/json"
```

### 4. API Testing (Python)

```python
import requests

response = requests.post(
    "http://13.126.159.27/predict",
    json={
        "sender_id": "ACC_0001",
        "receiver_id": "ACC_0002",
        "transaction_amount": 15000,
        "timestamp": "2024-01-15 14:30:00",
        "transaction_type": "transfer"
    }
)
print(response.json())
```

---

## 📝 API Reference

### POST /predict

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| sender_id | string | Yes | Account ID of sender (e.g., "ACC_0001") |
| receiver_id | string | Yes | Account ID of receiver (e.g., "ACC_0002") |
| transaction_amount | number | Yes | Transaction amount in currency units |
| timestamp | string | Yes | ISO format datetime (e.g., "2024-01-15 14:30:00") |
| transaction_type | string | Yes | One of: cash_out, deposit, payment, transfer, withdrawal |

**Response:**
```json
{
  "fraud_probability": 0.8333,
  "risk_classification": "high",
  "ensemble_probability": 0.8333,
  "baseline_probability": 0.041,
  "graphsage_probability": 1.0,
  "tgn_probability": 1.0,
  "model_weights": {
    "baseline": 0.424,
    "graphsage": 0.483,
    "tgn": 0.093
  }
}
```

### Risk Classification
| Probability Range | Classification | Color |
|-------------------|----------------|-------|
| < 0.3 | low | Green |
| 0.3 - 0.7 | medium | Orange |
| > 0.7 | high | Red |

---

## 🏗️ AWS Infrastructure Details

### EC2 Instance
| Property | Value |
|----------|-------|
| Instance ID | i-06e112358555f3546 |
| Instance Type | t3.micro (Free Tier eligible) |
| Region | ap-south-1 (Mumbai) |
| Elastic IP | 13.126.159.27 |
| OS | Ubuntu 22.04 LTS |
| Storage | 20 GB gp3 EBS |

### Security Group
| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | 0.0.0.0/0 | SSH |
| 80 | TCP | 0.0.0.0/0 | HTTP (nginx) |
| 443 | TCP | 0.0.0.0/0 | HTTPS (future) |
| 5000 | TCP | 0.0.0.0/0 | Flask API (direct) |

### Service Configuration
- **Web Server**: nginx (reverse proxy)
- **Application Server**: gunicorn (2 workers)
- **Service Manager**: systemd (auto-restart enabled)

---

## 🔧 Server Management

### SSH Access
```bash
ssh -i ~/.ssh/aml-detection-key.pem ubuntu@13.126.159.27
```

### Service Commands
```bash
# Check status
sudo systemctl status aml-detection

# Restart service
sudo systemctl restart aml-detection

# View logs
sudo journalctl -u aml-detection -f

# Stop service
sudo systemctl stop aml-detection

# Start service
sudo systemctl start aml-detection
```

### Nginx Commands
```bash
# Check status
sudo systemctl status nginx

# Restart nginx
sudo systemctl restart nginx

# Test config
sudo nginx -t
```

---

## 📁 Server File Structure

```
/opt/aml-detection/
├── venv/                    # Python virtual environment
├── artifacts/
│   ├── models/              # Trained ML models
│   ├── processed/           # Preprocessed data
│   └── reports/             # Evaluation reports
├── data/                    # Raw transaction data
├── deployment/
│   ├── app.py               # Flask API
│   └── static/
│       └── index.html       # Web interface
├── graph/                   # Graph construction modules
├── models/                  # Model architectures
├── preprocessing/           # Data preprocessing
└── training/                # Training scripts
```

---

## 💰 Cost Information

### Current Setup (Free Tier Eligible)
- **EC2 t3.micro**: Free for 750 hours/month (first 12 months)
- **EBS Storage**: 30 GB free per month
- **Elastic IP**: Free when attached to running instance
- **Data Transfer**: 1 GB outbound free per month

### Monthly Cost After Free Tier
- EC2 t3.micro: ~$8-10/month
- EBS 20 GB: ~$2/month
- **Total**: ~$10-12/month

---

## 🔄 Updating the Deployment

### Upload New Code
```powershell
$KEY_PATH = "C:\Users\anwee\.ssh\aml-detection-key.pem"

# Upload updated files
scp -r -i $KEY_PATH models/ ubuntu@13.126.159.27:/opt/aml-detection/
scp -r -i $KEY_PATH deployment/ ubuntu@13.126.159.27:/opt/aml-detection/

# Restart service
ssh -i $KEY_PATH ubuntu@13.126.159.27 "sudo systemctl restart aml-detection"
```

### Retrain Models on Server
```bash
cd /opt/aml-detection
source venv/bin/activate
python preprocessing/preprocess.py
python training/train.py
sudo systemctl restart aml-detection
```

---

## 🛑 Terminating the Deployment

To stop incurring charges:

### Option 1: Stop Instance (Preserves Data)
```bash
aws ec2 stop-instances --instance-ids i-06e112358555f3546
```
Note: Elastic IP will incur charges (~$0.005/hour) when not attached to running instance.

### Option 2: Terminate Instance (Deletes Everything)
```bash
# Release Elastic IP first
aws ec2 release-address --allocation-id <allocation-id>

# Terminate instance
aws ec2 terminate-instances --instance-ids i-06e112358555f3546
```

---

## 🔒 Security Recommendations

For production use, consider:

1. **HTTPS**: Add SSL certificate (Let's Encrypt)
2. **Firewall**: Restrict SSH to your IP only
3. **Authentication**: Add API key authentication
4. **Rate Limiting**: Prevent abuse
5. **Monitoring**: Set up CloudWatch alarms
6. **Backups**: Enable EBS snapshots

---

## 📞 Troubleshooting

### Service Not Starting
```bash
sudo journalctl -u aml-detection -n 100 --no-pager
```

### API Returns 500 Error
```bash
# Check gunicorn logs
sudo journalctl -u aml-detection -f

# Restart service
sudo systemctl restart aml-detection
```

### Cannot Access via Browser
1. Check security group allows port 80
2. Check nginx is running: `sudo systemctl status nginx`
3. Check instance is running in AWS console

---

## ✅ Deployment Summary

| Component | Status |
|-----------|--------|
| EC2 Instance | ✅ Running |
| Elastic IP | ✅ Assigned (13.126.159.27) |
| nginx | ✅ Configured |
| gunicorn | ✅ Running |
| Flask API | ✅ Healthy |
| Web Interface | ✅ Accessible |
| Models | ✅ Loaded |

**Your AML Detection System is live at: http://13.126.159.27/**

---

*Deployed on: April 6, 2026*
*Last Updated: April 6, 2026*
