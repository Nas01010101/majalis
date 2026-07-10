# Deploy — Alibaba Cloud ECS

The competition requires the backend to RUN on Alibaba Cloud (proof = short
recording + a repo file using AliCloud services). This service is
self-contained: one small ECS instance, no managed extras.

## One-time provision (aliyun CLI or console)

- ECS `ecs.t6-c1m1.small` (1 vCPU / 1 GiB) or larger, Ubuntu 24.04,
  region from `ALIBABA_CLOUD_REGION` in `.env`.
- Security group: allow inbound TCP 8080 (API + dashboard) and 22 (SSH).

## Install & run on the instance

```bash
sudo apt-get update && sudo apt-get install -y python3-venv git
git clone https://github.com/Nas01010101/agora && cd agora
python3 -m venv .venv && .venv/bin/pip install -e ".[api]"
# secrets: copy .env to the instance out-of-band (scp); never commit it
scp .env <instance>:agora/.env && chmod 600 .env

# run under systemd so it survives reboots
sudo tee /etc/systemd/system/agora.service <<'EOF'
[Unit]
Description=Agora debate society API
After=network.target
[Service]
WorkingDirectory=/root/agora
ExecStart=/root/agora/.venv/bin/uvicorn agora.api:app --host 0.0.0.0 --port 8080
Restart=on-failure
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now agora
curl -s localhost:8080/healthz   # {"ok": true}
```

## Proof-of-deploy checklist (submission)

1. Screen recording: ECS console showing the instance + `curl /healthz` +
   one `/ingest` + `/ask` round-trip from the public IP.
2. Code pointer for the submission form: `src/agora/api.py` (service) +
   `src/agora/llm.py` (DashScope/Qwen Cloud API usage).
