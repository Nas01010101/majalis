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
git clone https://github.com/Nas01010101/majalis && cd majalis
python3 -m venv .venv && .venv/bin/pip install -e ".[api]"
# secrets: copy .env to the instance out-of-band (scp); never commit it
scp .env <instance>:majalis/.env && chmod 600 .env

# run under systemd so it survives reboots
sudo tee /etc/systemd/system/majalis.service <<'EOF'
[Unit]
Description=Majalis debate society API
After=network.target
[Service]
WorkingDirectory=/root/majalis
ExecStart=/root/majalis/.venv/bin/uvicorn majalis.api:app --host 0.0.0.0 --port 8080
Restart=on-failure
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now majalis
curl -s localhost:8080/healthz   # {"ok": true}
```

## Update an existing instance (redeploy current HEAD)

The running instance keeps its own `.env` + `.venv`; sync only the code and
restart — no secrets touched, no git auth needed on the box:

```bash
rsync -az --delete --exclude='.env' --exclude='.venv' --exclude='.git' \
  --exclude='__pycache__' -e "ssh -i ~/.ssh/agora_deploy" \
  ./ root@<instance>:/root/majalis/
ssh -i ~/.ssh/agora_deploy root@<instance> \
  'cd /root/majalis && .venv/bin/pip install -q -e ".[api]" && systemctl restart majalis'
curl -s http://<instance>:8080/healthz   # {"ok": true}
curl -s http://<instance>:8080/board -o /dev/null -w "%{http_code}\n"  # 200
```

## Proof-of-deploy checklist (submission)

1. Screen recording: ECS console showing the instance + `curl /healthz` +
   one `/ingest` + `/ask` round-trip from the public IP.
2. Code pointer for the submission form: `src/majalis/api.py` (service) +
   `src/majalis/llm.py` (DashScope/Qwen Cloud API usage).
