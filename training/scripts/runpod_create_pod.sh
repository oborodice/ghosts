#!/usr/bin/env bash
set -euo pipefail

# Network Volumeのデータセンターとpodのデータセンターは一致している必要があるため、volume-id から
# 自動で引く(利用者にデータセンターIDを別途調べさせない)
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
TEMPLATE_ID="runpod-torch-v280"
GPU_ID="NVIDIA GeForce RTX 4090"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <network-volume-id> [pod-name]" >&2
  exit 1
fi

VOLUME_ID="$1"
POD_NAME="${2:-ghosts-$(date +%Y%m%d%H%M%S)}"

DATA_CENTER_ID="$(runpodctl network-volume list | jq -r --arg id "${VOLUME_ID}" '.[] | select(.id == $id) | .dataCenterId')"
if [ -z "${DATA_CENTER_ID}" ]; then
  echo "Network volume ${VOLUME_ID} not found" >&2
  exit 1
fi

echo "Creating pod '${POD_NAME}' on volume ${VOLUME_ID} (datacenter ${DATA_CENTER_ID}) ..."
POD_JSON="$(runpodctl pod create \
  --template-id "${TEMPLATE_ID}" \
  --gpu-id "${GPU_ID}" \
  --cloud-type SECURE \
  --data-center-ids "${DATA_CENTER_ID}" \
  --network-volume-id "${VOLUME_ID}" \
  --volume-mount-path "/workspace" \
  --ports "22/tcp" \
  --name "${POD_NAME}" \
  --wait)"

POD_ID="$(echo "${POD_JSON}" | jq -r '.id')"
POD_IP="$(echo "${POD_JSON}" | jq -r '.ssh.ip')"
POD_PORT="$(echo "${POD_JSON}" | jq -r '.ssh.port')"

# マウントパスが想定通りにならないケースが過去にあったため(コンテナのエフェメラルディスクに
# 書き込んでしまい、pod終了時に消える)、Network Volumeとして実際にマウントされているか確認する
MOUNT_CHECK="$(ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no root@"${POD_IP}" -p "${POD_PORT}" "df -h /workspace")"
if ! echo "${MOUNT_CHECK}" | grep -q "runpod.net"; then
  echo "Error: /workspace does not look like a Network Volume mount (pod_id=${POD_ID}):" >&2
  echo "${MOUNT_CHECK}" >&2
  exit 1
fi

echo "pod_id=${POD_ID}"
echo "ip=${POD_IP}"
echo "port=${POD_PORT}"

# このワークロードはGPUよりシングルスレッドCPU性能がボトルネックになりやすいことが分かっている
# (RTX 4090・128vCPU・低クロックのAMD EPYCホストで1エポックが約68秒、RTX 4090・24コアのAMD
# Threadripperホストでは約26秒)。コード転送前にシステムのPython(venv不要)で軽い計算ベンチマークを
# 走らせ、遅いホストに当たったかを早期に判定する。目安: このループが1秒未満なら当たり(実測0.91秒で
# 26秒/エポック)、10秒を超えるようならハズレ(実測13.10秒で68秒/エポック)。ハズレの場合はterminateして
# 作り直すか、在庫の多い別のデータセンターを試す(Network Volumeはデータセンター固定のため、切り替える
# 場合は新しいVolumeの作成が必要)
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no root@"${POD_IP}" -p "${POD_PORT}" \
  "nproc; lscpu | grep -iE 'model name|mhz'; python3 -c 'import time; s = time.time(); x = 0
for i in range(20_000_000):
    x += i
print(f\"single-thread loop: {time.time() - s:.2f}s\")'"
