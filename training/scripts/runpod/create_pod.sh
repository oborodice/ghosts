#!/usr/bin/env bash
set -euo pipefail

# RunPodのpodを1つ作り、SSHで入れることと、ホストのCPUの速さを確かめる。
# Network Volumeは指定したときだけ使う。Network Volumeはデータセンターに固定されるので、そのデータセンターのGPUに空きが
# ないとpodを作れない。指定しなければpod自身のディスク(podを消すと中身も消える)を使い、空きのあるデータセンターを
# RunPodに選ばせる(その場合、結果はpodを消す前に download_results.sh で必ず落とす)。
# --gpu を複数指定すると、空きがないときに次のGPUを順に試す。課金が発生するので、1つずつ作り、作れなかったときは
# podの一覧(runpodctl pod list)に同じ名前のpodがないことを確かめてから次を試し、1つ作れたらそこで止める
SSH_KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
TEMPLATE_ID="runpod-torch-v280"
DEFAULT_GPU_ID="NVIDIA GeForce RTX 4090"
CONTAINER_DISK_GB=40

usage() {
  echo "Usage: $0 [--volume <network-volume-id>] [--gpu <gpu-id>]... [--name <pod-name>]" >&2
  echo "  --volume: attach a Network Volume at /workspace (the pod is created in the volume's datacenter)." >&2
  echo "    Without it the pod uses its own ${CONTAINER_DISK_GB}GB disk, which is deleted with the pod." >&2
  echo "  --gpu: defaults to '${DEFAULT_GPU_ID}'. Give it several times to try each in order until one has stock," >&2
  echo "    e.g. --gpu \"NVIDIA GeForce RTX 4090\" --gpu \"NVIDIA RTX PRO 6000 Blackwell Server Edition\" --gpu \"NVIDIA L40S\"" >&2
  echo "    (the GAN is small, so the GPU type changes speed but not results; see 'runpodctl gpu list' for ids)." >&2
  exit 1
}

VOLUME_ID=""
GPU_IDS=()
POD_NAME="ghosts-$(date +%Y%m%d%H%M%S)"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --volume) [ "$#" -ge 2 ] || usage; VOLUME_ID="$2"; shift 2 ;;
    --gpu) [ "$#" -ge 2 ] || usage; GPU_IDS+=("$2"); shift 2 ;;
    --name) [ "$#" -ge 2 ] || usage; POD_NAME="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ "${#GPU_IDS[@]}" -gt 0 ] || GPU_IDS=("${DEFAULT_GPU_ID}")

STORAGE_ARGS=(--container-disk-in-gb "${CONTAINER_DISK_GB}")
if [ -n "${VOLUME_ID}" ]; then
  DATA_CENTER_ID="$(runpodctl network-volume list | jq -r --arg id "${VOLUME_ID}" '.[] | select(.id == $id) | .dataCenterId')"
  if [ -z "${DATA_CENTER_ID}" ]; then
    echo "Network volume ${VOLUME_ID} not found" >&2
    exit 1
  fi
  STORAGE_ARGS+=(--data-center-ids "${DATA_CENTER_ID}" --network-volume-id "${VOLUME_ID}" --volume-mount-path "/workspace")
  LOCATION="on volume ${VOLUME_ID} (datacenter ${DATA_CENTER_ID})"
else
  LOCATION="without a Network Volume"
fi

POD_ID=""
for GPU_ID in "${GPU_IDS[@]}"; do
  echo "Creating pod '${POD_NAME}' (${GPU_ID}) ${LOCATION} ..."
  POD_JSON="$(runpodctl pod create --template-id "${TEMPLATE_ID}" --gpu-id "${GPU_ID}" --cloud-type SECURE \
    "${STORAGE_ARGS[@]}" --ports "22/tcp" --name "${POD_NAME}" --wait 2>&1 || true)"
  # --wait の出力は、先頭にsshを待つ進み具合の行があり、そのあとにpodのJSONが続く。JSONの部分だけを読む
  # (全体をJSONとして読むと、作れていても読み取りに失敗し、作れなかったと誤って判定する)
  POD_JSON="$(echo "${POD_JSON}" | sed -n '/^{/,$p')"
  POD_ID="$(echo "${POD_JSON}" | jq -r '.id // empty' 2>/dev/null || true)"
  [ -n "${POD_ID}" ] && break
  echo "Pod was not created: $(echo "${POD_JSON}" | tail -1)" >&2
  # 作れなかったと読めても、実は作られていた場合に2つ目を作らないよう、同じ名前のpodがないことを確かめてから次を試す
  # (止まっているpodも含める。一覧を読めないときも、ないとはみなさずに止まる)
  SAME_NAME_PODS="$(runpodctl pod list --all --name "${POD_NAME}" | jq 'length')"
  if [ "${SAME_NAME_PODS}" != 0 ]; then
    echo "Error: a pod named '${POD_NAME}' exists although creation looked failed; check 'runpodctl pod list --all'" >&2
    exit 1
  fi
done
if [ -z "${POD_ID}" ]; then
  echo "No pod was created with any of: ${GPU_IDS[*]}" >&2
  exit 1
fi
POD_IP="$(echo "${POD_JSON}" | jq -r '.ssh.ip')"
POD_PORT="$(echo "${POD_JSON}" | jq -r '.ssh.port')"
SSH=(ssh -n -i "${SSH_KEY}" -o StrictHostKeyChecking=no -p "${POD_PORT}" "root@${POD_IP}")

if [ -n "${VOLUME_ID}" ]; then
  # マウントパスが想定と違うと、コンテナのエフェメラルディスクに書き込んでしまい、pod終了時に消えるので、
  # Network Volumeとして実際にマウントされているか確認する
  if ! "${SSH[@]}" "df -h /workspace" | grep -q "runpod.net"; then
    echo "Error: /workspace does not look like a Network Volume mount (pod_id=${POD_ID})" >&2
    exit 1
  fi
fi

echo "pod_id=${POD_ID}"
echo "ip=${POD_IP}"
echo "port=${POD_PORT}"

# 学習の速さは、GPUよりホストのCPUの1スレッドの速さで決まりやすい(小さな計算を1つずつGPUに投げる手間が大きいため)。
# 同じGPUの機種でもホストによって違うので、軽い計算で確かめる。目安: このループが1秒前後なら速いホスト、2秒を超えると遅め。
# nprocはホスト全体のコア数を返すだけで、このpodに割り当てられたCPU(cgroupの制限)とは違うので、cgroupの値を見る
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no -p "${POD_PORT}" "root@${POD_IP}" bash -s <<'EOF'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 - <<'PY'
import time
started = time.time()
x = 0
for i in range(20_000_000):
    x += i
print(f"single-thread loop: {time.time() - started:.2f}s")
PY
# ホストによって、cgroup v2(cpu.max)とv1(cpu.cfs_quota_us。制限なしは -1)のどちらかになる
echo -n "actual cgroup CPU budget: "
if [ -f /sys/fs/cgroup/cpu.max ]; then
  read -r quota period < /sys/fs/cgroup/cpu.max
elif [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
  quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
else
  quota=""
fi
if [ -z "${quota}" ]; then
  echo unknown
elif [ "${quota}" = max ] || [ "${quota}" = -1 ]; then
  echo unlimited
else
  awk -v q="${quota}" -v p="${period}" 'BEGIN { printf "%.1f vCPU\n", q / p }'
fi
EOF
