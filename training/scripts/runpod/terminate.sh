#!/usr/bin/env bash
set -euo pipefail

# Stopではなくdeleteする: Stopだけだと課金(特にストレージ)が完全には止まらない。
# Network Volume自体は次回以降の再利用のため削除しない(pod単位でのみ削除する)
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <pod-id>" >&2
  exit 1
fi

runpodctl pod delete "$1"
