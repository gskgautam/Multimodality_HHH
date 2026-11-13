#!/usr/bin/env bash
set -u
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
mkdir -p logs
run_one() {
 provider="$1"; slug="$2"; model="$3"; port="$4"
 export PROVIDER="$provider" SLUG="$slug" MODEL="$model" PORT="$port"
 python3 scripts/provider_compat_proxy.py >"logs/$slug-proxy.log" 2>&1 &
 proxy_pid=$!
 sleep 2
 echo "START $slug $(date -Is)" | tee -a logs/closed_models_status.log
 python3 scripts/run_ch3ef_full1416_model.py --slug "$slug" --model "$model" --api "http://127.0.0.1:$port/api/chat" >>"logs/$slug-generation.log" 2>&1
 result=$?
 kill "$proxy_pid" 2>/dev/null || true
 echo "END $slug exit=$result $(date -Is)" | tee -a logs/closed_models_status.log
}
run_one openai gpt-6.1-terra gpt-6.1-terra 11436
run_one google gemini-4-argon gemini-4-argon 11436
run_one anthropic claude-sonnet-5-5 claude-sonnet-5-5 11436
