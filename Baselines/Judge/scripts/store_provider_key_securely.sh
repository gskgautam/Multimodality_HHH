#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  openai) file_name=openai_api_key; label=OpenAI ;;
  google|gemini) file_name=google_gemini_api_key; label=Google-Gemini ;;
  anthropic) file_name=anthropic_api_key; label=Anthropic ;;
  *) echo 'Usage: bash scripts/store_provider_key_securely.sh {openai|google|anthropic}' >&2; exit 2 ;;
esac
config_dir="$HOME/.config/multimodality"
mkdir -p "$config_dir"
chmod 700 "$config_dir"
key_file="$config_dir/$file_name"
read -r -s -p "Paste $label API key (input hidden): " api_key
printf '\n'
if [[ -z "$api_key" ]]; then echo 'No key entered; nothing saved.' >&2; exit 1; fi
printf '%s\n' "$api_key" > "$key_file"
chmod 600 "$key_file"
unset api_key
printf 'Saved %s key (permission 600) at %s\n' "$label" "$key_file"
