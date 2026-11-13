#!/usr/bin/env bash
set -euo pipefail

key_dir="$HOME/.config/multimodality"
key_file="$key_dir/openai_api_key"
mkdir -p "$key_dir"
chmod 700 "$key_dir"

if [[ -e "$key_file" ]]; then
  printf 'A key file already exists at %s; refusing to overwrite it.\n' "$key_file" >&2
  exit 1
fi

read -r -s -p 'Paste the newly created OpenAI API key (input hidden): ' api_key
printf '\n'
if [[ "$api_key" != sk-* ]]; then
  unset api_key
  printf 'That does not look like an OpenAI API key; nothing was saved.\n' >&2
  exit 1
fi
umask 077
printf '%s\n' "$api_key" > "$key_file"
chmod 600 "$key_file"
unset api_key
printf 'Saved securely at %s (file permission 600).\n' "$key_file"
