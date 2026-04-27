#!/usr/bin/env bash
# Flip all published vindex repos from dataset → model type in-place.
# Safe to re-run: 404s are silently skipped (repo not yet created).

REPOS=(
  # Gemma (existing)
  chrishayuk/gemma-3-4b-it-vindex
  chrishayuk/gemma-3-4b-it-vindex-client
  chrishayuk/gemma-3-4b-it-vindex-attn
  chrishayuk/gemma-3-4b-it-vindex-embed
  chrishayuk/gemma-3-4b-it-vindex-server
  chrishayuk/gemma-3-4b-it-vindex-browse

  # Mistral 7B v0.1
  chrishayuk/mistral-7b-v0.1-vindex
  chrishayuk/mistral-7b-v0.1-vindex-client
  chrishayuk/mistral-7b-v0.1-vindex-attn
  chrishayuk/mistral-7b-v0.1-vindex-embed
  chrishayuk/mistral-7b-v0.1-vindex-server
  chrishayuk/mistral-7b-v0.1-vindex-browse

  # Mistral 7B Instruct v0.3
  chrishayuk/mistral-7b-instruct-v0.3-vindex
  chrishayuk/mistral-7b-instruct-v0.3-vindex-client
  chrishayuk/mistral-7b-instruct-v0.3-vindex-attn
  chrishayuk/mistral-7b-instruct-v0.3-vindex-embed
  chrishayuk/mistral-7b-instruct-v0.3-vindex-server
  chrishayuk/mistral-7b-instruct-v0.3-vindex-browse

  # Llama 2 7B
  chrishayuk/llama-2-7b-vindex
  chrishayuk/llama-2-7b-vindex-client
  chrishayuk/llama-2-7b-vindex-attn
  chrishayuk/llama-2-7b-vindex-embed
  chrishayuk/llama-2-7b-vindex-server
  chrishayuk/llama-2-7b-vindex-browse

  # Llama 2 7B Chat
  chrishayuk/llama-2-7b-chat-vindex
  chrishayuk/llama-2-7b-chat-vindex-client
  chrishayuk/llama-2-7b-chat-vindex-attn
  chrishayuk/llama-2-7b-chat-vindex-embed
  chrishayuk/llama-2-7b-chat-vindex-server
  chrishayuk/llama-2-7b-chat-vindex-browse

  # Llama 3 8B
  chrishayuk/llama-3-8b-vindex
  chrishayuk/llama-3-8b-vindex-client
  chrishayuk/llama-3-8b-vindex-attn
  chrishayuk/llama-3-8b-vindex-embed
  chrishayuk/llama-3-8b-vindex-server
  chrishayuk/llama-3-8b-vindex-browse

  # Llama 3 8B Instruct
  chrishayuk/llama-3-8b-instruct-vindex
  chrishayuk/llama-3-8b-instruct-vindex-client
  chrishayuk/llama-3-8b-instruct-vindex-attn
  chrishayuk/llama-3-8b-instruct-vindex-embed
  chrishayuk/llama-3-8b-instruct-vindex-server
  chrishayuk/llama-3-8b-instruct-vindex-browse

  # Llama 3.2 1B
  chrishayuk/llama-3.2-1b-vindex
  chrishayuk/llama-3.2-1b-vindex-client
  chrishayuk/llama-3.2-1b-vindex-attn
  chrishayuk/llama-3.2-1b-vindex-embed
  chrishayuk/llama-3.2-1b-vindex-server
  chrishayuk/llama-3.2-1b-vindex-browse

  # Llama 3.2 1B Instruct
  chrishayuk/llama-3.2-1b-instruct-vindex
  chrishayuk/llama-3.2-1b-instruct-vindex-client
  chrishayuk/llama-3.2-1b-instruct-vindex-attn
  chrishayuk/llama-3.2-1b-instruct-vindex-embed
  chrishayuk/llama-3.2-1b-instruct-vindex-server
  chrishayuk/llama-3.2-1b-instruct-vindex-browse

  # Llama 3.2 3B
  chrishayuk/llama-3.2-3b-vindex
  chrishayuk/llama-3.2-3b-vindex-client
  chrishayuk/llama-3.2-3b-vindex-attn
  chrishayuk/llama-3.2-3b-vindex-embed
  chrishayuk/llama-3.2-3b-vindex-server
  chrishayuk/llama-3.2-3b-vindex-browse

  # Llama 3.2 3B Instruct
  chrishayuk/llama-3.2-3b-instruct-vindex
  chrishayuk/llama-3.2-3b-instruct-vindex-client
  chrishayuk/llama-3.2-3b-instruct-vindex-attn
  chrishayuk/llama-3.2-3b-instruct-vindex-embed
  chrishayuk/llama-3.2-3b-instruct-vindex-server
  chrishayuk/llama-3.2-3b-instruct-vindex-browse
)

for repo in "${REPOS[@]}"; do
  echo -n "  $repo → model ... "
  result=$(hf repos settings "$repo" --repo-type model --type dataset 2>&1)
  if echo "$result" | grep -qi "not found\|404\|Repository Not Found"; then
    echo "skipped (not yet created)"
  else
    echo "done"
  fi
done

echo "Migration complete."
