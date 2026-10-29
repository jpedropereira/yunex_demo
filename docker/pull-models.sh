#!/bin/sh
# Pull the models the demo needs, through the running Ollama server so they
# land in the shared volume. `ollama pull` is a no-op when already current,
# so this is safe to run on every `up`.
set -eu

: "${OLLAMA_HOST:?OLLAMA_HOST must be set}"
: "${MODEL_MAIN:?MODEL_MAIN must be set}"
: "${MODEL_EMBED:?MODEL_EMBED must be set}"

echo "Provisioning models via ${OLLAMA_HOST}"

for model in "$MODEL_MAIN" "$MODEL_EMBED"; do
    echo "--> pulling ${model}"
    ollama pull "$model"
done

echo "Models ready:"
ollama list
