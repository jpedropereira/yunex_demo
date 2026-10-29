#!/bin/sh
# Exit 0 if Docker can actually run GPU containers, 1 otherwise.
#
# Without this check, `up-gpu` on a host missing nvidia-container-toolkit dies
# with the opaque daemon error:
#   could not select device driver "nvidia" with capabilities: [[gpu]]
# which says nothing about what to install.
set -eu

runtimes=$(docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null || echo "")

case " ${runtimes} " in
    *" nvidia "*) exit 0 ;;
esac

echo "------------------------------------------------------------------------" >&2
echo "GPU not available to Docker: the 'nvidia' runtime is not registered." >&2
echo "Docker sees only these runtimes: ${runtimes:-(none reported)}" >&2
cat >&2 <<'MSG'

On Linux this needs a one-time host install of nvidia-container-toolkit --
see the "Linux + NVIDIA" section of the README. On macOS there is no GPU
path from inside Docker at all; use `make up-external` instead.

Falling back to CPU so the stack still comes up.
------------------------------------------------------------------------
MSG
exit 1
