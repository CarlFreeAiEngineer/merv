#!/bin/sh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
case "$(uname -s)" in
    Darwin) UV="$DIR/bin/uv.mac" ;;
    Linux)  UV="$DIR/bin/uv.linux" ;;
    *)      echo "Use run.bat on Windows" >&2; exit 1 ;;
esac
chmod +x "$UV" 2>/dev/null || true
exec "$UV" run "$DIR/serve.py" "$@"
