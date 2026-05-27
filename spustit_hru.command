#!/bin/bash
cd "$(dirname "$0")"

# Kill any existing server on port 8080
lsof -ti :8080 | xargs kill -9 2>/dev/null

# Start the proxy server
python3 serve.py &
SERVER_PID=$!

# Wait for server to start
sleep 1

# Open in browser
open http://localhost:8080/

echo "Server běží (PID $SERVER_PID). Zavři toto okno pro ukončení."
wait $SERVER_PID
