#!/usr/bin/env bash
#
# ASR Server management script
# Usage: ./asr.sh {start|stop|restart|status}
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PID_DIR="$SCRIPT_DIR/.pids"
LOG_DIR="$SCRIPT_DIR/logs"

API_PID_FILE="$PID_DIR/api.pid"
UI_PID_FILE="$PID_DIR/ui.pid"
API_LOG="$LOG_DIR/api.log"
UI_LOG="$LOG_DIR/ui.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

ensure_dirs() {
    mkdir -p "$PID_DIR" "$LOG_DIR"
}

check_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        log_error "Virtual environment not found at $VENV_DIR"
        log_error "Create it with: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
}

is_running() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

start_api() {
    if is_running "$API_PID_FILE"; then
        log_warn "API server already running (PID: $(cat "$API_PID_FILE"))"
        return 0
    fi

    log_info "Starting API server..."
    source "$VENV_DIR/bin/activate"

    cd "$SCRIPT_DIR"
    nohup python src/server.py > "$API_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$API_PID_FILE"

    # Wait a moment and verify it started
    sleep 2
    if is_running "$API_PID_FILE"; then
        log_info "API server started (PID: $pid)"
        log_info "Logs: $API_LOG"
    else
        log_error "API server failed to start. Check logs: $API_LOG"
        rm -f "$API_PID_FILE"
        return 1
    fi
}

start_ui() {
    if is_running "$UI_PID_FILE"; then
        log_warn "Gradio UI already running (PID: $(cat "$UI_PID_FILE"))"
        return 0
    fi

    log_info "Starting Gradio UI on http://localhost:7860 ..."
    source "$VENV_DIR/bin/activate"

    cd "$SCRIPT_DIR"
    nohup python src/gradio_app.py > "$UI_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$UI_PID_FILE"

    # Wait a moment and verify it started
    sleep 2
    if is_running "$UI_PID_FILE"; then
        log_info "Gradio UI started (PID: $pid)"
        log_info "Logs: $UI_LOG"
    else
        log_error "Gradio UI failed to start. Check logs: $UI_LOG"
        rm -f "$UI_PID_FILE"
        return 1
    fi
}

stop_process() {
    local name="$1"
    local pid_file="$2"

    if [[ ! -f "$pid_file" ]]; then
        log_warn "$name not running (no PID file)"
        return 0
    fi

    local pid=$(cat "$pid_file")

    if ! kill -0 "$pid" 2>/dev/null; then
        log_warn "$name not running (stale PID file)"
        rm -f "$pid_file"
        return 0
    fi

    log_info "Stopping $name (PID: $pid)..."
    kill -TERM "$pid" 2>/dev/null || true

    # Wait for graceful shutdown
    local count=0
    while kill -0 "$pid" 2>/dev/null && [[ $count -lt 10 ]]; do
        sleep 1
        ((count++))
    done

    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        log_warn "$name didn't stop gracefully, forcing..."
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$pid_file"
    log_info "$name stopped"
}

cmd_start() {
    ensure_dirs
    check_venv

    log_info "Starting ASR Server system..."
    echo ""

    start_api
    start_ui

    echo ""
    log_info "ASR Server system started"
    log_info "  API:      http://localhost:8005 (see config.yaml)"
    log_info "  API Docs: http://localhost:8005/docs"
    log_info "  Gradio:   http://localhost:7860"
}

cmd_stop() {
    log_info "Stopping ASR Server system..."
    echo ""

    stop_process "Gradio UI" "$UI_PID_FILE"
    stop_process "API server" "$API_PID_FILE"

    echo ""
    log_info "ASR Server system stopped"
}

cmd_restart() {
    cmd_stop
    echo ""
    sleep 1
    cmd_start
}

cmd_status() {
    echo "ASR Server Status"
    echo "================="

    if is_running "$API_PID_FILE"; then
        echo -e "API server:  ${GREEN}running${NC} (PID: $(cat "$API_PID_FILE"))"
    else
        echo -e "API server:  ${RED}stopped${NC}"
    fi

    if is_running "$UI_PID_FILE"; then
        echo -e "Gradio UI:   ${GREEN}running${NC} (PID: $(cat "$UI_PID_FILE"))"
    else
        echo -e "Gradio UI:   ${RED}stopped${NC}"
    fi
}

cmd_logs() {
    local target="${1:-all}"

    case "$target" in
        api)
            if [[ -f "$API_LOG" ]]; then
                tail -f "$API_LOG"
            else
                log_error "No API log file found"
            fi
            ;;
        ui)
            if [[ -f "$UI_LOG" ]]; then
                tail -f "$UI_LOG"
            else
                log_error "No UI log file found"
            fi
            ;;
        all|*)
            log_info "Tailing both logs (Ctrl+C to stop)..."
            tail -f "$API_LOG" "$UI_LOG" 2>/dev/null || log_error "No log files found"
            ;;
    esac
}

usage() {
    echo "ASR Server Management"
    echo ""
    echo "Usage: $0 {command}"
    echo ""
    echo "Commands:"
    echo "  start     Start API server and Gradio UI"
    echo "  stop      Stop all services"
    echo "  restart   Restart all services"
    echo "  status    Show service status"
    echo "  logs      Tail all logs (or: logs api, logs ui)"
    echo ""
}

# Main
case "${1:-}" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    status)
        cmd_status
        ;;
    logs)
        cmd_logs "${2:-all}"
        ;;
    *)
        usage
        exit 1
        ;;
esac
