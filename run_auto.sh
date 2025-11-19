#!/bin/bash
# Auto-start Flask ML API on server
# This script runs Flask in background and keeps it running

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Log file
LOG_FILE="$SCRIPT_DIR/flask_auto.log"
PID_FILE="$SCRIPT_DIR/flask.pid"

# Function to check if Flask is already running
check_flask_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            return 0  # Running
        else
            rm -f "$PID_FILE"
            return 1  # Not running
        fi
    fi
    return 1  # Not running
}

# Function to stop Flask
stop_flask() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Stopping Flask (PID: $PID)..."
            kill $PID
            sleep 2
            if ps -p $PID > /dev/null 2>&1; then
                kill -9 $PID
            fi
        fi
        rm -f "$PID_FILE"
        echo "Flask stopped"
    else
        echo "Flask is not running"
    fi
}

# Function to start Flask
start_flask() {
    if check_flask_running; then
        echo "Flask is already running (PID: $(cat $PID_FILE))"
        return 1
    fi
    
    echo "Starting Flask ML API..."
    echo "Log file: $LOG_FILE"
    
    # Start Flask in background
    nohup python3 app.py >> "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    
    sleep 3
    
    # Check if it's still running
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Flask started successfully (PID: $PID)"
        echo "📋 Logs: tail -f $LOG_FILE"
        echo "🛑 Stop: ./run_auto.sh stop"
        return 0
    else
        echo "❌ Failed to start Flask"
        rm -f "$PID_FILE"
        echo "Check logs: tail -20 $LOG_FILE"
        return 1
    fi
}

# Function to restart Flask
restart_flask() {
    echo "Restarting Flask..."
    stop_flask
    sleep 2
    start_flask
}

# Function to show status
show_status() {
    if check_flask_running; then
        PID=$(cat "$PID_FILE")
        echo "✅ Flask is running (PID: $PID)"
        echo "📋 Logs: tail -f $LOG_FILE"
        
        # Test health endpoint
        echo ""
        echo "Testing health endpoint..."
        curl -s http://localhost:8000/health/ | head -5
    else
        echo "❌ Flask is not running"
        echo "Start with: ./run_auto.sh start"
    fi
}

# Main command handler
case "$1" in
    start)
        start_flask
        ;;
    stop)
        stop_flask
        ;;
    restart)
        restart_flask
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "Commands:"
        echo "  start   - Start Flask API in background"
        echo "  stop    - Stop Flask API"
        echo "  restart - Restart Flask API"
        echo "  status  - Show Flask status"
        exit 1
        ;;
esac

