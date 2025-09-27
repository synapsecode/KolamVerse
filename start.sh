#!/bin/bash
uvicorn main:app --host 0.0.0.0 --port 3011 > /app/gunicorn.log 2>&1 &
tail -f /dev/null