#!/bin/sh
exec uvicorn main:app --host 0.0.0.0 --port 3011 --workers 1