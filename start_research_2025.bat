@echo off
echo Starting Deep Research Agent on port 2025...
cd /d "D:\Workspace\open_deep_research"
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --port 2025 --allow-blocking