# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About This Project

Open Deep Research is an open-source deep research agent that automates comprehensive research and report generation. It supports multiple model providers, search APIs, and MCP (Model Context Protocol) servers for extensible research capabilities.

## Core Architecture

The main research engine is in `src/open_deep_research/deep_researcher.py` which implements a multi-agent supervisor-researcher architecture:

- **Supervisor Agent**: Manages research process and assembles final reports
- **Researcher Agents**: Work in parallel to research different aspects of topics
- **State Management**: Uses LangGraph for state coordination (`src/open_deep_research/state.py`)
- **Configuration**: Centralized config in `src/open_deep_research/configuration.py`

### Key Components

- `src/open_deep_research/deep_researcher.py`: Main agent implementation and graph definition
- `src/open_deep_research/configuration.py`: Configuration classes with support for multiple search APIs and models
- `src/open_deep_research/state.py`: State management classes for agent coordination
- `src/open_deep_research/prompts.py`: System prompts for different agent roles
- `src/open_deep_research/utils.py`: Utility functions for model initialization, token limits, and tool handling
- `src/legacy/`: Contains alternative implementations (workflow-based and multi-agent approaches)

### LangGraph Integration

The project is configured as a LangGraph application with:
- Graph definition in `langgraph.json`
- Main graph at `deep_researcher` function
- Authentication support via `src/security/auth.py`

## Common Development Commands

### Environment Setup
```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r pyproject.toml

# Install with development dependencies
uv pip install -e ".[dev]"
```

### Development Server
```bash
# Start LangGraph development server with Studio UI
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking

# Access points:
# - API: http://127.0.0.1:2024
# - Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
# - API Docs: http://127.0.0.1:2024/docs
```

### Testing and Evaluation
```bash
# Run comprehensive evaluation
python tests/run_evaluate.py

# Test specific implementation
python tests/run_test.py --agent multi_agent
python tests/run_test.py --agent graph

# Run pytest
pytest
```

### Code Quality
```bash
# Run linting (configured in pyproject.toml)
ruff check

# Run type checking
mypy src/
```

## Configuration

The system uses a flexible configuration approach:
- Environment variables (`.env` file)
- Direct configuration via `Configuration` class
- Web UI configuration (when deployed)

### Supported Search APIs
- **Tavily**: Works with all models (default)
- **OpenAI Native Web Search**: Requires OpenAI models
- **Anthropic Native Web Search**: Requires Anthropic models
- **None**: No web search capability

### Model Requirements
All models must support:
1. Structured outputs
2. Tool calling functionality
3. Compatibility with selected search API

### MCP Server Support
The system supports both local and remote MCP servers for extending research capabilities. MCP configurations are defined in the `MCPConfig` class.

## Development Notes

- The codebase follows Google docstring conventions
- Uses Ruff for linting with specific rules defined in `pyproject.toml`
- Token limit management is handled automatically via utility functions
- Authentication and security components are in `src/security/`
- Legacy implementations in `src/legacy/` provide alternative approaches for reference

## Testing Strategy

The project includes comprehensive evaluation systems:
- 9-dimensional quality assessment for generated reports
- Batch evaluation capabilities
- Specialized evaluators in `tests/evaluators.py`
- Support for testing different agent implementations