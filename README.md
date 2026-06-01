# Explainable Federated Multi-Agent Prototype

This prototype demonstrates the core architecture:

- Client-side multi-agent collaboration
- Local private memory and trace storage
- Autonomous learning through memory, prompt, and rule updates
- Trainable tool-side components, starting with a `ToolRouter`
- Flower-based aggregation of shared trainable components
- Explainability across planning, memory, tool use, training, and aggregation

## Quick Start

Run the local simulation without external dependencies:

```powershell
python .\demos\run_single_client.py
python .\demos\run_federated_simulation.py
python .\demos\run_slm_client.py
```

The SLM demo defaults to a dependency-free mock backend. To use a local SLM
through Ollama:

```powershell
ollama pull qwen2.5:1.5b
$env:SLM_BACKEND="ollama"
$env:OLLAMA_MODEL="qwen2.5:1.5b"
python .\demos\run_slm_client.py
```

The SLM is used for answer generation and an optional non-federated routing
baseline. The federated component remains `shared.tool_router.ToolRouter`.

Install Flower later when you want real federated runs:

```powershell
pip install flwr
flwr run .
```
