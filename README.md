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
python .\demos\run_server_task_dispatch.py
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

## Real Flower Simulation

The dependency-free demo in `demos/run_federated_simulation.py` manually
simulates FedAvg. To run the actual Flower app:

```powershell
conda activate federated-agent
pip install ".[flower]"
$env:PYTHONIOENCODING="utf-8"
$env:PATH="C:\Users\Peter\miniconda3\envs\federated-agent\Scripts;" + $env:PATH
flwr run . --federation-config "num-supernodes=3" --stream
```

Flower uses:

```text
client.federation.flower_client:app
server.flower_server:app
local-simulation with 3 supernodes
```

## Server Orchestration

The server-side runtime is separate from Flower:

```text
TaskGateway
  receives external tasks

ServerMemoryAgent
  stores client states, activity histories, and specialities

ClientSelector
  arranges and forwards tasks to selected clients
```

The server memory stores orchestration metadata only. Raw private client data
and local client memories remain inside each client.

## Client Agents

Each client currently runs:

```text
MemoryAgent
  retrieves and updates private local memory

PlanningAgent
  selects task type and first tool through the trainable ToolRouter

ToolAgent
  executes the selected tool

SLM
  generates the final response from the selected tool result
```

The MemoryAgent, ToolAgent, and SLM are static local execution components. The
collaboratively trained component is the PlanningAgent's `ToolRouter`.
