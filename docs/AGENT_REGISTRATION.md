Agent registration & endpoint guide
=================================

Location: `docs/AGENT_REGISTRATION.md` in the repository root.

Purpose
-------

This document explains the difference between an agent identity (seed/name) and a reachable endpoint, and provides an actionable checklist you can follow to register uAgents in AgentVerse (or any registry) and make them callable.

When to use identity vs endpoint
--------------------------------

- Identity (seed/name): used for discovery, trust, routing within a shared message bus or registry. Examples in this repo: `Agent(name="librarian", seed="eppn-librarian")`.
- Endpoint (HTTP/webhook): used when AgentVerse (or any external service) must call your agent over the network. Required when the caller cannot publish into your message bus directly.

Actionable checklist (step-by-step)
-----------------------------------

1. Pick invocation mode
   - Brokered (preferred inside shared infra): use the same uagents/ASI registry or message bus for AgentVerse and your agents. No public HTTP endpoint needed.
   - HTTP/webhook (when AgentVerse is remote and must reach you): provide a stable URL and authentication.

2. Ensure stable identities
   - Verify each agent's Python entrypoint has a stable `name` and `seed` (e.g., `agents/librarian/main.py` sets name and seed).
   - Keep those values consistent; they are your canonical identity.

3. Decide reachability and register endpoints
   - If using HTTP: implement a small gateway or sidecar that accepts AgentVerse requests and forwards them to your local agent runtime.
   - If using brokered messages: ensure AgentVerse has access to the shared registry/bus.

4. Add endpoint to registration metadata
   - In `agentverse_integration.register_agent(...)` pass an `endpoint` URL and `local_path` (we added `code_location` earlier).
   - Example contact_info structure stored in `AgentProfile.contact_info` contains `endpoint` and `code_path`.

5. Add mapping in the integration manager
   - Add `agent_endpoints` to your integration manager config and map agent role -> endpoint.
   - Example `config` snippet (YAML):

```yaml
agent_endpoints:
  librarian: "https://your-host.example.com/agents/librarian/message"
  interpreter: "https://your-host.example.com/agents/interpreter/message"
  summarizer: "https://your-host.example.com/agents/summarizer/message"
```

6. Secure your endpoints
   - Use API keys or mTLS. If you host behind a gateway (Nginx/API Gateway) put auth and rate limiting there.
   - Add a health check endpoint (e.g., `GET /health`) and a readiness probe.

7. Test registration and invocation
   - Register agents on AgentVerse (or call your registry) including the `endpoint` field.
   - From AgentVerse or a test client, POST a message to the endpoint and confirm the local gateway forwards it into the agent runtime (and the agent replies or produces expected side effects).
   - Use `curl` or Postman to verify request/response flows.

8. Observe and iterate
   - Watch logs (`/app/logs` or the logging location you use) on both sides (gateway and agent).
   - Add telemetry and Prometheus metrics where helpful.

Minimal HTTP gateway example (FastAPI)
--------------------------------------

This gateway sits beside your agents and forwards HTTP POSTs into the uagents runtime. It is intentionally minimal — adapt to your runtime API.

```python
from fastapi import FastAPI, Request
import asyncio

app = FastAPI()

# Pseudo: implement a method that enqueues a message into the uagents runtime
async def forward_to_agent(agent_name: str, payload: dict):
    # Example: using a runtime API or local transport to send the message
    # await agent_runtime.send_to_agent(agent_name, payload)
    return {"status": "forwarded"}

@app.post("/agents/{agent_name}/message")
async def receive(agent_name: str, req: Request):
    body = await req.json()
    result = await forward_to_agent(agent_name, body)
    return result
```

Where the registration data will be stored in this repository
-------------------------------------------------------------

- Local reference and instructions: this file `docs/AGENT_REGISTRATION.md`.
- Programmatic storage: `AgentProfile.contact_info` includes `endpoint` and `code_path` and is created/returned by `agentverse_integration.register_agent(...)`.
- Integration wiring: `integration_manager.py` will read a `agent_endpoints` section from the `config` you pass to `EPPNIntegrationManager` and can pass them into registration calls.

Search & reuse on ASI1.ai / AgentVerse
--------------------------------------

- Use ASI1.ai or AgentVerse UI/registry to search by capability keywords (e.g., `pdf_retrieval`, `document_summarization`) or agent name / domain.
- If you find an existing agent that matches, record its `agent_id` and `endpoint` in your `agent_endpoints` mapping or update `AgentVerseIntegration.discovered_agents`.

Common pitfalls & gotchas
------------------------

- NAT / firewall: if your agent is local, either expose a public endpoint (securely) or use a reverse-tunnel (ngrok, cloud tunnel) or deploy a gateway in a public environment.
- Schema mismatches: make sure payloads match `schemas/messages.py` (doc ids, field names).
- Auth & security: never expose endpoints without auth.

Next-actions (suggested)
------------------------

- Decide invocation mode (Brokered vs HTTP). If unsure, start with brokered and use a dev registry.
- If HTTP, pick a gateway pattern (FastAPI sidecar or API gateway) and add endpoint URLs to `config.agent_endpoints`.
- Run a test registration flow with one agent (e.g., `librarian`) to validate the end-to-end path.

Reviewer notes
--------------

When you open a PR to persist code changes around registration, include: how you tested (manual curl, logs), sample `config` values, and any security constraints.

---

If you want, I can also add a short checklist into `DEPLOYMENT_GUIDE.md` referencing this doc and include an example `config/dev.yaml`. I will not push or create a PR until you say go.
