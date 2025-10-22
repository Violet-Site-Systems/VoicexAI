# EPPN - Ethical Policy Pipeline Network

## Overview

The Ethical Policy Pipeline Network (EPPN) is a hybrid multi-agent system that combines ASI:uAgents distributed intelligence with OpenCog-inspired cognitive reasoning for autonomous policy document analysis and ethical evaluation.

## Architecture

### Multi-Agent System (uAgents)
- **Librarian uAgent**: Fetches PDFs from public government portals
- **Interpreter uAgent**: Extracts and structures content from PDFs
- **Summarizer uAgent**: Summarizes structured data using LLMs
- **Ethical Analyst uAgent**: Integrates with OpenCog AtomSpace for ethical reasoning
- **Communicator uAgent**: Interfaces with human ethics partner via dashboard

### Cognitive Core (OpenCog Integration)
- **AtomSpace**: Concept graph representations of policy data
- **Reasoning Engine**: Pattern mining, probabilistic logic reasoning, and ECAN-style attention control
- **PLN Reasoner**: Probabilistic Logic Networks for ethical inference
- **Concept Mapping**: Symbolic representation of policy concepts and relationships

## Key Features

1. **Autonomous Policy Retrieval**: Automated discovery and fetching of policy documents
2. **Cognitive Understanding**: Symbolic representation and reasoning about policy content
3. **Ethical Analysis**: Advanced reasoning about ethical implications and fairness patterns
4. **Human-AI Collaboration**: Interactive dashboard for human oversight and feedback
5. **Distributed Intelligence**: Scalable multi-agent architecture on ASI:cloud

## Project Structure

```
eppn/
├── agents/                    # uAgent implementations
│   ├── librarian/            # PDF retrieval agent
│   ├── interpreter/          # Content extraction agent
│   ├── summarizer/           # LLM-based summarization agent
│   ├── ethical_analyst/      # OpenCog-integrated ethical reasoning
│   └── communicator/         # Human interface agent
├── cognitive_core/           # OpenCog-inspired cognitive system
│   ├── atomspace/           # Concept graph and atom storage
│   ├── reasoning/           # PLN reasoning and pattern mining
│   └── api/                 # Cognitive core APIs
├── schemas/                  # Inter-agent message schemas
├── config/                   # Configuration files
├── tests/                    # Test suites
└── docs/                     # Documentation
```

## Agent Message Flow

1. Librarian receives `CrawlRequest` and emits `PDFReady` for each URL.
2. Interpreter receives `PDFReady`, extracts text to `ParsedText` and emits back.
3. Summarizer receives `ParsedText`, emits `SummaryReady`.
4. Ethical Analyst receives `ParsedText`, writes to AtomSpace, runs cognitive analysis, emits `EthicsReport`.
5. Communicator receives `SummaryReady` and `EthicsReport` for human review.

## Quickstart (Local)

1. Install deps: `pip install -r requirements.txt`
2. Run agents in separate terminals:
   - `python agents/librarian/main.py`
   - `python agents/interpreter/main.py`
   - `python agents/summarizer/main.py`
   - `python agents/ethical_analyst/main.py`
   - `python agents/communicator/main.py`
3. Start the FastAPI backend:
   ```bash
   cd frontend
   uvicorn app:app --reload --port 8000
   ```

Static demo (Netlify):

1. Deploy the static site to Netlify or use `netlify dev`:

```bash
# From repository root
netlify deploy --dir=frontend/static --prod
```

Set BACKEND_URL to your local backend in the static UI.
3. Send a `CrawlRequest` to Librarian with a public PDF URL.

### CLI helper

- `python tools/cli_crawl.py https://treasury.go.ke/budget/SomeBudgetDoc.pdf`

### Social monitoring

- Start `social_monitor` to poll feeds: `python agents/social_monitor/main.py`

### Dashboard in Docker

```
cd frontend
docker build -t eppn-dashboard .
docker run -p 8000:8000 -v %cd%/../data:/app/data eppn-dashboard
```

## Deployment (ASI:cloud)

- Use `config/asi-config.yaml` for registry and agents. Then:
```
asi deploy --config config/asi-config.yaml
```

## Development Stack

- **Python 3.9+**
- **Core Libraries**: uagents, requests, pypdf2, transformers, openai
- **Cognitive Libraries**: networkx, pandas, numpy, spacy
- **Advanced Reasoning**: opencog-hyperon (optional)
- **Deployment**: ASI:cloud multi-agent registry

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Cognitive Core**:
   ```bash
   python -m cognitive_core.init
   ```

3. **Start Agent Development**:
   ```bash
   python -m agents.librarian.main
   ```

4. **Deploy to ASI:cloud**:
   ```bash
   asi deploy --config asi-config.yaml
   ```

## Cognitive Pipeline

1. **Document Discovery**: Librarian agent identifies and retrieves policy documents
2. **Content Extraction**: Interpreter agent parses and structures document content
3. **Semantic Understanding**: Content is mapped to AtomSpace concepts
4. **Summarization**: Summarizer agent creates human-readable summaries
5. **Ethical Reasoning**: Ethical Analyst applies PLN reasoning for ethical evaluation
6. **Human Review**: Communicator agent presents findings for human oversight

## Ethical Reasoning Capabilities

- **Contradiction Detection**: Identifies conflicting policy statements
- **Fairness Analysis**: Evaluates policy fairness across different groups
- **Ethical Red Flags**: Detects potentially problematic policy elements
- **Conceptual Mapping**: Maps policy concepts to ethical frameworks
- **Probabilistic Inference**: Uses PLN for uncertain ethical reasoning

## Development Roadmap

1. **Phase 1**: Core agent implementation and basic cognitive reasoning
2. **Phase 2**: Advanced PLN reasoning and ethical pattern recognition
3. **Phase 3**: Human-AI collaboration interface and feedback loops
4. **Phase 4**: Cloud deployment and scaling optimization
5. **Phase 5**: Advanced cognitive capabilities and autonomous learning

## Contributing

This project follows ethical AI development principles and requires careful consideration of bias, fairness, and transparency in all implementations.

## License

[License information to be added]
