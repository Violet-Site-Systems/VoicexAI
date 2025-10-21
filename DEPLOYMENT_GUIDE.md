# EPPN Deployment Guide

This guide covers deploying the Ethical Policy Pipeline Network (EPPN) across multiple platforms including GitHub, Vercel, Cudos, and AgentVerse.

## Table of Contents

1. [GitHub Setup](#github-setup)
2. [Frontend Hosting on Vercel](#frontend-hosting-on-vercel)
3. [Cudos Blockchain Integration](#cudos-blockchain-integration)
4. [AgentVerse Registration](#agentverse-registration)
5. [Local Development](#local-development)
6. [Production Deployment](#production-deployment)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)

## GitHub Setup

### 1. Create GitHub Repository

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: EPPN with urban planning ethics framework"

# Create GitHub repository and push
gh repo create eppn-uagents --public --description "Ethical Policy Pipeline Network for Urban Planning and Resource Allocation"
git remote add origin https://github.com/yourusername/eppn-uagents.git
git branch -M main
git push -u origin main
```

### 2. Repository Structure

```
eppn-uagents/
├── agents/                    # uAgent implementations
├── cognitive_core/           # Cognitive reasoning system
├── frontend/                 # Dashboard and UI
├── deployment/               # Deployment configurations
├── docs/                     # Documentation
├── tests/                    # Test suites
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Local development
└── README.md                 # Project documentation
```

### 3. GitHub Actions CI/CD

Create `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest tests/ --cov=cognitive_core --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  deploy-frontend:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Deploy to Vercel
      uses: amondnet/vercel-action@v20
      with:
        vercel-token: ${{ secrets.VERCEL_TOKEN }}
        vercel-org-id: ${{ secrets.ORG_ID }}
        vercel-project-id: ${{ secrets.PROJECT_ID }}
        working-directory: ./frontend
```

## Frontend Hosting on Vercel

### Why Vercel over Netlify?

Vercel is chosen for the frontend hosting because:
- **Better Next.js/React Support**: Native integration with modern frontend frameworks
- **Edge Functions**: Better performance for API routes
- **Automatic HTTPS**: Built-in SSL certificates
- **Git Integration**: Automatic deployments from GitHub
- **Better Developer Experience**: Superior CLI and dashboard

### 1. Vercel Setup

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy from frontend directory
cd frontend
vercel --prod
```

### 2. Environment Variables

Set these in Vercel dashboard:

```
COGNITIVE_CORE_URL=https://your-cognitive-core-api.com
REDIS_URL=redis://your-redis-instance:6379
CUDOS_RPC_URL=https://rpc.cudos.org
AGENTVERSE_API_URL=https://api.agentverse.ai
AGENTVERSE_API_KEY=sk_dfdb772ee76f4a519f2d81870831911bb99530554f754e36a29aad452bf729df
```

### Troubleshooting: Vercel `vercel.json` conflict (functions vs builds)

One of the most common deployment errors when hosting Python (FastAPI) or hybrid backends on Vercel is mixing the new `functions` configuration with the old `builds` configuration in `vercel.json`.

- The `functions` property is the newer configuration style used to declare Serverless/Edge functions and their runtimes (for example `python3.11`).
- The `builds` array is the older, Next.js 12-era style. Vercel does not allow both at the same time and will fail with:

```
The functions property cannot be used in conjunction with the builds property.
```

Why this happens:

- You may have a project template (or an existing `vercel.json`) that contains `builds`, then add a `functions` entry (for example to pin `python3.11`). Mixing them triggers the error.

How to fix it (recommended):

1. Decide which style you want to use. For most Python/FastAPI single-file deployments on Vercel it's simpler to use the `functions` style so you can explicitly pin the Python runtime.
2. Remove the entire `builds` array from your `vercel.json` if you add `functions`.
3. Ensure routes still point to your entry file (for example `frontend/app.py` or `app.py`).

Example: converting an old `builds` config to the new `functions` style

Old (builds) style (what you may currently have):

```json
{
  "version": 2,
  "name": "voicexai-frontend",
  "builds": [{
    "src": "frontend/app.py",
    "use": "@vercel/python"
  }],
  "routes": [{ "src": "/(.*)", "dest": "frontend/app.py" }],
  "env": { "PYTHON_VERSION": "3.9" }
}
```

New (functions) style — explicit runtime (recommended):

```json
{
  "version": 2,
  "name": "voicexai-frontend",
  "functions": {
    "frontend/app.py": { "runtime": "python3.11" }
  },
  "routes": [{ "src": "/(.*)", "dest": "frontend/app.py" }],
  "env": { "PYTHON_VERSION": "3.11" }
}
```

Notes and tips:

- If you prefer to keep `builds` (old style) then do not add any `functions` property. Instead, keep your `builds` and `routes` as-is and update `PYTHON_VERSION` only if needed.
- If you use the Vercel UI to create Functions or change runtimes it may automatically add a `functions` section; in that case remove `builds` from the file stored in your repo.
- Use `python3.11` where possible for FastAPI to benefit from newer stdlib/security fixes and better runtime support.
- If you try the `functions` style, Vercel expects a runtime identifier in a specific format (for example some runtimes use `now-php@1.0.0` or `python@1.0.0`), and using `python3.11` directly may result in the error:

```
Error: Function Runtimes must have a valid version, for example `now-php@1.0.0`.
```

Because of this, I reverted the repository `vercel.json` files to the older `builds` style with `@vercel/python` which is compatible with Vercel's current validation. If you'd like, I can re-introduce `functions` with a correct runtime identifier — tell me which runtime string to use and I'll update the files again.
- After making the change locally, run `vercel dev` (local dev) and `vercel --prod` (production) to test deployments.

Quick checklist to resolve the error:

1. Open any `vercel.json` files in your repo (root and `frontend/`).
2. If you see both `functions` and `builds`, pick one style. I recommend `functions` for Python backends and pin `python3.11`.
3. Commit the change and redeploy.

If you'd like, I can convert the project's `vercel.json` files to `functions` style for you and pin `python3.11` — tell me whether you want me to update both the root `vercel.json` and `frontend/vercel.json` or just one.

Update applied in this repository:

- The root `vercel.json` and `frontend/vercel.json` were converted to the `functions` style and `PYTHON_VERSION` was updated to `3.11` to pin the runtime.

Final recommended `vercel.json` (root) example:

```json
{
  "version": 2,
  "name": "voicexai-frontend",
  "functions": {
    "frontend/app.py": { "runtime": "python3.11" }
  },
  "routes": [{ "src": "/(.*)", "dest": "frontend/app.py" }],
  "env": { "PYTHON_VERSION": "3.11" }
}
```


### 3. Custom Domain (Optional)

```bash
# Add custom domain
vercel domains add yourdomain.com
vercel domains verify yourdomain.com
```

## Cudos Blockchain Integration

### 1. Cudos Network Setup

```bash
# Install Cudos CLI
curl -sSL https://get.cudos.org | bash

# Initialize wallet
cudosd keys add eppn-wallet

# Get wallet address
cudosd keys show eppn-wallet -a

# Fund wallet (testnet)
cudosd tx bank send <sender> <eppn-wallet-address> 1000000acudos
```

### 2. Deploy to Cudos Network

```bash
# Build Docker image for Cudos
docker build -f deployment/Dockerfile.cudos -t eppn-cudos:latest .

# Tag for Cudos registry
docker tag eppn-cudos:latest cudos-registry/eppn-cudos:latest

# Push to Cudos registry
docker push cudos-registry/eppn-cudos:latest

# Deploy using Cudos CLI
cudosd tx compute submit-job \
  --from eppn-wallet \
  --job-file deployment/cudos-job.json \
  --gas auto \
  --gas-adjustment 1.5
```

### 3. Cudos Job Configuration

Create `deployment/cudos-job.json`:

```json
{
  "job": {
    "id": "eppn-ethical-analysis",
    "name": "EPPN Ethical Analysis",
    "description": "Distributed ethical analysis of urban planning policies",
    "docker_image": "cudos-registry/eppn-cudos:latest",
    "resources": {
      "cpu": "2",
      "memory": "4Gi",
      "storage": "10Gi"
    },
    "replicas": 3,
    "timeout": 3600,
    "reward": "10acudos"
  }
}
```

### 4. Smart Contract Integration

```solidity
// contracts/EPPNGovernance.sol
pragma solidity ^0.8.0;

contract EPPNGovernance {
    struct PolicyProposal {
        string title;
        string description;
        uint256 votesFor;
        uint256 votesAgainst;
        bool executed;
        uint256 deadline;
    }
    
    mapping(uint256 => PolicyProposal) public proposals;
    uint256 public proposalCount;
    
    function createProposal(
        string memory title,
        string memory description
    ) public returns (uint256) {
        uint256 proposalId = proposalCount++;
        proposals[proposalId] = PolicyProposal({
            title: title,
            description: description,
            votesFor: 0,
            votesAgainst: 0,
            executed: false,
            deadline: block.timestamp + 7 days
        });
        return proposalId;
    }
    
    function vote(uint256 proposalId, bool support) public {
        require(block.timestamp < proposals[proposalId].deadline, "Voting closed");
        
        if (support) {
            proposals[proposalId].votesFor++;
        } else {
            proposals[proposalId].votesAgainst++;
        }
    }
}
```

## AgentVerse Registration

### 1. AgentVerse Account Setup

```bash
# Register on AgentVerse platform
# Visit: https://agentverse.ai/register

# Get API key from dashboard
# Set environment variable
export AGENTVERSE_API_KEY=sk_dfdb772ee76f4a519f2d81870831911bb99530554f754e36a29aad452bf729df
```

### 2. Register EPPN Agents

```python
# Register all EPPN agents
from agentverse_integration import AgentVerseIntegration

agentverse = AgentVerseIntegration(
    agentverse_api_url="https://api.agentverse.ai",
    api_key="sk_dfdb772ee76f4a519f2d81870831911bb99530554f754e36a29aad452bf729df"
)

# Register each agent
librarian_id = agentverse.register_librarian_agent()
interpreter_id = agentverse.register_interpreter_agent()
summarizer_id = agentverse.register_summarizer_agent()
ethical_analyst_id = agentverse.register_ethical_analyst_agent()
communicator_id = agentverse.register_communicator_agent()

print(f"Registered agents: {[librarian_id, interpreter_id, summarizer_id, ethical_analyst_id, communicator_id]}")
```

### 3. Agent Discovery and Collaboration

```python
# Discover other ethical analysis agents
ethical_agents = agentverse.find_ethical_analysis_agents()
urban_planning_agents = agentverse.find_urban_planning_agents()

# Initiate collaboration
collaboration_id = agentverse.initiate_collaboration(
    participating_agents=[ethical_analyst_id] + [agent.agent_id for agent in ethical_agents[:2]],
    collaboration_type="distributed_ethical_analysis",
    task_description="Analyze urban planning policy for ethical implications"
)
```

## Local Development

### 1. Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Install Docker and Docker Compose
# Install Node.js and npm
# Install Git
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.template .env

# Edit environment variables
nano .env
```

### 3. Start Development Environment

```bash
# Start all services
docker-compose -f deployment/docker-compose.yml up -d

# Start individual agents
python agents/librarian/main.py &
python agents/interpreter/main.py &
python agents/summarizer/main.py &
python agents/ethical_analyst/main.py &
python agents/communicator/main.py &

# Start cognitive core API
uvicorn cognitive_core.api.cognitive_api:app --reload --port 8001

# Start frontend
cd frontend
uvicorn app:app --reload --port 8000
```

### 4. Development URLs

- Frontend Dashboard: http://localhost:8000
- Cognitive Core API: http://localhost:8001
- API Documentation: http://localhost:8001/docs
- Grafana Monitoring: http://localhost:3000
- Prometheus Metrics: http://localhost:9090

## Production Deployment

### 1. Cloud Infrastructure

#### Option A: AWS Deployment

```bash
# Deploy using AWS CDK
cd deployment/aws
npm install
cdk bootstrap
cdk deploy EPPNStack --require-approval never
```

#### Option B: Google Cloud Deployment

```bash
# Deploy using Google Cloud Run
gcloud run deploy eppn-frontend \
  --source frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

gcloud run deploy eppn-cognitive-core \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Option C: Azure Deployment

```bash
# Deploy using Azure Container Instances
az container create \
  --resource-group eppn-rg \
  --name eppn-frontend \
  --image your-registry/eppn-frontend:latest \
  --dns-name-label eppn-frontend \
  --ports 8000
```

### 2. Database Setup

```bash
# PostgreSQL setup
createdb eppn_production
psql eppn_production < deployment/init.sql

# Redis setup
redis-server --port 6379 --daemonize yes
```

### 3. SSL Certificate

```bash
# Using Let's Encrypt
certbot --nginx -d yourdomain.com -d api.yourdomain.com
```

## Monitoring and Maintenance

### 1. Health Checks

```python
# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "cognitive_core": check_cognitive_core_health(),
            "database": check_database_health(),
            "redis": check_redis_health(),
            "cudos": check_cudos_health(),
            "agentverse": check_agentverse_health()
        }
    }
```

### 2. Logging Configuration

```python
# Logging setup
import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('logs/eppn.log', maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)
```

### 3. Backup Strategy

```bash
# Database backup
pg_dump eppn_production > backups/eppn_$(date +%Y%m%d_%H%M%S).sql

# Redis backup
redis-cli BGSAVE

# File system backup
tar -czf backups/eppn_data_$(date +%Y%m%d_%H%M%S).tar.gz data/
```

### 4. Performance Monitoring

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('eppn_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('eppn_request_duration_seconds', 'Request duration')

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(time.time() - start_time)
    return response
```

## Security Considerations

### 1. API Security

```python
# JWT Authentication
from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
```

### 2. Environment Security

```bash
# Secure environment variables
export CUDOS_PRIVATE_KEY="$(openssl rand -base64 32)"
export JWT_SECRET_KEY="$(openssl rand -base64 32)"
export DATABASE_PASSWORD="$(openssl rand -base64 16)"
```

### 3. Network Security

```yaml
# nginx.conf security headers
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
```

## Troubleshooting

### Common Issues

1. **Agent Registration Fails**
   - Check AgentVerse API key
   - Verify network connectivity
   - Check agent capability definitions

2. **Cudos Integration Issues**
   - Verify wallet funding
   - Check RPC endpoint connectivity
   - Validate smart contract addresses

3. **Frontend Deployment Issues**
   - Check Vercel environment variables
   - Verify build process
   - Check domain configuration

### Support Channels

- GitHub Issues: https://github.com/yourusername/eppn-uagents/issues
- Documentation: https://docs.eppn.ai
- Community Discord: https://discord.gg/eppn
- Email Support: support@eppn.ai

## Conclusion

This deployment guide provides comprehensive instructions for deploying EPPN across multiple platforms. The system is designed to be modular and scalable, allowing for flexible deployment strategies based on your specific requirements.

For additional support or questions, please refer to the documentation or contact the development team.
