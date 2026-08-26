# FinSightAI

> **AI-powered investment research assistant that transforms fragmented market data, financial news, and user-provided documents into structured, evidence-backed research.**

FinSightAI is an AI-powered investment research platform designed to simplify financial research by combining **market data, financial news, and user-provided documents** into a single research workflow.

Instead of manually collecting information from multiple sources, FinSightAI uses a **multi-agent research pipeline** to independently analyze different information sources and then synthesize the findings into a concise, structured report with **source attribution and evidence validation**.

> ⚠️ **Disclaimer:** FinSightAI is a research assistant and does **not** provide financial advice or investment recommendations.

---

## ✨ Key Features

* 🔎 **Multi-source research** — Combines market data, financial news, and uploaded documents.
* 🤖 **Multi-agent architecture** — Specialized AI agents independently analyze different research dimensions.
* 🧠 **Query planning** — Converts a broad research request into focused research queries.
* 📈 **Market analysis** — Retrieves and analyzes historical and financial market data.
* 📰 **News research** — Searches and analyzes relevant financial news.
* 📄 **Document intelligence** — Extracts useful information from user-provided documents.
* 🔗 **Evidence attribution** — Connects generated claims to supporting sources.
* ✅ **Evidence validation** — Validates research findings before they reach the final report.
* 🛡️ **Failure handling** — Detects unavailable or insufficient data instead of silently generating unsupported claims.
* ⚡ **Parallel research** — Independent research tasks can execute concurrently.
* 📊 **Structured reports** — Produces concise, readable investment research reports.
* 🔐 **Secure API architecture** — Backend APIs are separated into modular services and schemas.

---

# 🏗️ Architecture

FinSightAI follows a **planner → parallel research agents → synthesis → validation** architecture.

```text
                         ┌─────────────────────┐
                         │       User          │
                         │ Company + Query     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     FastAPI API     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Query Planner     │
                         │                     │
                         │ Breaks request into │
                         │ research questions  │
                         └──────────┬──────────┘
                                    │
                   ┌────────────────┼────────────────┐
                   │                │                │
                   ▼                ▼                ▼
          ┌────────────────┐ ┌───────────────┐ ┌────────────────┐
          │ Market Agent   │ │ News Agent    │ │ Document Agent │
          │                │ │               │ │                │
          │ Price / trends │ │ News / events │ │ User documents │
          │ fundamentals  │ │ catalysts     │ │ evidence       │
          └───────┬────────┘ └───────┬───────┘ └───────┬────────┘
                  │                  │                 │
                  └──────────────────┼─────────────────┘
                                     ▼
                         ┌─────────────────────┐
                         │   Research         │
                         │   Synthesizer       │
                         │                     │
                         │ Combines findings   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Evidence Validator  │
                         │                     │
                         │ Checks claims and   │
                         │ supporting sources  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Final Report      │
                         │                     │
                         │ Structured +        │
                         │ Source Attributed   │
                         └─────────────────────┘
```

---

# 🔄 Research Workflow

The complete research flow is:

```text
User Request
     │
     ▼
API Validation
     │
     ▼
Query Planner
     │
     ├───────────────┐
     │               │
     ▼               ▼
Research Tasks   Research Tasks
     │               │
     └───────┬───────┘
             ▼
     Parallel Agents
             │
      ┌──────┼──────┐
      ▼      ▼      ▼
    Market  News  Documents
      │      │      │
      └──────┼──────┘
             ▼
       Synthesis Agent
             │
             ▼
     Evidence Validation
             │
             ▼
        Final Report
```

---

# 🤖 Multi-Agent System

FinSightAI separates research responsibilities between specialized agents.

## 1. Query Planner

The Query Planner is responsible for converting the user's high-level research request into structured research tasks.

For example:

```text
User:
"Analyze Apple's growth catalysts and key risks."

                    ↓

Query Planner

                    ↓

┌────────────────────────────────────┐
│ Market Research                    │
│ - Revenue / price trends            │
│ - Historical performance             │
│ - Financial indicators               │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ News Research                      │
│ - Recent announcements              │
│ - Product launches                  │
│ - Market events                     │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ Risk Research                      │
│ - Competition                      │
│ - Regulatory risks                 │
│ - Macroeconomic factors             │
└────────────────────────────────────┘
```

This prevents every agent from receiving the same broad prompt and enables more focused research.

---

## 2. Market Analyst

The Market Analyst focuses on structured financial and market information.

Responsibilities include:

* Historical price analysis
* Market trends
* Financial metrics
* Company fundamentals
* Performance analysis
* Market-related evidence

The agent retrieves data through configured market-data providers and returns structured findings.

---

## 3. News Researcher

The News Researcher focuses on external financial news.

Responsibilities include:

* Recent company developments
* Product announcements
* Corporate events
* Industry developments
* Market catalysts
* Negative events and risks

News findings are returned with their corresponding source information.

---

## 4. Document Researcher

The Document Researcher processes documents uploaded by the user.

Documents can provide:

* Company reports
* Financial statements
* Research notes
* Presentations
* PDFs
* Other supporting material

The agent extracts relevant information and makes it available to the synthesis stage.

---

## 5. Research Synthesizer

The Research Synthesizer combines findings from the independent agents.

It is responsible for:

* Combining research findings
* Removing unnecessary duplication
* Organizing insights
* Connecting related evidence
* Generating structured conclusions

The synthesizer should rely on collected evidence rather than inventing unsupported information.

---

## 6. Evidence Validator

The Evidence Validator acts as a final quality-control layer.

It checks:

```text
Generated Claim
      │
      ▼
Supporting Evidence?
      │
 ┌────┴────┐
 │         │
YES        NO
 │         │
 ▼         ▼
Accept   Reject /
         Flag
```

This helps reduce unsupported or hallucinated claims in the final report.

---

# 🧩 Core Design Principles

## Separation of Responsibilities

Each agent has a specific responsibility.

```text
Planner       → What should we research?
Market Agent  → What does the market data show?
News Agent    → What is happening?
Document Agent→ What do the uploaded documents say?
Synthesizer   → What does all the evidence mean together?
Validator     → Is the final result supported?
```

This makes the system easier to debug, extend, and maintain.

---

## Evidence-First Generation

FinSightAI follows an evidence-first approach.

Instead of:

```text
LLM → Generate Answer
```

the system follows:

```text
Data
  ↓
Research
  ↓
Evidence
  ↓
Synthesis
  ↓
Validation
  ↓
Answer
```

This improves reliability and makes the generated research more transparent.

---

# 📁 Project Structure

```text
FinSightAI/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │
│   │   ├── agents/
│   │   │   ├── market_analyst.py
│   │   │   ├── news_researcher.py
│   │   │   ├── document_researcher.py
│   │   │   └── synthesizer.py
│   │   │
│   │   ├── schemas/
│   │   │   └── research.py
│   │   │
│   │   ├── services/
│   │   │
│   │   ├── llm/
│   │   │   └── groq_provider.py
│   │   │
│   │   ├── core/
│   │   │   └── config.py
│   │   │
│   │   └── main.py
│   │
│   ├── tests/
│   │
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── docs/
│   └── architecture/
│
├── docker-compose.yml
├── README.md
└── .gitignore
```

> The exact directory structure may evolve as the project grows.

---

# 🛠️ Tech Stack

## Backend

| Technology     | Purpose                       |
| -------------- | ----------------------------- |
| **Python**     | Core backend language         |
| **FastAPI**    | REST API framework            |
| **Pydantic**   | Request/response validation   |
| **AsyncIO**    | Concurrent research execution |
| **HTTPX**      | Async HTTP requests           |
| **SQLAlchemy** | Database interaction          |
| **PostgreSQL** | Persistent storage            |

## AI / LLM

| Technology             | Purpose                    |
| ---------------------- | -------------------------- |
| **Groq**               | LLM inference              |
| **LLM Agents**         | Specialized research tasks |
| **Structured Outputs** | Consistent agent responses |
| **Prompt Engineering** | Research task execution    |

## Data Sources

FinSightAI is designed around multiple external data sources:

```text
Market Data Providers
        +
Financial News Sources
        +
User Documents
        ↓
     FinSightAI
```

Provider availability can vary, and the application explicitly handles unavailable data sources.

## Frontend

The frontend provides:

* Company/ticker input
* Research question selection
* Research generation
* Research status
* Structured results
* Evidence/source display

---

# 🚀 Getting Started

## Prerequisites

Make sure you have:

* Python 3.12+
* Node.js
* npm
* Git
* PostgreSQL
* API keys for configured external services

---

# 📥 Clone the Repository

```bash
git clone https://github.com/<your-username>/FinSightAI.git

cd FinSightAI
```

---

# 🐍 Backend Setup

Navigate to the backend:

```bash
cd backend
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file inside the backend directory.

Example:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/finsightai

GROQ_API_KEY=your_groq_api_key

NEWS_API_KEY=your_news_api_key

MARKET_DATA_API_KEY=your_market_data_api_key

JWT_SECRET_KEY=your_secret_key
```

> Do not commit `.env` files or API keys to GitHub.

---

# 🗄️ Database Setup

Create the PostgreSQL database:

```sql
CREATE DATABASE finsightai;
```

Then configure the database connection in your environment variables.

If migrations are configured:

```bash
alembic upgrade head
```

---

# ▶️ Run the Backend

From the backend directory:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

---

# 💻 Frontend Setup

Navigate to the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will then be available at the development URL displayed by the frontend tooling.

---

# 🐳 Docker

FinSightAI can also be run using Docker Compose.

```bash
docker compose up --build
```

To run in the background:

```bash
docker compose up -d --build
```

To stop the services:

```bash
docker compose down
```

---

# 🔌 API Overview

The backend exposes APIs for research generation and retrieval.

Typical flow:

```text
POST /api/v1/research
        │
        ▼
Create Research Job
        │
        ▼
Research Pipeline
        │
        ▼
GET /api/v1/research/{id}
        │
        ▼
Research Result
```

Example request:

```json
{
  "company": "Apple",
  "ticker": "AAPL",
  "research_topics": [
    "Growth catalysts",
    "Key risks",
    "Competitive landscape"
  ]
}
```

Example conceptual response:

```json
{
  "status": "completed",
  "company": "Apple",
  "research": {
    "growth_catalysts": [],
    "key_risks": [],
    "competitive_landscape": []
  },
  "evidence": []
}
```

The exact API schema may change as the application evolves.

---

# 🧠 Query Orchestration

A major component of FinSightAI is its Query Planner.

The planner avoids sending one large prompt to a single LLM.

Instead:

```text
                    User Query
                         │
                         ▼
                  Query Planner
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Market           News         Documents
       Query            Query          Query
          │              │              │
          ▼              ▼              ▼
       Market           News        Document
       Agent            Agent         Agent
```

The resulting research outputs are then passed into the synthesis layer.

This architecture allows individual components to evolve independently.

---

# ⚡ Parallel Execution

Independent research tasks can be executed concurrently.

Conceptually:

```python
results = await asyncio.gather(
    market_agent.run(),
    news_agent.run(),
    document_agent.run()
)
```

Instead of:

```text
Market
  ↓
News
  ↓
Documents
```

the system can perform:

```text
Market ──────┐
             │
News ────────┼──→ Synthesis
             │
Documents ───┘
```

This reduces unnecessary waiting between independent research tasks.

---

# 🛡️ Reliability & Failure Handling

Financial research requires reliable data.

FinSightAI therefore treats missing data as a system state rather than silently replacing it with generated information.

For example:

```text
Market Provider
      │
      ▼
Data Available?
   │       │
  YES      NO
   │       │
   ▼       ▼
Analyze   Try configured
Data      fallback
             │
             ▼
       Still unavailable?
             │
             ▼
       Report limitation
```

If required historical market data is unavailable from configured providers, the system can explicitly mark the research task as failed instead of fabricating market information.

---

# 📚 Evidence Model

Research findings are associated with evidence.

Conceptually:

```text
Finding
  │
  ├── Claim
  │
  ├── Source
  │
  ├── Source Type
  │
  ├── Timestamp
  │
  └── Confidence / Validation
```

This allows users to understand **where a research statement came from**.

---

# 🔍 Example Research Flow

Input:

```text
Company: Microsoft
Question: What are Microsoft's growth catalysts and key risks?
```

### Step 1 — Planning

```text
Growth Catalysts
Key Risks
Competitive Landscape
Market Performance
Recent News
```

### Step 2 — Independent Research

```text
Market Agent
     ↓
Market findings

News Agent
     ↓
Recent developments

Document Agent
     ↓
Uploaded evidence
```

### Step 3 — Synthesis

```text
All research findings
        ↓
Synthesis Agent
        ↓
Structured report
```

### Step 4 — Validation

```text
Generated claims
        ↓
Evidence Validator
        ↓
Validated report
```

---

# 📊 Research Output

The final report is designed around structured sections such as:

```text
Company Overview

Growth Catalysts
├── Catalyst 1
├── Catalyst 2
└── Catalyst 3

Key Risks
├── Risk 1
├── Risk 2
└── Risk 3

Competitive Landscape
├── Competitor 1
├── Competitor 2
└── Competitor 3

Market Analysis

Recent Developments

Evidence & Sources
```

---

# 🔐 Security Considerations

FinSightAI follows several security principles:

* API keys are stored using environment variables.
* Secrets are excluded from version control.
* Input validation is handled through Pydantic models.
* API endpoints are separated from business logic.
* External API failures are handled explicitly.
* User-provided documents should be processed within controlled boundaries.
* Generated claims should be validated against retrieved evidence.

---

# 🧪 Testing

Run the test suite with:

```bash
pytest
```

For verbose output:

```bash
pytest -v
```

The test suite can cover:

* API validation
* Query planning
* Agent execution
* Provider failures
* Evidence validation
* Research status transitions
* End-to-end research workflows

---

# 📈 Scalability

The architecture is designed so research agents can be scaled independently.

```text
                   API
                    │
                    ▼
              Query Planner
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     Market        News       Documents
     Workers       Workers      Workers
        │           │           │
        └───────────┼───────────┘
                    ▼
                Synthesis
```

Potential future improvements include:

* Background job queues
* Redis-based caching
* Worker pools
* Provider fallback chains
* Rate-limit management
* Streaming research status
* Distributed agent execution

---

# 🚧 Current Limitations

FinSightAI depends on external information providers.

Therefore:

* Market data availability may vary by provider.
* Some tickers may not return historical data.
* News APIs may have rate limits.
* External APIs can temporarily fail.
* LLM responses may occasionally be incomplete.
* Uploaded documents may contain incomplete or conflicting information.

The system is designed to **surface these limitations instead of presenting unsupported information as fact**.

---

# 🔮 Future Improvements

## Research

* [ ] More market-data providers
* [ ] More financial-news providers
* [ ] SEC filing integration
* [ ] Earnings-call analysis
* [ ] Financial statement extraction
* [ ] Historical research comparison
* [ ] Industry-specific research agents

## AI

* [ ] Improved agent planning
* [ ] Agent-to-agent verification
* [ ] Better evidence ranking
* [ ] Confidence scoring
* [ ] Citation-aware generation
* [ ] Long-context document analysis
* [ ] Adaptive research depth

## Infrastructure

* [ ] Redis caching
* [ ] Background task queue
* [ ] Distributed workers
* [ ] Better observability
* [ ] Automated evaluation
* [ ] Provider health monitoring

## Product

* [ ] Saved research reports
* [ ] Research history
* [ ] Watchlists
* [ ] Portfolio-level research
* [ ] Custom research templates
* [ ] Export to PDF
* [ ] Report sharing

---

# 🎯 Why FinSightAI?

Traditional investment research often requires switching between:

```text
Financial Websites
       +
Market Data
       +
News
       +
Company Reports
       +
User Documents
       +
Manual Analysis
```

FinSightAI brings these workflows together:

```text
                FinSightAI

     ┌──────────────────────────┐
     │     Market Data          │
     ├──────────────────────────┤
     │     Financial News       │
     ├──────────────────────────┤
     │     User Documents       │
     └────────────┬─────────────┘
                  │
                  ▼
            Query Planner
                  │
                  ▼
          Multi-Agent Research
                  │
                  ▼
             Synthesis
                  │
                  ▼
          Evidence Validation
                  │
                  ▼
          Structured Research
```

The goal is not simply to generate an answer with an LLM.

The goal is to build a **research pipeline where information is collected, analyzed, synthesized, and validated before being presented to the user.**

---

# 🧑‍💻 Development Philosophy

FinSightAI is built around four principles:

### 1. Research before generation

The LLM should reason over retrieved information rather than relying entirely on its pretrained knowledge.

### 2. Specialization over monolithic prompting

Different research tasks should be handled by specialized agents.

### 3. Evidence over unsupported claims

Important claims should be connected to supporting evidence.

### 4. Explicit failure over hallucination

When required data cannot be retrieved, the system should report the limitation rather than inventing an answer.

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/your-feature
```

3. Make your changes.
4. Add or update tests.
5. Run the test suite.

```bash
pytest
```

6. Commit your changes.

```bash
git commit -m "feat: add your feature"
```

7. Push the branch.

```bash
git push origin feature/your-feature
```

8. Open a Pull Request.

---

# 📄 License

This project is intended for educational, research, and demonstration purposes.

Add the project's chosen open-source license here if/when one is formally selected.

---

# ⚠️ Disclaimer

FinSightAI is an **AI-powered investment research assistant**.

It provides information and research summaries based on available data and sources. It does **not** provide financial, investment, legal, tax, or other professional advice.

Users should independently verify information and consult qualified professionals before making financial decisions.

---

# ⭐ FinSightAI

**Research smarter. Analyze faster. Verify the evidence.**
