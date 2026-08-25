# EquityLens

EquityLens is a multi-tenant investment-research SaaS prototype. Users sign up into an organization, submit a company question, follow a persisted research job, and review an evidence-backed report. It is a research assistant—not financial advice.

## Architecture

```text
User → React + TypeScript (Vercel) → FastAPI (Render) → PostgreSQL
                                      ↓
                         Research Orchestrator ── Finnhub
                           ├─ Market Agent
                           ├─ News Agent
                           └─ Document/RAG Agent ── OpenAI + tenant-filtered FAISS
                                      ↓
                            Research Synthesizer → Research Report
```

## Stack and core features

- React, TypeScript, Vite; FastAPI, SQLAlchemy, Alembic, PostgreSQL.
- JWT authentication, bcrypt password hashing, role checks, organization isolation.
- Bounded concurrent specialist-agent workflow with validated Pydantic results and evidence allow-lists.
- Local FAISS RAG for tenant-owned text/PDF documents; document, chunk, job, and report reads are organization-scoped.
- Background in-process research jobs, persisted reports, pagination, health checks, structured logging, and safe API error envelopes.

## Database and tenant isolation

Shared canonical data lives in `companies`, `market_data`, `news_articles`, and `research_sources`. Tenant-owned `documents`, `document_chunks`, `research_jobs`, and `research_reports` carry `organization_id`; repositories add this predicate server-side. The frontend never selects an organization. Alembic is the production schema authority—`Base.metadata.create_all()` exists only in test fixtures.

## Local setup

1. Copy `.env.example` to `.env` and provide real local PostgreSQL, JWT, Finnhub, and OpenAI values.
2. `docker compose up --build` starts PostgreSQL and the API, applying `alembic upgrade head` first.
3. In `frontend/`, run `npm install`, then `npm run dev`.
4. Open `http://localhost:5173`, sign up, and submit: `Analyze recent performance, major risks, and growth opportunities.` for `NVDA`.

The frontend uses `VITE_API_URL` at build time. Its localhost fallback is development-only; deploy with the HTTPS backend URL. Browser storage holds the short-lived JWT for this prototype; it is vulnerable to XSS, so production evolution should use hardened cookies/CSP/token rotation.

## Environment variables

See `.env.example` for descriptions. Required production values are `ENVIRONMENT=production`, `DEBUG=false`, PostgreSQL `DATABASE_URL`, a unique 32+ character `JWT_SECRET_KEY`, explicit JSON `CORS_ORIGINS`, `FRONTEND_URL`, `FINNHUB_API_KEY`, and `GROQ_API_KEY`. Set `LLM_PROVIDER=groq` and `LLM_MODEL=openai/gpt-oss-120b`. Never commit `.env`, database passwords, JWTs, or provider keys.

## Tests and build

```bash
cd backend && pytest -q
cd frontend && npm test && npm run build
```

Tests use fake providers, so no external API access is required. The smoke test covers health, signup/login, protected access, research/job/report flow, and cross-tenant report denial.

## Deployment

The supported simple architecture is Vercel frontend + Render web service + Render managed PostgreSQL. `render.yaml` defines the backend service, migration start command, health path, and database. Create the web service/database from that blueprint; set remaining secrets in Render’s dashboard, including `CORS_ORIGINS=["https://YOUR-VERCEL-DOMAIN"]` and `FRONTEND_URL`. Create a Vercel project with `frontend/` as root and set `VITE_API_URL=https://YOUR-RENDER-DOMAIN/api/v1`.

Before release, verify Render logs show `alembic upgrade head` succeeds, request `GET /api/v1/health`, and perform the demo flow above. No live deployment URL is committed because deployment credentials are not part of this repository.

## API examples

```text
POST /api/v1/auth/signup
POST /api/v1/auth/login
GET  /api/v1/health
GET  /api/v1/companies/NVDA
POST /api/v1/research
GET  /api/v1/research?page=1&page_size=10
GET  /api/v1/research/{job_id}
GET  /api/v1/reports/{report_id}
```

OpenAPI is available at `/docs`; it documents auth, organization, companies, research, reports, and health.

## Known limitations and next steps

Background tasks and the in-memory research rate limit are per-process, so they do not provide durable queuing or distributed quotas. Local FAISS is intentionally small-scale. Provider outages produce controlled job failures rather than fabricated research. Next steps: durable workers/quotas, object storage and asynchronous ingestion, hardened cookie authentication, CSP/security headers, monitoring/alerts, database backups, and production load testing.
