import logging
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from app.api.v1 import admin, auth, companies, health, organization, research, reports
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
app = FastAPI(title="EquityLens API", version="0.1.0", debug=settings.debug, openapi_tags=[{"name": "auth", "description": "Authentication"}, {"name": "organization", "description": "Tenant organization"}, {"name": "companies", "description": "Canonical market data"}, {"name": "research", "description": "Tenant research jobs"}, {"name": "reports", "description": "Tenant research reports"}, {"name": "health", "description": "Deployment health"}])
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False, allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"], allow_headers=["Authorization", "Content-Type"])
app.include_router(auth.router, prefix="/api/v1")
app.include_router(organization.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
app.include_router(companies.router, prefix="/api/v1")
app.include_router(research.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")


@app.middleware("http")
async def request_observability(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH"}:
        length = request.headers.get("content-length")
        if length and int(length) > settings.max_request_body_bytes:
            return JSONResponse(status_code=413, content={"success": False, "error": {"code": "REQUEST_TOO_LARGE", "message": "Request body is too large."}})
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled request error", extra={"method": request.method, "path": request.url.path, "error_category": "unhandled"})
        return JSONResponse(status_code=500, content={"success": False, "error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred."}})
    logger.info("request completed", extra={"method": request.method, "path": request.url.path, "status": response.status_code, "duration_ms": round((time.monotonic() - started) * 1000, 2)})
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "HTTP_ERROR", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content={"success": False, "error": detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    first = exc.errors()[0]
    return JSONResponse(status_code=422, content={"success": False, "error": {"code": "VALIDATION_ERROR", "message": first["msg"]}})


@app.exception_handler(SQLAlchemyError)
async def database_exception_handler(_: Request, __: SQLAlchemyError):
    return JSONResponse(status_code=503, content={"success": False, "error": {"code": "DATABASE_ERROR", "message": "Database operation failed."}})
