from fastapi import FastAPI, HTTPException, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import logging
import traceback
from datetime import datetime

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────
MAIN_SERVER_URL = os.getenv("MAIN_SERVER_URL", "").rstrip("/")
# Optional: forward the same Authorization header the client sends,
# or override with a fixed token stored in MAIN_AUTH env var.
MAIN_AUTH = os.getenv("MAIN_AUTH", "")

# ── App ────────────────────────────────────────────────────
app = FastAPI(
    title="TopUp Proxy",
    description="Lightweight proxy – forwards /topup to MAIN_SERVER_URL",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


# ── Request logger ─────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"▶ {request.method} {request.url.path}  origin={request.headers.get('origin', '-')}")
    start = datetime.utcnow()
    try:
        response = await call_next(request)
        ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(f"◀ {response.status_code}  ({ms:.1f} ms)")
        return response
    except Exception as exc:
        logger.error(f"❌ Middleware error: {exc}\n{traceback.format_exc()}")
        raise


# ── Pydantic model (mirrors the original API exactly) ──────
class TopUpRequest(BaseModel):
    orderid: str = Field(..., description="Unique order ID")
    playerid: str = Field(..., description="Player/User ID")
    code: str = Field(..., description="Comma-separated UC codes (max 5)")
    url: str = Field(..., description="Callback URL for order status")
    metadata: Optional[dict] = Field(None, description="Optional metadata (any key-value pairs)")


class TopUpResponse(BaseModel):
    status: str
    orderid: str
    message: Optional[str] = None


# ── Helper ─────────────────────────────────────────────────
def _get_token(authorization: Optional[str], query_token: Optional[str]) -> Optional[str]:
    if authorization:
        return authorization.replace("Bearer ", "").replace("bearer ", "").strip()
    if query_token:
        return query_token.strip()
    return None


# ══════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """
    Health check.
    Also pings the main server's /health endpoint and surfaces its data.
    """
    main_health: dict = {}

    if MAIN_SERVER_URL:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{MAIN_SERVER_URL}/health")
                if r.status_code == 200:
                    main_health = r.json()
        except Exception as exc:
            logger.warning(f"⚠️  Could not reach main server health: {exc}")
            main_health = {"error": "main server unreachable"}
    else:
        main_health = {"error": "MAIN_SERVER_URL not configured"}

    return {
        "status": "healthy",
        "proxy": "TopUp Proxy v1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "main_server": main_health,
    }


@app.post("/topup", response_model=TopUpResponse)
async def topup_order(
    request_data: TopUpRequest,
    req: Request,
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
):
    """
    Receives a top-up request exactly as the original API does,
    then forwards it (with the same payload + auth) to MAIN_SERVER_URL/topup.
    """
    if not MAIN_SERVER_URL:
        raise HTTPException(status_code=503, detail="MAIN_SERVER_URL is not configured")

    # ── Build the token to forward ──────────────────────────
    client_token = _get_token(authorization, token)

    # Prefer the token the client sent; fall back to MAIN_AUTH if set
    forward_token = client_token or MAIN_AUTH

    if not forward_token:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Missing API token",
                "message": "Provide token via Authorization header or ?token= query param",
            },
        )

    # ── Build forwarding headers ────────────────────────────
    forward_headers = {
        "Content-Type": "application/json",
        "Authorization": forward_token,
    }

    # ── Build payload (identical to what the client sent) ───
    payload: dict = {
        "orderid": request_data.orderid,
        "playerid": request_data.playerid,
        "code": request_data.code,
        "url": request_data.url,
    }
    if request_data.metadata is not None:
        payload["metadata"] = request_data.metadata

    target_url = f"{MAIN_SERVER_URL}/topup"
    logger.info(f"📤 Forwarding order {request_data.orderid!r} → {target_url}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(target_url, json=payload, headers=forward_headers)

        logger.info(f"📥 Main server responded {resp.status_code}")

        # Surface any error from the main server verbatim
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise HTTPException(status_code=resp.status_code, detail=detail)

        return resp.json()

    except HTTPException:
        raise
    except httpx.TimeoutException:
        logger.error("❌ Timeout while contacting main server")
        raise HTTPException(status_code=504, detail="Main server timed out")
    except Exception as exc:
        logger.error(f"❌ Unexpected error forwarding request: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=502, detail=f"Proxy error: {str(exc)}")


# ── Catch-all (keeps the same UX as the original) ─────────
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def catch_all(full_path: str):
    return {
        "status": "info",
        "proxy": "TopUp Proxy v1.0.0",
        "requested_path": f"/{full_path}",
        "available_endpoints": ["GET /health", "POST /topup"],
    }


# ── Entry-point ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting proxy on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)
