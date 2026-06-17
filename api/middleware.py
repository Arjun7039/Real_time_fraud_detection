import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram

# Prometheus Metrics
REQUEST_COUNT = Counter(
    "realguard_api_requests_total",
    "Total HTTP requests to the API",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "realguard_api_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint"]
)

FRAUD_DETECTED = Counter(
    "realguard_fraud_detected_total",
    "Total number of transactions flagged as fraud",
    ["endpoint"]
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to track API latency and request counts."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        # We only care about tracking specific endpoints, not metrics/health
        if request.url.path == "/predict":
            REQUEST_LATENCY.labels(endpoint=request.url.path).observe(process_time)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()
            
        return response
