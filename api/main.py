import logging
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from api.schemas import PredictRequest, PredictResponse
from api.predictor import FraudPredictor
from api.middleware import PrometheusMiddleware, FRAUD_DETECTED

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RealGuard Fraud Detection API",
    description="Real-time machine learning inference endpoint for PaySim transactions.",
    version="1.0.0"
)

# Add metrics middleware
app.add_middleware(PrometheusMiddleware)

# Initialize the predictor (singleton) on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application...")
    FraudPredictor() # Forces initialization of models

@app.post("/predict", response_model=PredictResponse)
async def predict_fraud(tx: PredictRequest):
    """
    Evaluates a single transaction for fraud.
    
    Returns the fraud probability, boolean decision, and top 3 SHAP reason codes
    if the transaction is flagged as fraud.
    """
    try:
        predictor = FraudPredictor()
        response = predictor.predict(tx)
        
        # Track fraud detections in Prometheus
        if response.is_fraud:
            FRAUD_DETECTED.labels(endpoint="/predict").inc()
            
        return response
    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Simple health probe for Docker/Kubernetes."""
    return {"status": "ok", "service": "realguard-api"}

@app.get("/metrics")
async def metrics():
    """Prometheus scraping endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
