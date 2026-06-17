"""Simulates a banking frontend hitting the /predict API at high speed."""

import csv
import json
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8000/predict"
CSV_PATH = "data/raw/paysim_dataset.csv"
RATE = 100  # requests per second

def send_request(payload):
    try:
        requests.post(API_URL, json=payload, timeout=2)
    except Exception:
        pass

def main():
    print(f"🚀 Firing transactions at {API_URL} ({RATE} TPS)...")
    
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            batch_start = time.perf_counter()
            count = 0
            
            for row in reader:
                payload = {
                    "step": int(row["step"]),
                    "type": row["type"],
                    "amount": float(row["amount"]),
                    "nameOrig": row["nameOrig"],
                    "oldbalanceOrg": float(row["oldbalanceOrg"]),
                    "newbalanceOrig": float(row["newbalanceOrig"]),
                    "nameDest": row["nameDest"],
                    "oldbalanceDest": float(row["oldbalanceDest"]),
                    "newbalanceDest": float(row["newbalanceDest"]),
                }
                
                executor.submit(send_request, payload)
                count += 1
                
                # Simple rate limiting
                if count >= RATE:
                    elapsed = time.perf_counter() - batch_start
                    if elapsed < 1.0:
                        time.sleep(1.0 - elapsed)
                    
                    batch_start = time.perf_counter()
                    count = 0

if __name__ == "__main__":
    main()
