"""Kafka producer — streams PaySim CSV rows as live transaction events.

Reads the PaySim dataset row by row and publishes each transaction
as a JSON message to the Kafka topic 'raw-transactions' at a
configurable rate to simulate real-time transaction flow.

Usage:
    python ingestion/producer.py --rate 100
    python ingestion/producer.py --rate 10 --max-rows 1000
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
from typing import Optional

from confluent_kafka import Producer, KafkaError

# --------------- Configuration ---------------
KAFKA_BROKER: str = os.getenv("KAFKA_BROKER", "localhost:29092")
KAFKA_TOPIC: str = "raw-transactions"
CSV_PATH: str = os.path.join("data", "raw", "paysim_dataset.csv")

# Graceful shutdown flag
_shutdown: bool = False


def _signal_handler(signum: int, frame) -> None:
    """Handle SIGINT / SIGTERM for graceful shutdown."""
    global _shutdown
    print("\n⛔ Shutdown signal received — finishing current batch...")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _delivery_callback(err: Optional[KafkaError], msg) -> None:
    """Called once per message to indicate delivery result."""
    if err is not None:
        print(f"❌ Delivery failed: {err}")


def create_producer() -> Producer:
    """Create and return a configured Kafka producer instance.

    Returns:
        Producer: A confluent_kafka Producer connected to the broker.
    """
    conf = {
        "bootstrap.servers": KAFKA_BROKER,
        "client.id": "realguard-producer",
        "linger.ms": 5,          # micro-batch for throughput
        "batch.num.messages": 500,
        "queue.buffering.max.messages": 100000,
        "compression.type": "lz4",
    }
    return Producer(conf)


def stream_transactions(
    producer: Producer,
    csv_path: str,
    rate: int,
    max_rows: Optional[int] = None,
) -> None:
    """Read CSV and publish each row to Kafka at the specified rate.

    Args:
        producer: The Kafka producer instance.
        csv_path: Path to the PaySim CSV file.
        rate: Target transactions per second.
        max_rows: Optional cap on total rows to publish.
    """
    batch_size: int = max(1, rate // 10)  # flush every ~100ms worth
    interval: float = batch_size / rate   # seconds between batches
    sent: int = 0

    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        batch_start = time.perf_counter()
        batch_count = 0

        for row in reader:
            if _shutdown:
                break
            if max_rows is not None and sent >= max_rows:
                break

            # Build JSON payload (cast numeric types)
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
                "isFraud": int(row["isFraud"]),
                "isFlaggedFraud": int(row["isFlaggedFraud"]),
            }

            producer.produce(
                topic=KAFKA_TOPIC,
                key=row["nameOrig"],
                value=json.dumps(payload).encode("utf-8"),
                callback=_delivery_callback,
            )
            sent += 1
            batch_count += 1

            # Rate limiting: sleep after each micro-batch
            if batch_count >= batch_size:
                producer.poll(0)  # trigger delivery callbacks
                elapsed = time.perf_counter() - batch_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                batch_start = time.perf_counter()
                batch_count = 0

            # Progress logging every 10,000 rows
            if sent % 10_000 == 0:
                print(f"📤 Published {sent:,} transactions to '{KAFKA_TOPIC}'")

    # Final flush
    remaining = producer.flush(timeout=10)
    print(f"\n✅ Done — {sent:,} transactions published to '{KAFKA_TOPIC}'")
    if remaining > 0:
        print(f"⚠️  {remaining} messages still in queue after flush timeout")


def main() -> None:
    """Entry point — parse CLI args and start streaming."""
    parser = argparse.ArgumentParser(
        description="RealGuard Kafka Producer — stream PaySim transactions"
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=100,
        help="Transactions per second (default: 100)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum rows to publish (default: all)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=CSV_PATH,
        help=f"Path to PaySim CSV (default: {CSV_PATH})",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"❌ CSV file not found: {args.csv}")
        sys.exit(1)

    print(f"🚀 RealGuard Producer starting")
    print(f"   Broker : {KAFKA_BROKER}")
    print(f"   Topic  : {KAFKA_TOPIC}")
    print(f"   Rate   : {args.rate} TPS")
    print(f"   CSV    : {args.csv}")
    print()

    producer = create_producer()
    stream_transactions(producer, args.csv, args.rate, args.max_rows)


if __name__ == "__main__":
    main()
