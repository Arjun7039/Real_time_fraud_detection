"""Pydantic schemas for transaction events.

Defines the Transaction model used for Kafka message
serialisation/deserialisation throughout the pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional


class Transaction(BaseModel):
    """Represents a single financial transaction from the PaySim dataset.

    Attributes:
        step: Time unit (1 step = 1 hour, total 744 steps = 30 days).
        type: Transaction type — TRANSFER, CASH_OUT, PAYMENT, CASH_IN, DEBIT.
        amount: Transaction amount in currency units.
        nameOrig: Origin account ID.
        oldbalanceOrg: Origin account balance before transaction.
        newbalanceOrig: Origin account balance after transaction.
        nameDest: Destination account ID.
        oldbalanceDest: Destination account balance before transaction.
        newbalanceDest: Destination account balance after transaction.
        isFraud: Ground truth label (1 = fraud, 0 = legitimate).
        isFlaggedFraud: Heuristic flag from the simulator.
    """

    step: int = Field(..., description="Time unit (1 step = 1 hour)")
    type: str = Field(..., description="Transaction type: TRANSFER, CASH_OUT, etc.")
    amount: float = Field(..., description="Transaction amount")
    nameOrig: str = Field(..., description="Origin account ID")
    oldbalanceOrg: float = Field(..., description="Origin balance before transaction")
    newbalanceOrig: float = Field(..., description="Origin balance after transaction")
    nameDest: str = Field(..., description="Destination account ID")
    oldbalanceDest: float = Field(..., description="Destination balance before transaction")
    newbalanceDest: float = Field(..., description="Destination balance after transaction")
    isFraud: int = Field(0, description="Ground truth label (1=fraud, 0=legit)")
    isFlaggedFraud: int = Field(0, description="Heuristic flag from simulator")
