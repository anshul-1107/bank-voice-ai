"""Domain models. Mirrors what a real core-banking / LMS / policy-admin API returns, so swapping the mock for real APIs is a drop-in."""

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Language(str, Enum):
    EN = "en-IN"
    HI = "hi-IN"        # Hindi / Hinglish (code-mixed)


class Customer(BaseModel):
    customer_id: str
    name: str
    first_name: str
    phone: str                       # E.164, e.g. +919876543210
    dob: date
    preferred_language: Language = Language.EN
    gender: str = "U"                # used only for Hindi verb agreement in addressing ("aap" is neutral anyway)
    consent_service_calls: bool = True
    do_not_call: bool = False        # customer opted out of reminder calls
    preferred_call_hour: int | None = None


class LoanStatus(str, Enum):
    ACTIVE = "active"
    OVERDUE = "overdue"
    CLOSED = "closed"


class Loan(BaseModel):
    account_no: str
    customer_id: str
    product: str                     # "Personal Loan", "Home Loan", "Two-Wheeler Loan", "Consumer Durable Loan"
    sanctioned_amount: int           # rupees
    outstanding_principal: int
    emi_amount: int
    interest_rate_pa: float
    next_due_date: date
    emis_remaining: int
    status: LoanStatus = LoanStatus.ACTIVE
    overdue_amount: int = 0
    late_fee: int = 0
    last_payment_date: date | None = None
    autopay_mandate: bool = False    # NACH / e-mandate active

    @property
    def masked(self) -> str:
        return f"XXXX{self.account_no[-4:]}"


class PolicyStatus(str, Enum):
    ACTIVE = "active"
    DUE_FOR_RENEWAL = "due_for_renewal"
    GRACE = "grace_period"
    LAPSED = "lapsed"


class Policy(BaseModel):
    policy_no: str
    customer_id: str
    product: str                     # "Health Insurance", "Term Life", "Motor Insurance", "Loan Protection"
    insurer: str
    sum_assured: int
    premium: int
    renewal_date: date
    status: PolicyStatus = PolicyStatus.ACTIVE
    grace_days: int = 30

    @property
    def masked(self) -> str:
        return f"XXXX{self.policy_no[-4:]}"


class Ticket(BaseModel):
    ticket_id: str
    customer_id: str
    category: str
    summary: str
    priority: str = "normal"


class PaymentLink(BaseModel):
    link_id: str
    account_no: str
    amount: int
    sent_to: str                     # masked phone
    channel: str = "SMS"


class CallPurpose(str, Enum):
    INBOUND = "inbound"
    EMI_DUE = "emi_due_reminder"
    EMI_OVERDUE = "emi_overdue"
    POLICY_RENEWAL = "policy_renewal_reminder"


class CallContext(BaseModel):
    """What we know before the first word is spoken."""
    call_id: str
    direction: str = "inbound"       # inbound | outbound
    purpose: CallPurpose = CallPurpose.INBOUND
    caller_number: str = ""
    customer_id: str | None = None   # pre-identified for outbound; ANI-matched (unverified) for inbound
    reference_id: str | None = None  # loan account / policy number the reminder is about
    language_hint: Language = Language.EN
    metadata: dict = Field(default_factory=dict)
