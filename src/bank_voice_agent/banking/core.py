"""Core-banking service interface + an in-memory mock with realistic Indian data.

Swap MockCoreBanking for a real client implementing CoreBanking (LMS, policy admin, payments, CRM).
Every method is read-only except create_ticket / send_payment_link / record_* — the agent can never move money.
"""

from __future__ import annotations

import random
import uuid
from datetime import date, timedelta
from typing import Protocol

from bank_voice_agent.banking.models import (
    Customer,
    Language,
    Loan,
    LoanStatus,
    PaymentLink,
    Policy,
    PolicyStatus,
    Ticket,
)


class CoreBanking(Protocol):
    def find_customer_by_phone(self, phone: str) -> Customer | None: ...
    def get_customer(self, customer_id: str) -> Customer | None: ...
    def find_customer_by_account_suffix(self, last4: str) -> list[Customer]: ...
    def get_loans(self, customer_id: str) -> list[Loan]: ...
    def get_policies(self, customer_id: str) -> list[Policy]: ...
    def foreclosure_quote(self, account_no: str, on: date) -> dict: ...
    def send_payment_link(self, customer_id: str, account_no: str, amount: int) -> PaymentLink: ...
    def create_ticket(self, customer_id: str, category: str, summary: str, priority: str) -> Ticket: ...
    def record_promise_to_pay(self, customer_id: str, account_no: str, promised_date: date) -> dict: ...
    def record_opt_out(self, customer_id: str) -> None: ...
    def all_customers(self) -> list[Customer]: ...


def _mask_phone(phone: str) -> str:
    return f"XXXXXX{phone[-4:]}"


class MockCoreBanking:
    """Deterministic mock. Dates are relative to `today` so reminder campaigns always have work to do."""

    FIRST = ["Anshul","Ditya","Ramu","Rahul", "Priya", "Amit", "Sunita", "Arjun", "Kavya", "Rohan", "Neha", "Vikram", "Anjali",
             "Suresh", "Meera", "Karan", "Pooja", "Imran", "Lakshmi", "Deepak", "Ritu", "Manoj", "Fatima"]
    LAST = ["Mathur","Saini","Singh","Sharma", "Iyer", "Mehta", "Nair", "Reddy", "Gupta", "Khan", "Patel", "Menon", "Singh",
            "Das", "Joshi", "Kulkarni", "Verma", "Pillai", "Banerjee"]
    LOANS = [("Personal Loan", 14.5, 200_000, 1_500_000), ("Home Loan", 8.6, 2_500_000, 9_000_000),
             ("Two-Wheeler Loan", 11.0, 60_000, 180_000), ("Consumer Durable Loan", 0.0, 20_000, 120_000)]
    POLICIES = [("Health Insurance", "Star Health", 500_000, 18_000), ("Term Life", "HDFC Life", 10_000_000, 14_500),
                ("Motor Insurance", "ICICI Lombard", 600_000, 9_800), ("Loan Protection", "SBI Life", 1_000_000, 6_200)]

    def __init__(self, n_customers: int = 500, seed: int = 7, today: date | None = None):
        self.today = today or date.today()
        self._rng = random.Random(seed)
        self.customers: dict[str, Customer] = {}
        self.loans: dict[str, Loan] = {}
        self.policies: dict[str, Policy] = {}
        self.tickets: list[Ticket] = []
        self.payment_links: list[PaymentLink] = []
        self.promises: list[dict] = []
        self._seed_golden()
        self._seed_random(n_customers)

    # ---------- seed ----------
    def _seed_golden(self) -> None:
        """Fixed customers used by the eval suite. Do not change without updating evals/scenarios."""
        t = self.today
        golden = [
            (Customer(customer_id="C0001", name="Rahul Sharma", first_name="Rahul", phone="+919800000001",
                      dob=date(1990, 5, 14), preferred_language=Language.HI, gender="M"),
             [Loan(account_no="PL20234821", customer_id="C0001", product="Personal Loan", sanctioned_amount=500_000,
                   outstanding_principal=312_640, emi_amount=12_450, interest_rate_pa=14.5,
                   next_due_date=t + timedelta(days=3), emis_remaining=28, last_payment_date=t - timedelta(days=27))],
             [Policy(policy_no="HI99007312", customer_id="C0001", product="Health Insurance", insurer="Star Health",
                     sum_assured=500_000, premium=18_400, renewal_date=t + timedelta(days=45))]),
            (Customer(customer_id="C0002", name="Sunita Iyer", first_name="Sunita", phone="+919800000002",
                      dob=date(1978, 11, 2), preferred_language=Language.EN, gender="F"),
             [],
             [Policy(policy_no="TL55001190", customer_id="C0002", product="Term Life", insurer="HDFC Life",
                     sum_assured=10_000_000, premium=14_500, renewal_date=t + timedelta(days=7),
                     status=PolicyStatus.DUE_FOR_RENEWAL)]),
            (Customer(customer_id="C0003", name="Arjun Mehta", first_name="Arjun", phone="+919800000003",
                      dob=date(1986, 1, 23), preferred_language=Language.HI, gender="M"),
             [Loan(account_no="TW20220077", customer_id="C0003", product="Two-Wheeler Loan", sanctioned_amount=120_000,
                   outstanding_principal=41_300, emi_amount=4_120, interest_rate_pa=11.0,
                   next_due_date=t - timedelta(days=4), emis_remaining=11, status=LoanStatus.OVERDUE,
                   overdue_amount=4_120, late_fee=500, last_payment_date=t - timedelta(days=34))],
             []),
            (Customer(customer_id="C0004", name="Kavya Nair", first_name="Kavya", phone="+919800000004",
                      dob=date(1995, 8, 30), preferred_language=Language.EN, gender="F", do_not_call=True),
             [Loan(account_no="HL20190456", customer_id="C0004", product="Home Loan", sanctioned_amount=4_500_000,
                   outstanding_principal=3_862_000, emi_amount=38_900, interest_rate_pa=8.6,
                   next_due_date=t + timedelta(days=1), emis_remaining=196, autopay_mandate=True)],
             []),
            (Customer(customer_id="C0005", name="Anshul Mathur", first_name="Anshul", phone="+919636274630",
                      dob=date(2004, 7, 11), preferred_language=Language.HI, gender="M"),
             [Loan(account_no="PL20249636", customer_id="C0005", product="Personal Loan", sanctioned_amount=500_000,
                   outstanding_principal=312_640, emi_amount=12_450, interest_rate_pa=14.5,
                   next_due_date=t + timedelta(days=3), emis_remaining=28, last_payment_date=t - timedelta(days=27))],
             [Policy(policy_no="HI20249636", customer_id="C0005", product="Health Insurance", insurer="Star Health",
                     sum_assured=500_000, premium=18_400, renewal_date=t + timedelta(days=45))]),
            (Customer(customer_id="C0006", name="Ditya Saini", first_name="Ditya", phone="+918209641146",
                      dob=date(2003, 12, 5), preferred_language=Language.HI, gender="M"),
             [],
             [Policy(policy_no="TL20248209", customer_id="C0006", product="Term Life", insurer="HDFC Life",
                     sum_assured=10_000_000, premium=14_500, renewal_date=t + timedelta(days=7),
                     status=PolicyStatus.DUE_FOR_RENEWAL)]),
            (Customer(customer_id="C0007", name="Ramu Singh", first_name="Ramu", phone="+918619048715",
                      dob=date(1986, 1, 23), preferred_language=Language.HI, gender="M"),
             [Loan(account_no="TW20248619", customer_id="C0007", product="Two-Wheeler Loan", sanctioned_amount=120_000,
                   outstanding_principal=41_300, emi_amount=4_120, interest_rate_pa=11.0,
                   next_due_date=t - timedelta(days=4), emis_remaining=11, status=LoanStatus.OVERDUE,
                   overdue_amount=4_120, late_fee=500, last_payment_date=t - timedelta(days=34))],
             []),
        ]
        for c, loans, pols in golden:
            self.customers[c.customer_id] = c
            for loan in loans:
                self.loans[loan.account_no] = loan
            for p in pols:
                self.policies[p.policy_no] = p

    def _seed_random(self, n: int) -> None:
        r = self._rng
        for i in range(8, n + 8):
            cid = f"C{i:04d}"
            fn, ln = r.choice(self.FIRST), r.choice(self.LAST)
            c = Customer(
                customer_id=cid, name=f"{fn} {ln}", first_name=fn, phone=f"+9198{r.randint(10_000_000, 99_999_999)}",
                dob=date(r.randint(1960, 2002), r.randint(1, 12), r.randint(1, 28)),
                preferred_language=r.choice([Language.EN, Language.HI, Language.HI]),
                do_not_call=r.random() < 0.03,
            )
            self.customers[cid] = c
            if r.random() < 0.75:
                prod, rate, lo, hi = r.choice(self.LOANS)
                sanct = r.randrange(lo, hi, 5_000)
                outstanding = int(sanct * r.uniform(0.15, 0.95))
                emi = max(1_000, int(sanct / r.choice([12, 24, 36, 60, 120, 240]) * 1.08) // 10 * 10)
                due_offset = r.randint(-10, 30)
                overdue = due_offset < 0 and r.random() < 0.6
                acct = f"{prod[:2].upper()}{r.randint(10_000_000, 99_999_999)}"
                self.loans[acct] = Loan(
                    account_no=acct, customer_id=cid, product=prod, sanctioned_amount=sanct,
                    outstanding_principal=outstanding, emi_amount=emi, interest_rate_pa=rate,
                    next_due_date=self.today + timedelta(days=due_offset), emis_remaining=r.randint(3, 200),
                    status=LoanStatus.OVERDUE if overdue else LoanStatus.ACTIVE,
                    overdue_amount=emi if overdue else 0, late_fee=500 if overdue else 0,
                    autopay_mandate=r.random() < 0.4,
                )
            if r.random() < 0.45:
                prod, insurer, sa, prem = r.choice(self.POLICIES)
                ren = r.randint(-20, 60)
                status = (PolicyStatus.GRACE if ren < 0 else
                          PolicyStatus.DUE_FOR_RENEWAL if ren <= 30 else PolicyStatus.ACTIVE)
                pno = f"{prod[:2].upper()}{r.randint(10_000_000, 99_999_999)}"
                self.policies[pno] = Policy(policy_no=pno, customer_id=cid, product=prod, insurer=insurer,
                                            sum_assured=sa, premium=int(prem * r.uniform(0.8, 1.3)),
                                            renewal_date=self.today + timedelta(days=ren), status=status)

    # ---------- reads ----------
    def find_customer_by_phone(self, phone: str) -> Customer | None:
        norm = phone[-10:]
        return next((c for c in self.customers.values() if c.phone[-10:] == norm), None)

    def get_customer(self, customer_id: str) -> Customer | None:
        return self.customers.get(customer_id)

    def find_customer_by_account_suffix(self, last4: str) -> list[Customer]:
        ids = {l.customer_id for l in self.loans.values() if l.account_no.endswith(last4)}
        ids |= {p.customer_id for p in self.policies.values() if p.policy_no.endswith(last4)}
        return [self.customers[i] for i in ids]

    def get_loans(self, customer_id: str) -> list[Loan]:
        return [l for l in self.loans.values() if l.customer_id == customer_id and l.status != LoanStatus.CLOSED]

    def get_policies(self, customer_id: str) -> list[Policy]:
        return [p for p in self.policies.values() if p.customer_id == customer_id]

    def all_customers(self) -> list[Customer]:
        return list(self.customers.values())

    def foreclosure_quote(self, account_no: str, on: date) -> dict:
        loan = self.loans[account_no]
        # Mock rule: 4% foreclosure charge on personal / two-wheeler, 0% on floating-rate home loans (RBI).
        pct = 0.0 if loan.product == "Home Loan" else 0.04
        accrued = round(loan.outstanding_principal * loan.interest_rate_pa / 100 / 365 * 15)
        charge = round(loan.outstanding_principal * pct)
        gst = round(charge * 0.18)
        total = loan.outstanding_principal + accrued + charge + gst + loan.overdue_amount + loan.late_fee
        return {"account": loan.masked, "valid_till": (on + timedelta(days=7)).isoformat(),
                "principal": loan.outstanding_principal, "accrued_interest": accrued,
                "foreclosure_charge": charge, "gst_on_charge": gst, "overdue": loan.overdue_amount,
                "late_fee": loan.late_fee, "total_payable": total}

    # ---------- safe writes (no money movement) ----------
    def send_payment_link(self, customer_id: str, account_no: str, amount: int) -> PaymentLink:
        c = self.customers[customer_id]
        link = PaymentLink(link_id=f"PAY-{uuid.uuid4().hex[:8]}", account_no=account_no, amount=amount,
                           sent_to=_mask_phone(c.phone))
        self.payment_links.append(link)
        return link

    def create_ticket(self, customer_id: str, category: str, summary: str, priority: str = "normal") -> Ticket:
        t = Ticket(ticket_id=f"TKT{len(self.tickets) + 10001}", customer_id=customer_id, category=category,
                   summary=summary[:500], priority=priority)
        self.tickets.append(t)
        return t

    def record_promise_to_pay(self, customer_id: str, account_no: str, promised_date: date) -> dict:
        rec = {"customer_id": customer_id, "account_no": account_no, "promised_date": promised_date.isoformat()}
        self.promises.append(rec)
        return rec

    def record_opt_out(self, customer_id: str) -> None:
        c = self.customers[customer_id]
        self.customers[customer_id] = c.model_copy(update={"do_not_call": True})
