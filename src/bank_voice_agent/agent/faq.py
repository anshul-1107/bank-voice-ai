"""Approved answers for general questions. The agent may only answer policy questions from here.
Replace with the bank's compliance-approved content; keep each answer short enough to speak in ~15 seconds."""

FAQ: dict[str, str] = {
    "emi_bounce_charges": "If an EMI bounces, a bounce charge of ₹500 plus GST applies, and late payment interest is "
                          "charged on the overdue amount until it is paid.",
    "credit_score_impact": "Payments that are late are reported to credit bureaus such as CIBIL, which can lower the "
                           "credit score. Paying the overdue amount as soon as possible limits the impact.",
    "nach_autopay": "With NACH auto-debit, the EMI is debited from the linked bank account on the due date. Keeping "
                    "enough balance a day before the due date avoids a bounce.",
    "change_emi_date": "EMI date changes need a request through the branch or the app, and are reviewed by the loans "
                       "team. I can raise a request for you.",
    "foreclosure_process": "You can close the loan early by paying the foreclosure amount. I can give you a quote valid "
                           "for 7 days and send a payment link. Floating-rate home loans have no foreclosure charge.",
    "part_prepayment": "Part prepayment is allowed after 6 EMIs are paid. It reduces either the EMI or the tenure, as you "
                       "choose, through the app or branch.",
    "insurance_grace_period": "Most policies have a grace period after the renewal date, usually 30 days for yearly "
                              "premiums. Cover can be affected if the premium is not paid within the grace period, so "
                              "renewing before the due date is safest.",
    "kyc_update": "KYC can be updated in the app under Profile, or at any branch with your PAN and Aadhaar.",
    "branch_hours": "Branches are open Monday to Saturday, 10 AM to 4 PM, except the second and fourth Saturday.",
    "fraud_warning": "We never ask for OTP, PIN, CVV or passwords on a call, SMS or email. If someone asks, please "
                     "hang up and report it on our helpline.",
    "grievance": "You can raise a complaint with me now, or write to our grievance officer. If unresolved in 30 days, "
                 "you can approach the RBI Ombudsman at cms.rbi.org.in.",
}

TOPICS = sorted(FAQ)
