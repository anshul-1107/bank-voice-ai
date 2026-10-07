from datetime import date

from bank_voice_agent.agent.facts import FactLedger
from bank_voice_agent.qa.guardrails import check_sentence
from bank_voice_agent.qa.record import redact
from bank_voice_agent.speech.normalize import detect_lang, inr, number_to_words, verbalize


# ---------------------------------------------------------------- speech
def test_indian_grouping():
    assert inr(312640) == "₹3,12,640"
    assert inr(10_000_000) == "₹1,00,00,000"
    assert inr(999) == "₹999"


def test_lakh_crore_words():
    assert number_to_words(312640) == "three lakh twelve thousand six hundred forty"
    assert number_to_words(12_450, "hi") == "baarah hazaar chaar sau pachaas"
    assert number_to_words(25_000_000) == "two crore fifty lakh"


def test_verbalize_banking_sentence():
    out = verbalize("EMI of ₹12,450 for loan ending 4821 is due on 17 October.", "en")
    assert out == ("EMI of twelve thousand four hundred fifty rupees for loan ending four eight two one "
                   "is due on seventeenth October.")
    hi = verbalize("Aapka EMI ₹4,120 hai, 3 November ko.", "hi")
    assert "chaar hazaar ek sau bees rupaye" in hi and "teen November" in hi


def test_language_detection():
    assert detect_lang("mera EMI kab hai") == "hi"
    assert detect_lang("achha link bhej dijiye") == "hi"
    assert detect_lang("when is my EMI due") == "en"


# ---------------------------------------------------------------- guardrails
def ledger_with_loan():
    led = FactLedger(today=date(2026, 10, 7))
    led.add({"emi_amount": "₹12,450", "next_due_date": "2026-10-10", "loan": "Personal Loan ending 4821"})
    return led


def test_grounded_sentence_passes():
    g = check_sentence("Your EMI of ₹12,450 is due on 10 October for the loan ending 4821.", ledger_with_loan(), True)
    assert g.ok, g.violations


def test_invented_amount_blocked():
    g = check_sentence("Your EMI is ₹12,500.", ledger_with_loan(), True)
    assert g.blocked and g.violations[0].code == "ungrounded_amount"
    assert "12,500" not in g.text


def test_invented_date_blocked():
    g = check_sentence("It is due on 15 October.", ledger_with_loan(), True)
    assert g.blocked and any(v.code == "ungrounded_date" for v in g.violations)


def test_account_digits_before_verification_blocked():
    g = check_sentence("I see your loan ending 4821.", ledger_with_loan(), verified=False)
    assert g.blocked and any(v.code == "unverified_disclosure" for v in g.violations)


def test_asking_for_otp_blocked_but_warning_allowed():
    led = FactLedger()
    assert check_sentence("Please tell me the OTP you received.", led, True).blocked
    assert check_sentence("Can you share your PIN?", led, True).blocked
    assert check_sentence("Please don't share your OTP with anyone, including us.", led, True).ok
    assert check_sentence("We never ask for your PIN or CVV.", led, True).ok


def test_coercion_and_false_promises_blocked():
    led = FactLedger()
    assert check_sentence("If you don't pay we will inform your employer.", led, True).blocked
    assert check_sentence("Legal action will start tomorrow.", led, True).blocked
    assert check_sentence("Don't worry, I will waive the late fee.", led, True).blocked
    assert check_sentence("This plan gives guaranteed returns.", led, True).blocked


def test_claiming_human_blocked():
    assert check_sentence("Yes, I am a real human.", FactLedger(), True).blocked
    assert check_sentence("I'm the bank's AI assistant.", FactLedger(), True).ok


def test_caller_numbers_can_be_repeated():
    led = FactLedger()
    led.add_text("I can pay ₹5,000 by Friday")
    assert check_sentence("Okay, ₹5,000 by Friday, noted.", led, True).ok


def test_redaction():
    r = redact("my dob is 14 May 1990, pan ABCDE1234F, phone 9876543210, otp 482913")
    assert "1990" not in r and "ABCDE1234F" not in r and "9876543210" not in r and "482913" not in r
