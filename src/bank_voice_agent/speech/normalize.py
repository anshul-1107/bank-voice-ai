"""Turn agent text into what a human Indian call-centre executive would SAY.

The LLM writes "₹3,12,640 due on 17 October, loan ending 4821".
TTS engines mangle that ("rupees three hundred twelve thousand..." or "four thousand eight hundred twenty-one").
We verbalise deterministically instead:
  en: "three lakh twelve thousand six hundred forty rupees due on seventeenth October, loan ending four eight two one"
  hi: "teen lakh baarah hazaar chhah sau chaalis rupaye ... satrah October ... chaar aath do ek"
Deterministic = testable = no surprises on 10k calls a day.
"""

from __future__ import annotations

import re

# ---------------- English ----------------
_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
            "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_EN_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
           "twelve": "twelfth"}

# ---------------- Hindi (Roman, as Hinglish speakers read it) ----------------
_HI_0_99 = (
    "shunya ek do teen chaar paanch chhah saat aath nau das gyaarah baarah terah chaudah pandrah solah satrah "
    "athaarah unnees bees ikkees baees teish chaubees pachchees chhabbees sattaees athaaees untees tees ikatees "
    "battees taintees chauntees paintees chhattees saintees adtees untaalees chaalees iktaalees bayaalees "
    "taintaalees chavaalees paintaalees chhiyaalees saintaalees adtaalees unchaas pachaas ikyaavan baavan tirpan "
    "chauvan pachpan chhappan sattaavan atthaavan unsath saath iksath baasath tirsath chausath painsath chhiyaasath "
    "sadsath adsath unhattar sattar ikhattar bahattar tihattar chauhattar pachhattar chhihattar satattar athattar "
    "unaasi assi ikyaasi bayaasi tiraasi chauraasi pachaasi chhiyaasi sattaasi athaasi navaasi nabbe ikyaanave "
    "baanave tiraanave chauraanave pachaanave chhiyaanave sattaanave atthaanave ninyaanave"
).split()
assert len(_HI_0_99) == 100


def _en_below_100(n: int) -> str:
    if n < 20:
        return _EN_ONES[n]
    t, o = divmod(n, 10)
    return _EN_TENS[t] + ("-" + _EN_ONES[o] if o else "")


def _en_below_1000(n: int) -> str:
    h, rest = divmod(n, 100)
    parts = []
    if h:
        parts.append(f"{_EN_ONES[h]} hundred")
    if rest:
        parts.append(_en_below_100(rest))
    return " ".join(parts)


def _hi_below_1000(n: int) -> str:
    h, rest = divmod(n, 100)
    parts = []
    if h:
        parts.append(f"{_HI_0_99[h]} sau")
    if rest:
        parts.append(_HI_0_99[rest])
    return " ".join(parts)


def number_to_words(n: int, lang: str = "en") -> str:
    """Indian system: crore, lakh, thousand (hazaar)."""
    if n == 0:
        return "zero" if lang == "en" else "shunya"
    below100 = _en_below_100 if lang == "en" else (lambda x: _HI_0_99[x])
    below1000 = _en_below_1000 if lang == "en" else _hi_below_1000
    words = {"crore": "crore", "lakh": "lakh", "thousand": "thousand" if lang == "en" else "hazaar"}
    parts = []
    crore, n = divmod(n, 10_000_000)
    lakh, n = divmod(n, 100_000)
    thousand, n = divmod(n, 1000)
    if crore:
        parts.append(f"{number_to_words(crore, lang)} {words['crore']}")
    if lakh:
        parts.append(f"{below100(lakh)} {words['lakh']}")
    if thousand:
        parts.append(f"{below100(thousand)} {words['thousand']}")
    if n:
        parts.append(below1000(n))
    return " ".join(parts)


def ordinal_en(n: int) -> str:
    w = number_to_words(n, "en")
    last = w.split("-")[-1].split(" ")[-1]
    if last in _EN_ORD:
        return w[: len(w) - len(last)] + _EN_ORD[last]
    if last.endswith("y"):
        return w[:-1] + "ieth"
    return w + "th"


def inr(amount: int | float) -> str:
    """₹3,12,640 — Indian digit grouping. Use this everywhere a rupee amount is shown to the LLM."""
    amount = int(round(amount))
    s = str(abs(amount))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", head)
        s = f"{head},{tail}"
    return f"₹{s}"


_DIGIT_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_DIGIT_HI = ["shunya", "ek", "do", "teen", "chaar", "paanch", "chhah", "saat", "aath", "nau"]
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

_RUPEE = re.compile(r"(?:₹|Rs\.?\s?|INR\s?)(\d[\d,]*)(?:\.(\d{1,2}))?(?:\s*(?:rupees|rupaye|rs))?", re.I)
_DATE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTHS})\b", re.I)
_SUFFIX = re.compile(r"\b(ending(?:\s+in|\s+with)?|last\s+four\s+digits\s+(?:are\s+)?|XXXX)\s*(\d{4})\b", re.I)
_CODE = re.compile(r"\b([A-Z]{2,5})(\d{3,})\b")      # ticket ids, link ids
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_NUM = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def _digits(s: str, lang: str) -> str:
    table = _DIGIT_EN if lang == "en" else _DIGIT_HI
    return " ".join(table[int(c)] for c in s if c.isdigit())


def verbalize(text: str, lang: str = "en") -> str:
    """lang: 'en' or 'hi' (Hinglish). Pure function; safe to call on every sentence."""
    lang = "hi" if lang.startswith("hi") else "en"

    def rupee(m: re.Match) -> str:
        whole = int(m.group(1).replace(",", ""))
        paise = int((m.group(2) or "0").ljust(2, "0"))
        unit = "rupees" if lang == "en" else "rupaye"
        out = f"{number_to_words(whole, lang)} {unit}"
        if paise:
            out += f" {'and' if lang == 'en' else 'aur'} {number_to_words(paise, lang)} paise"
        return out

    def date_(m: re.Match) -> str:
        day = int(m.group(1))
        month = m.group(2).capitalize()
        return f"{ordinal_en(day)} {month}" if lang == "en" else f"{_HI_0_99[day]} {month}"

    def suffix(m: re.Match) -> str:
        lead = m.group(1)
        lead = "ending" if lead.upper() == "XXXX" else lead
        return f"{lead} {_digits(m.group(2), lang)}"

    def code(m: re.Match) -> str:
        return " ".join(m.group(1)) + " " + _digits(m.group(2), lang)

    def pct(m: re.Match) -> str:
        whole, _, frac = m.group(1).partition(".")
        out = number_to_words(int(whole), lang)
        if frac:
            out += (" point " if lang == "en" else " dashamlav ") + _digits(frac, lang)
        return out + (" percent" if lang == "en" else " pratishat")

    def num(m: re.Match) -> str:
        raw = m.group(0).replace(",", "")
        if "." in raw:
            whole, frac = raw.split(".")
            return f"{number_to_words(int(whole), lang)} point {_digits(frac, lang)}"
        return number_to_words(int(raw), lang)

    text = _RUPEE.sub(rupee, text)
    text = _DATE.sub(date_, text)
    text = _SUFFIX.sub(suffix, text)
    text = _CODE.sub(code, text)
    text = _PCT.sub(pct, text)
    text = _NUM.sub(num, text)
    # strip anything a TTS would read literally
    text = re.sub(r"[*_#`>|]", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


_HINGLISH_MARKERS = re.compile(
    r"\b(hai|hain|kya|kab|kitna|kitne|mera|meri|mujhe|aap|aapka|nahi|nahin|haan|ji|karna|kar|bata|batao|"
    r"chahiye|paisa|paise|kyun|kaise|abhi|kal|thoda|bhai|accha|achha|theek|sahi|wala|wali|bhej|bhejo|bhejiye|dijiye|kijiye|karo|karein|hoon|hun|raha|rahi|rahe|tha|thi|mein|mujhko|humko|se|ko|ka|ki|ke|aur|lekin|bas|shukriya|dhanyavaad|samajh|pata|kitni|kaun|kahan|jaldi|zaroor|batayiye|bataiye|nahin|matlab|yaar)\b", re.I)


def detect_lang(text: str) -> str:
    """Cheap per-turn language detector for code-mixed speech. 'hi' if >=2 Hinglish markers or any Devanagari."""
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"
    return "hi" if len(_HINGLISH_MARKERS.findall(text)) >= 2 else "en"
