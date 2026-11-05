"""POS domain pack: parse a POS record and check its labels against the structural rules.

The gate's model does the judging; this module is a QA tool. It parses the record format in
docs/labeling_guide.md, derives the facts a rule can decide (restricted flag, ref already
seen, SPQ/MOQ arithmetic, missing or impossible values, non-canonical formats, customer name
identical / format-only / different) and reports labels that contradict them.

It never decides the fuzzy parts (clear vs uncertain rename, whether a garbled row is
"contradictory"); where it cannot read a value it abstains instead of guessing.

Used by `s1gate audit`, by the synthetic-data pipeline (reject teacher rows whose labels
break a rule) and by the procedural generator's self-test.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

REPORTED_KEYS = ["distributor", "period", "customer", "country", "part", "qty", "unit_price", "ref"]
MASTER_KEYS = ["customer", "spq", "moq", "restricted", "seen_refs"]
REQUIRED = ["period", "customer", "country", "part", "qty", "unit_price", "ref"]

# (row_action, needs_human, severity) combinations allowed by the labelling guide, section 5
VALID_COMBOS = {
    ("commit", "no", "0"),
    ("normalize", "no", "1"),
    ("vendor_rename", "no", "2"),
    ("vendor_rename", "yes", "3"),
    ("pack_qty_issue", "yes", "2"),
    ("compliance_review", "yes", "4"),
    ("duplicate", "no", "3"),
    ("ask_human", "yes", "3"),
    ("no_op", "no", "1"),
    ("no_op", "no", "3"),
    ("no_op", "no", "4"),
}
PRECEDENCE = ["compliance_review", "no_op", "ask_human", "duplicate", "pack_qty_issue",
              "vendor_rename", "normalize", "commit"]

MONTHS = {m: i for i, names in enumerate([
    (), ("jan", "january", "ocak", "januar", "janvier", "enero"),
    ("feb", "february", "subat", "februar", "fevrier", "febrero"),
    ("mar", "march", "mart", "marz", "mars", "marzo"),
    ("apr", "april", "nisan", "avril", "abril"),
    ("may", "mayis", "mai", "mayo"),
    ("jun", "june", "haziran", "juni", "juin", "junio"),
    ("jul", "july", "temmuz", "juli", "juillet", "julio"),
    ("aug", "august", "agustos", "aout", "agosto"),
    ("sep", "sept", "september", "eylul", "septembre", "septiembre"),
    ("oct", "october", "ekim", "oktober", "octobre", "octubre"),
    ("nov", "november", "kasim", "novembre", "noviembre"),
    ("dec", "december", "aralik", "dezember", "decembre", "diciembre"),
]) for m in names}

COUNTRY_NAMES = {
    "germany": "DE", "deutschland": "DE", "france": "FR", "netherlands": "NL", "the netherlands": "NL",
    "holland": "NL", "belgium": "BE", "italy": "IT", "italia": "IT", "spain": "ES", "espana": "ES",
    "united kingdom": "GB", "uk": "GB", "great britain": "GB", "ireland": "IE", "austria": "AT",
    "switzerland": "CH", "portugal": "PT", "denmark": "DK", "sweden": "SE", "norway": "NO",
    "finland": "FI", "iceland": "IS", "poland": "PL", "polska": "PL", "czech republic": "CZ",
    "czechia": "CZ", "slovakia": "SK", "hungary": "HU", "romania": "RO", "bulgaria": "BG",
    "slovenia": "SI", "croatia": "HR", "lithuania": "LT", "latvia": "LV", "estonia": "EE",
    "turkey": "TR", "turkiye": "TR", "united arab emirates": "AE", "uae": "AE", "saudi arabia": "SA",
    "israel": "IL", "qatar": "QA", "egypt": "EG", "jordan": "JO", "oman": "OM", "india": "IN",
    "singapore": "SG", "malaysia": "MY", "thailand": "TH", "vietnam": "VN", "viet nam": "VN",
    "indonesia": "ID", "philippines": "PH", "china": "CN", "hong kong": "HK", "taiwan": "TW",
    "japan": "JP", "south korea": "KR", "korea": "KR", "brazil": "BR", "brasil": "BR",
    "mexico": "MX", "argentina": "AR", "chile": "CL", "colombia": "CO", "peru": "PE",
    "united states": "US", "usa": "US", "united states of america": "US", "canada": "CA",
    "russia": "RU", "denmark (dk)": "DK", "korea, republic of": "KR", "republic of korea": "KR",
    "korea (south)": "KR", "the uk": "GB", "england": "GB", "u.s.a.": "US", "u.s.": "US",
    "czech rep.": "CZ", "the philippines": "PH", "hongkong": "HK", "prc": "CN", "uae (dubai)": "AE",
}
SYMBOLS = {"$": "USD", "us$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "₺": "TRY", "¥": "JPY",
           "rs": "INR", "rs.": "INR", "s$": "SGD", "r$": "BRL", "zl": "PLN", "zł": "PLN",
           "euro": "EUR", "euros": "EUR", "tl": "TRY", "dollar": "USD", "dollars": "USD", "yen": "JPY",
           "yuan": "CNY", "rmb": "CNY", "kc": "CZK", "kč": "CZK", "ft": "HUF", "lei": "RON",
           "rupee": "INR", "rupees": "INR", "inr.": "INR", "sar.": "SAR", "aed.": "AED",
           "ca$": "CAD", "c$": "CAD", "a$": "AUD", "au$": "AUD", "hk$": "HKD", "nt$": "TWD",
           "nz$": "NZD", "mx$": "MXN"}
UNITS = r"(?:pcs|pc|pce|pieces|piece|ea|each|units|unit|stk|stück|adet|nos|no|qty|u)\.?"
HEADER_WORDS = {
    "customer", "customer name", "end customer", "end-customer", "client", "sold to", "part",
    "part no", "part no.", "part number", "p/n", "pn", "material", "item", "qty", "quantity",
    "qty.", "unit price", "price", "unit_price", "ref", "reference", "invoice", "invoice no",
    "invoice no.", "inv no", "doc no", "period", "month", "date", "country", "ctry",
}
FILLER = re.compile(r"^[\s\-=_*.~#/\u2500-\u257f\u2014\u2013]*$|^(n/?a|none|null|nil|-+|tbd)$", re.IGNORECASE)
GRAND = re.compile(r"\b(grand[\s-]*total|report[\s-]*total|total[\s-]*(all|report|for report)|overall[\s-]*total|genel[\s-]*toplam|gesamtsumme|gesamttotal|total[\s-]*general|total[\s-]*geral|totale[\s-]*(generale|complessivo)|totaal[\s-]*generaal|summa[\s-]*totalt)\b", re.IGNORECASE)
SUBTOTAL = re.compile(
    r"^\s*(sub[\s-]*total|subtotal|total\b|totals\b|totaal\b|subtotaal|sous[\s-]*total|totale[\s-]*parziale|"
    r"suma[\s-]+cz[eę]ściowa|tarpin[eė][\s-]+suma|междинна[\s-]+сума|ara[\s-]*toplam|toplam\b|zwischensumme|"
    r"summe\b|summa\b|delsum|samtals|v[aä]lisumma|yhteens[aä]|[oö]sszesen|小計|小计|소계|合計|합계)"
    r"|(\s|-)(sub[\s-]*total|total|totaal|toplam[ıi]?|summe|i alt)\s*:?\s*$", re.IGNORECASE)
TRUNCATED = re.compile(r"…|\.\.\.")
NOTE_IN_VALUE = re.compile(r"\?\?|\(see\b|\bsee (email|attachment|mail|note)|\bplease check\b|\bcheck\b|"
                           r"\btbc\b|\bto be confirmed\b|\bverify\b", re.IGNORECASE)
NOTE = re.compile(r"\*{3}|end of (report|file|list)|continued|page \d+|prices? (exclude|include)|^note\b|^remark|^comment|^-{3,}|^={3,}", re.IGNORECASE)


# ---------------------------------------------------------------- parsing

@dataclass
class PosRecord:
    reported: dict[str, str]       # raw values (not stripped), keyed by REPORTED_KEYS
    master: dict[str, str]         # raw values, keyed by MASTER_KEYS
    ok: bool = True                # False when the text is not in the record format
    problems: list[str] = field(default_factory=list)


def _split_fields(segment: str, keys: list[str]) -> tuple[dict[str, str], list[str]]:
    """Find `key=` for each known key in order; a value runs to the next known key.

    Values may contain ';' or '=' as long as they are not followed by another known key.
    """
    out: dict[str, str] = {}
    problems: list[str] = []
    positions = []
    for k in keys:
        m = re.search(rf"(?:^|;)\s*{re.escape(k)}\s*=", segment)
        if not m:
            problems.append(f"missing key {k!r}")
            continue
        positions.append((m.start(), m.end(), k))
    positions.sort()
    for i, (start, end, k) in enumerate(positions):
        stop = positions[i + 1][0] if i + 1 < len(positions) else len(segment)
        value = segment[end:stop]
        if value.endswith(";"):
            value = value[:-1]
        out[k] = value
    return out, problems


def parse(text: str) -> PosRecord:
    if "||" not in text:
        rep, probs = _split_fields(text, REPORTED_KEYS)
        return PosRecord(rep, {}, ok=False, problems=probs + ["no master block"])
    left, right = text.rsplit("||", 1)
    right = re.sub(r"^\s*master\s*:\s*", "", right.strip(), flags=re.IGNORECASE)
    rep, p1 = _split_fields(left.strip(), REPORTED_KEYS)
    mas, p2 = _split_fields(right, MASTER_KEYS)
    return PosRecord(rep, mas, ok=not (p1 or p2), problems=p1 + p2)


# ---------------------------------------------------------------- value readers
# Each returns (value, status). status is one of:
#   canonical | formatted (right value, wrong format) | missing | impossible | not_month |
#   invalid (cannot be read as this kind of value) | unknown (reader abstains)

def read_qty(raw: str) -> tuple[int | None, str]:
    s = raw.strip()
    if not s:
        return None, "missing"
    if re.fullmatch(r"[1-9]\d*", raw):
        return int(raw), "canonical"
    t = s.lower().replace(" ", " ").replace(" ", " ")
    t = re.sub(rf"\s*{UNITS}\s*$", "", t)
    t = re.sub(rf"^{UNITS}\s*[:.]?\s*", "", t)
    if re.fullmatch(r"[-−(].*|.*\)", t):
        return None, "impossible"  # negative / credit
    m = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*k", t)
    if m:
        from decimal import Decimal
        v = Decimal(m.group(1).replace(",", ".")) * 1000  # exact: 16.1k is 16100, not 16100.000000000002
        return (int(v), "formatted") if v == v.to_integral_value() and v > 0 else (None, "impossible")
    t = t.replace(" ", "").replace("'", "").replace("’", "")
    if not re.fullmatch(r"[\d.,]+", t) or not re.search(r"\d", t):
        return None, "invalid"
    value = _read_integer(t)
    if value is None:
        return None, "impossible"  # fractional pieces
    if value == 0:
        return 0, "impossible"
    return value, "formatted"


def _read_integer(t: str) -> int | None:
    """Digits with separators -> int, or None if the number is fractional."""
    if re.fullmatch(r"\d+", t):
        return int(t)
    if "." in t and "," in t:  # the last separator is the decimal one
        dec_sep = "." if t.rfind(".") > t.rfind(",") else ","
        thou_sep = "," if dec_sep == "." else "."
        whole, dec = t.rsplit(dec_sep, 1)
        western = re.fullmatch(rf"\d{{1,3}}(\{thou_sep}\d{{3}})*", whole)
        indian = thou_sep == "," and re.fullmatch(r"\d{1,2}(,\d{2})+,\d{3}", whole)
        if dec.strip("0") or not (western or indian):
            return None
        return int(whole.replace(thou_sep, ""))
    sep = "." if "." in t else ","
    if re.fullmatch(rf"\d{{1,3}}(\{sep}\d{{3}})+", t):
        return int(t.replace(sep, ""))
    if sep == "," and re.fullmatch(r"\d{1,2}(,\d{2})+,\d{3}", t):  # Indian grouping
        return int(t.replace(",", ""))
    if t.count(sep) == 1:
        whole, dec = t.split(sep)
        if whole.isdigit() and dec.isdigit() and not dec.strip("0"):
            return int(whole)
    return None


_CUR = r"(?:[A-Za-z]{1,3}\$|[A-Za-z]{3}\.?|[^\w\s()]{1,2})"
_ACCOUNTING_NEGATIVE = re.compile(rf"\s*(?:{_CUR}\s*)?\(\s*(?:{_CUR}\s*)?\d[\d.,' ]*(?:\s*{_CUR})?\s*\)(?:\s*{_CUR})?\s*")


def read_price(raw: str) -> tuple[float | None, str]:
    s = raw.strip()
    if not s:
        return None, "missing"
    m = re.fullmatch(r"(\d+(?:\.\d+)?) ([A-Z]{3})", raw)
    if m:
        v = float(m.group(1))
        return (v, "canonical") if v > 0 else (v, "impossible")
    t = s.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2212", "-")
    if not re.search(r"\d", t):
        return None, "invalid"  # TBD, n/a, see contract: no number at all
    if re.search(r"\d\s*/\s*\d|\?", t):
        return None, "invalid"  # two prices, or a question inside the value
    if re.search(r"(^|[^\w])-\s*\d", t) or _ACCOUNTING_NEGATIVE.fullmatch(t):
        return None, "impossible"  # negative: -2.31, \u22122.31, (2.31) USD, ($2.31)
    paise = re.fullmatch(r"(?i)\s*(rs\.?|inr\.?|\u20b9)\s*(\d[\d,]*)-(\d{2})\s*", t)
    if paise:  # rupee-paise notation: Rs. 105-00 is 105.00 INR
        whole = paise.group(2).replace(",", "")
        if whole.isdigit():
            v = int(whole) + int(paise.group(3)) / 100
            return (v, "formatted") if v > 0 else (v, "impossible")
    if re.search(r"\d\s*(?:[-\u2013\u2014~]|\bor\b|\bto\b|\bbis\b|\boder\b|\bou\b)\s*\d"
                 r"|\d[.,]\d+\s+\d+[.,]\d|\b(approx|approximately|circa|about|around|estimated|est)\b\.?"
                 r"|\bca\.|[~\u2248]", t, re.IGNORECASE):
        return None, "invalid"  # a range, two prices, or a qualified (not exact) price
    # split a symbol from the number ($2.31, 2,31€) but keep letter-prefixed ones whole (US$, R$, S$)
    t = re.sub(r"(?<![A-Za-z])([^\w\s.,'-])(?=\d)|(?<=\d)([^\w\s.,'-])", lambda m: f" {m.group(0)} ", t)
    t = re.sub(r"(?i)\b(us|r|s|ca|c|a|au|hk|nt|nz|mx)\$(?=\d)", lambda m: f"{m.group(0)} ", t)
    if len(re.findall(r"\d[\d.,' ]*\d|\d", t)) > 1:
        return None, "invalid"  # two numbers: '2.31 USD or 2.40 USD', '$2.31/$2.40', '2.31 USD, 2.40 USD'
    num_m = re.search(r"\d[\d.,' ]*\d|\d", t)
    num = re.sub(r"[ ']", "", num_m.group(0))
    rest = (t[:num_m.start()] + " " + t[num_m.end():]).strip()
    value = _read_decimal(num)
    if value is not None and value <= 0:
        return value, "impossible"  # zero, whatever the currency notation
    tokens = [x for x in re.split(r"\s+", rest) if x]
    codes = []
    for tok in tokens:
        key = tok.lower()
        if key in SYMBOLS:
            codes.append(SYMBOLS[key])
        elif re.fullmatch(r"[A-Za-z]{3}", tok):
            codes.append(tok.upper())
        elif re.fullmatch(r"[^\w\s]{1,2}", tok) and tok in SYMBOLS:
            codes.append(SYMBOLS[tok])
        else:
            return None, "unknown"  # words we cannot read as a currency: abstain
    value = _read_decimal(num)
    if value is None:
        return None, "unknown"
    if value <= 0:
        return value, "impossible"
    if not codes or len(set(codes)) > 1:
        return None, "unknown"  # no currency, or two different ones: no rule, abstain
    return value, "formatted"


def _read_decimal(t: str) -> float | None:
    neg = t.startswith("-")
    t = t.lstrip("-")
    if not t or not re.fullmatch(r"[\d.,]+", t):
        return None
    if "." in t and "," in t:
        dec_sep = "." if t.rfind(".") > t.rfind(",") else ","
        thou = "," if dec_sep == "." else "."
        t = t.replace(thou, "").replace(dec_sep, ".")
    elif "," in t:
        whole, _, dec = t.rpartition(",")
        # "1,250" is a thousands separator; "0,223" or "2,31" is a decimal comma
        thousands = len(dec) == 3 and whole and not whole.startswith("0")
        t = t.replace(",", "") if thousands else whole.replace(",", "") + "." + dec
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def read_period(raw: str) -> tuple[tuple[int, int] | None, str]:
    s = raw.strip()
    if not s:
        return None, "missing"
    m = re.fullmatch(r"(\d{4})-(\d{2})", raw)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return ((y, mo), "canonical") if 1 <= mo <= 12 else (None, "impossible")
    t = _fold_keep(s)
    if re.search(r"\bq[1-4]\b|\bh[12]\b|\bfy\s*\d{2,4}\b|^\d{4}$|\bquarter\b|\bytd\b", t) or re.fullmatch(r"[qh][1-4]\s*\d{2,4}|\d{2,4}\s*[qh][1-4]", t):
        return None, "not_month"
    nums = [int(x) for x in re.findall(r"\d+", t)]
    words = [w for w in re.findall(r"[a-z]+", t)]
    month_words = [MONTHS[w] for w in words if w in MONTHS]
    if month_words:
        years = [n for n in nums if n >= 1990 or (len(str(n)) == 2)]
        return ((years[-1] if years else 0, month_words[0]), "formatted") if years else (None, "unknown")
    if re.fullmatch(r"\d{4}[/.\-]\d{1,2}([/.\-]\d{1,2})?", t):  # YYYY/MM, YYYY-M, YYYY-MM-DD
        y, mo = nums[0], nums[1]
        if len(nums) == 3 and not 1 <= nums[2] <= 31:
            return None, "impossible"
        return ((y, mo), "formatted") if 1 <= mo <= 12 else (None, "impossible")
    if re.fullmatch(r"\d{1,2}[/.\-]\d{4}", t):  # MM/YYYY
        mo, y = nums
        return ((y, mo), "formatted") if 1 <= mo <= 12 else (None, "impossible")
    if re.fullmatch(r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}", t):  # DD.MM.YYYY or MM/DD/YYYY
        a, b, y = nums
        if "." in t or a > 12:  # dotted dates are day-first in every region we see
            day, mo = a, b
        elif b > 12:
            day, mo = b, a
        else:
            return None, "unknown"  # 03/04/2026: day-first or month-first, abstain
        if not 1 <= day <= 31 or not 1 <= mo <= 12:
            return None, "impossible"
        return (y, mo), "formatted"
    if re.fullmatch(r"\d{6}", t):  # 202609
        y, mo = int(t[:4]), int(t[4:])
        return ((y, mo), "formatted") if 1 <= mo <= 12 else (None, "impossible")
    return None, "unknown"


def read_part(raw: str, catalogue: set[str] | None = None) -> tuple[str | None, str]:
    s = raw.strip()
    if not s:
        return None, "missing"
    if re.search(r"\s/\s|\bor\b|\(|\?|,|;|\+", s):
        return None, "invalid"  # two parts, notes, question marks
    canonical = re.fullmatch(r"[A-Z0-9]+(-[A-Z0-9]+)*", raw) is not None
    if catalogue:
        if raw in catalogue:
            return raw, "canonical"
        folded = re.sub(r"[^A-Za-z0-9]", "", s).upper()
        for p in catalogue:
            if re.sub(r"[^A-Z0-9]", "", p) == folded:
                return p, "formatted"  # case, spaces or hyphens differ from the catalogue
        return (raw, "canonical") if canonical else (None, "unknown")
    if canonical:
        return raw, "canonical"
    if re.fullmatch(r"[A-Za-z0-9]+([ \-_][A-Za-z0-9]+)*", s):
        return s.upper().replace(" ", "-").replace("_", "-"), "formatted"
    return None, "unknown"


def read_country(raw: str) -> tuple[str | None, str]:
    s = raw.strip()
    if not s:
        return None, "missing"
    if re.fullmatch(r"[A-Z]{2}", raw):
        return raw, "canonical"
    if re.fullmatch(r"[A-Za-z]{2}", s):
        return s.upper(), "formatted"
    name = _fold(s)
    if name in COUNTRY_NAMES:
        return COUNTRY_NAMES[name], "formatted"
    return None, "unknown"


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    for a, b in (("ı", "i"), ("ł", "l"), ("ø", "o"), ("æ", "ae"), ("ß", "ss"), ("đ", "d"), ("ð", "d"),
                 ("þ", "th"), ("œ", "oe"), ("ħ", "h")):
        s = s.replace(a, b)
    return s


def _fold_keep(s: str) -> str:
    """Lower case and accents stripped, punctuation kept (for dates)."""
    return re.sub(r"\s+", " ", _strip_accents(s.lower())).strip()


def _fold(s: str) -> str:
    """Lower case, accents stripped, punctuation to spaces, whitespace collapsed."""
    s = re.sub(r"[^\w]+", " ", _strip_accents(s.lower()))
    return re.sub(r"\s+", " ", s).strip()


def name_key(s: str) -> str:
    """Letters and digits only, lower case, accents stripped: 'A.S.' and 'AS' share a key."""
    return re.sub(r"[\W_]+", "", _strip_accents(s.lower()))


def name_relation(reported: str, master: str) -> str:
    """identical | format_only (case, spacing, punctuation, accents) | different | missing."""
    if not reported.strip():
        return "missing"
    if unicodedata.normalize("NFC", reported) == unicodedata.normalize("NFC", master):
        return "identical"
    if name_key(reported) == name_key(master) and name_key(master):
        return "format_only"
    return "different"


def norm_ref(ref: str) -> str:
    return re.sub(r"\s+", "", ref).upper()


def seen_list(raw: str) -> list[str]:
    s = raw.strip()
    if not s or s.lower() in {"none", "-", "n/a", "[]"}:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


# words that say nothing about which company a name means (legal forms, filler)
_NAME_FILLER = {"ltd", "limited", "inc", "incorporated", "corp", "corporation", "company", "llc", "llp",
                "plc", "gmbh", "sas", "sarl", "srl", "spa", "pvt", "pte", "bhd", "sdn", "jsc", "ooo", "sapi",
                "the", "and", "und", "group", "holding", "holdings", "intl", "international"}


def _fold(s: str) -> str:
    return re.sub(r"[\s\-_]+", " ", _strip_accents(s.lower())).strip()


def _marker_key(raw: str) -> str:
    raw = raw.strip()
    if re.search(r"[^\W_]", raw):
        return _fold(raw).strip(" -:")  # 'Sub-Total', 'sub total' and 'subtotal:' alike
    return unicodedata.normalize("NFKC", raw)  # punctuation stays as it is: '---', '??', '...'


def _count(marker: str, text: str) -> int:
    if not re.search(r"[^\W_]", marker):
        return unicodedata.normalize("NFKC", text.lower()).count(marker)
    return len(re.findall(r"(?<![^\W_])" + re.escape(marker) + r"(?![^\W_])", _fold(text)))


def _name_words(s: str) -> set[str]:
    return {w for w in re.split(r"[\W_]+", _fold(s)) if len(w) > 2 and w not in _NAME_FILLER}


def new_marker(pattern: re.Pattern, value: str, master: str) -> bool:
    """`value` carries a match of `pattern` that the master customer name does not explain.

    Read the row, not the word: a marker word inside the real name ('Total Power Solutions',
    'Seguridad Total', 'Check Point Software', 'TBC Corp') is the name, not a subtotal or a note.
    It still counts as a marker when
      * the value is nothing but the marker ('Total', 'Subtotal:'), or there is no master name;
      * it occurs more often than in the master name ('Acme Ltd Total', 'TBC Corp (TBC)',
        'Total Power Solutions Ltd Total'); for a name with the master's very letters (a case,
        spacing or punctuation variant) only an added '??' or '...' can be new;
      * the master name holds it but the value shares no other distinctive word with that name
        ('Note: credit to follow' when the lookup matched 'Note AB', 'Total Acme Controls Ltd'
        when it matched 'Total Power Solutions Ltd').
    """
    found = {_marker_key(m.group(0)) for m in pattern.finditer(value)} - {""}
    if not found:
        return False
    if not re.search(r"[^\W_]", pattern.sub(" ", value)):
        return True
    master = master.strip()
    if not master or master.lower() in ("unknown", "?"):
        return True
    if name_key(value) == name_key(master):
        return any(_count(f, value) > _count(f, master) for f in found if not re.search(r"[^\W_]", f))
    if any(_count(f, value) > _count(f, master) for f in found):
        return True
    marker_words = {w for f in found for w in re.split(r"[\W_]+", f) if w}
    own, theirs = _name_words(value) - marker_words, _name_words(master) - marker_words
    return bool(own) and not (own & theirs)


def nonsales_kind(rec: PosRecord, catalogue: set[str] | None = None) -> str | None:
    """header | blank | note | subtotal | grand_total | None (looks like a sales line)."""
    r = {k: rec.reported.get(k, "").strip() for k in REPORTED_KEYS}
    content = [r[k] for k in ("customer", "part", "qty", "unit_price", "ref", "period", "country")]
    headerish = sum(1 for v in content if v.lower().rstrip(":") in HEADER_WORDS)
    if headerish >= 3:
        return "header"
    if all(FILLER.match(v) for v in (r["customer"], r["part"], r["qty"], r["unit_price"], r["ref"])):
        return "blank"
    # A total or note marker in the customer field is vetoed by sales-line evidence: a real ref
    # plus a catalogue-style part means a customer that happens to be called "Total ..." .
    ref_real = bool(r["ref"]) and not FILLER.match(r["ref"]) and not (GRAND.search(r["ref"]) or SUBTOTAL.search(r["ref"]))
    part_real = bool(r["part"]) and read_part(r["part"], catalogue)[1] in ("canonical", "formatted") \
        and not (GRAND.search(r["part"]) or SUBTOTAL.search(r["part"]))
    sales = ref_real and part_real
    mc = rec.master.get("customer", "")
    if GRAND.search(r["part"]) or GRAND.search(r["ref"]) or (new_marker(GRAND, r["customer"], mc) and not sales):
        return "grand_total"
    if SUBTOTAL.search(r["part"]) or SUBTOTAL.search(r["ref"]) or \
            (new_marker(SUBTOTAL, r["customer"], mc) and not sales):
        return "subtotal"
    if (not r["qty"] or FILLER.match(r["qty"])) and not sales and \
            (new_marker(NOTE, r["customer"], mc) or any(NOTE.search(r[k]) for k in ("part", "ref", "period"))):
        return "note"
    return None


# ---------------------------------------------------------------- facts and audit

@dataclass
class Facts:
    parsed: bool
    restricted: str | None          # yes | no | None
    nonsales: str | None
    lookup_failed: bool
    defects: list[str]              # reasons the row cannot be trusted (ask_human evidence)
    ref_seen: bool
    pack_ok: bool | None            # None = cannot tell
    qty: int | None
    nonc: list[str]                 # fields that are right but not canonical
    customer: str                   # identical | format_only | different | missing | unknown
    problems: list[str]
    unread: list[str] = field(default_factory=list)   # not canonical, and the reader abstains
    sales_evidence: bool = False    # catalogue part + readable qty + a ref, and nothing else wrong


def facts(text: str, catalogue: set[str] | None = None) -> Facts:
    rec = parse(text)
    rep = {k: rec.reported.get(k, "") for k in REPORTED_KEYS}
    mas = {k: rec.master.get(k, "").strip() for k in MASTER_KEYS}
    restricted = mas["restricted"].lower() if mas["restricted"].lower() in ("yes", "no") else None
    lookup_failed = (mas["customer"].lower() in ("unknown", "?", "")
                     or mas["spq"] in ("?", "") or mas["moq"] in ("?", ""))
    defects: list[str] = []
    nonc: list[str] = []

    for k in REQUIRED:
        v = rep[k].strip()
        if k == "country" and re.fullmatch(r"[A-Z]{2}", v):
            continue  # NA is Namibia, not "n/a"
        if not v or FILLER.match(v):  # '-', 'n/a', 'none' are empty too
            defects.append(f"{k} missing")
    unread: list[str] = []
    qty, qs = read_qty(rep["qty"])
    if qs in ("impossible", "invalid"):
        defects.append(f"qty {qs}: {rep['qty'].strip()!r}")
    elif qs == "formatted":
        nonc.append("qty")
    price, ps = read_price(rep["unit_price"])
    if ps in ("impossible", "invalid"):
        defects.append(f"unit_price {ps}: {rep['unit_price'].strip()!r}")
    elif ps == "formatted":
        nonc.append("unit_price")
    elif ps == "unknown":
        unread.append("unit_price")
    _, pe = read_period(rep["period"])
    if pe in ("impossible", "not_month"):
        defects.append(f"period {pe}: {rep['period'].strip()!r}")
    elif pe == "formatted":
        nonc.append("period")
    elif pe == "unknown":
        unread.append("period")
    _, pa = read_part(rep["part"], catalogue)
    if pa == "invalid":
        defects.append(f"part unreadable: {rep['part'].strip()!r}")
    elif pa == "formatted":
        nonc.append("part")
    elif pa == "unknown":
        unread.append("part")
    _, co = read_country(rep["country"])
    if co == "formatted":
        nonc.append("country")
    elif co == "unknown":
        unread.append("country")
    if lookup_failed:
        defects.append("master lookup failed")
    customer = "unknown" if lookup_failed else name_relation(rep["customer"], mas["customer"])
    for k in ("period", "customer", "country", "part", "qty", "unit_price", "ref"):
        v = rep[k].strip()
        # in the customer field a marker that is part of the master name ("Check Point ...",
        # "TBC Corp") is the name; one added to it ("Acme Ltd ??") is a note
        has = (lambda pat: new_marker(pat, v, mas["customer"])) if k == "customer" else (lambda pat: bool(pat.search(v)))
        if has(TRUNCATED):
            defects.append(f"{k} truncated: {v!r}")
        elif has(NOTE_IN_VALUE):
            defects.append(f"note inside {k}: {v!r}")
    v = rep["ref"]
    if v.strip() and (v != v.strip() or "  " in v):
        nonc.append("ref whitespace")  # the distributor's own name is never an issue
    if customer == "format_only":
        nonc.append("customer")

    ref = rep["ref"].strip()
    ref_seen = bool(ref) and norm_ref(ref) in {norm_ref(x) for x in seen_list(mas["seen_refs"])}

    pack_ok = None
    if qty and qs in ("canonical", "formatted") and re.fullmatch(r"[0-9]+", mas["spq"]) \
            and re.fullmatch(r"[0-9]+", mas["moq"]):
        spq, moq = int(mas["spq"]), int(mas["moq"])
        if spq > 0:
            pack_ok = qty % spq == 0 and qty >= moq
    # the reported side decides whether this is a sales line (rule 2 comes before rule 3): a failed
    # lookup on an otherwise complete line is a sales line with a lookup problem (ask_human)
    customer_only = ("master lookup failed", "note inside customer", "customer truncated")
    sales_evidence = (pa in ("canonical", "formatted") and qs in ("canonical", "formatted")
                      and bool(ref) and not FILLER.match(ref)
                      and (not lookup_failed or all(d.startswith(customer_only) for d in defects)))

    return Facts(rec.ok, restricted, nonsales_kind(rec, catalogue), lookup_failed, defects, ref_seen,
                 pack_ok, qty, nonc, customer, rec.problems, unread, sales_evidence)


@dataclass
class Finding:
    level: str      # violation | warning
    rule: str
    message: str


def check_labels(text: str, labels: dict[str, str], catalogue: set[str] | None = None) -> list[Finding]:
    """Labels that contradict a structural rule (violation) or look odd (warning)."""
    out: list[Finding] = []
    f = facts(text, catalogue)
    act, nh, sev = labels.get("row_action"), labels.get("needs_human"), labels.get("severity")

    def v(rule, msg):
        out.append(Finding("violation", rule, msg))

    def w(rule, msg):
        out.append(Finding("warning", rule, msg))

    if not f.parsed:
        w("format", "record not in the expected format: " + "; ".join(f.problems))
    if act and nh and sev and (act, nh, sev) not in VALID_COMBOS:
        v("combo", f"{act}/{nh}/{sev} is not one of the eleven valid combinations")
    if act is None:
        return out

    # 1 compliance
    if f.restricted == "yes" and act != "compliance_review":
        v("compliance", f"restricted=yes but row_action={act}")
    if act == "compliance_review" and f.restricted != "yes":
        v("compliance", f"row_action=compliance_review but restricted={f.restricted}")
    if f.restricted == "yes":
        return out  # nothing below can change a compliance row

    # 2 non-sales lines
    if f.nonsales and act != "no_op":
        v("no_op", f"row looks like a {f.nonsales} line but row_action={act}")
    if act == "no_op":
        if not f.nonsales and f.sales_evidence:
            v("no_op", "row_action=no_op but the row is a sales line (catalogue part, quantity, ref)")
        elif not f.nonsales:
            w("no_op", "row_action=no_op but no header/blank/note/subtotal/total marker found")
        expected = {"header": "1", "blank": "1", "note": "1", "subtotal": "3", "grand_total": "4"}
        if f.nonsales in expected and sev and sev != expected[f.nonsales]:
            w("no_op", f"{f.nonsales} line usually has severity {expected[f.nonsales]}, got {sev}")
        return out

    # 3 untrustworthy sales lines
    if f.defects and act != "ask_human":
        v("ask_human", f"row has defects ({'; '.join(f.defects)}) but row_action={act}")
    if act == "ask_human":
        if not f.defects and f.pack_ok is not None:
            w("ask_human", "row_action=ask_human but no missing, impossible or contradictory value found")
        return out

    # 4 duplicates
    if f.ref_seen and act != "duplicate":
        v("duplicate", f"ref is in seen_refs but row_action={act}")
    if act == "duplicate":
        if not f.ref_seen:
            v("duplicate", "row_action=duplicate but ref is not in seen_refs")
        return out

    # 5 pack quantities
    if act == "pack_qty_issue" and f.pack_ok is True:
        v("pack_qty", f"row_action=pack_qty_issue but qty {f.qty} is a valid SPQ multiple >= MOQ")
    if act != "pack_qty_issue" and f.pack_ok is False:
        v("pack_qty", f"qty {f.qty} breaks SPQ/MOQ but row_action={act}")
    if act == "pack_qty_issue":
        return out

    # 6-8 customer name and formats
    if act == "vendor_rename" and f.customer in ("identical", "format_only"):
        v("vendor_rename", f"row_action=vendor_rename but the customer name is {f.customer.replace('_', ' ')}")
    if act in ("commit", "normalize") and f.customer == "different":
        v("vendor_rename", f"customer name differs from master beyond formatting but row_action={act}")
    if act == "commit" and (f.nonc or f.unread):
        v("normalize", f"row_action=commit but not canonical: {', '.join(f.nonc + f.unread)}")
    if act == "normalize" and not f.nonc and not f.unread:
        v("normalize", "row_action=normalize but every value is canonical")
    return out


def load_catalogue(path: str | None) -> set[str] | None:
    if not path:
        return None
    import yaml

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return {p["part"] for p in data.get("parts", [])}
