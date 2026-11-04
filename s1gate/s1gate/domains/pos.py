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
    "russia": "RU", "denmark (dk)": "DK",
}
SYMBOLS = {"$": "USD", "us$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "₺": "TRY", "¥": "JPY",
           "rs": "INR", "rs.": "INR", "s$": "SGD", "r$": "BRL", "zl": "PLN", "zł": "PLN"}
UNITS = r"(?:pcs|pc|pce|pieces|piece|ea|each|units|unit|stk|stück|adet|nos|no\.?|qty|u)"
HEADER_WORDS = {
    "customer", "customer name", "end customer", "end-customer", "client", "sold to", "part",
    "part no", "part no.", "part number", "p/n", "pn", "material", "item", "qty", "quantity",
    "qty.", "unit price", "price", "unit_price", "ref", "reference", "invoice", "invoice no",
    "invoice no.", "inv no", "doc no", "period", "month", "date", "country", "ctry",
}
FILLER = re.compile(r"^[\s\-=_*.~#/]*$|^(n/?a|none|null|nil|-+|tbd)$", re.IGNORECASE)
GRAND = re.compile(r"\b(grand[\s-]*total|report[\s-]*total|total[\s-]*(all|report|for report)|overall[\s-]*total|genel[\s-]*toplam|gesamtsumme)\b", re.IGNORECASE)
SUBTOTAL = re.compile(r"^\s*(sub[\s-]*total|subtotal|total\b|totals\b|ara[\s-]*toplam|zwischensumme)", re.IGNORECASE)
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
        v = float(m.group(1).replace(",", ".")) * 1000
        return (int(v), "formatted") if v == int(v) and v > 0 else (None, "impossible")
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
        if dec.strip("0") or not re.fullmatch(rf"\d{{1,3}}(\{thou_sep}\d{{3}})*", whole):
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


def read_price(raw: str) -> tuple[float | None, str]:
    s = raw.strip()
    if not s:
        return None, "missing"
    m = re.fullmatch(r"(\d+(?:\.\d+)?) ([A-Z]{3})", raw)
    if m:
        v = float(m.group(1))
        return (v, "canonical") if v > 0 else (v, "impossible")
    t = s.replace(" ", " ")
    cur = None
    num = None
    m = re.fullmatch(r"([A-Za-z]{3}|[^\d\s\-.,]{1,3}\.?)\s*(-?[\d.,]+)", t)
    if m:
        cur, num = m.group(1), m.group(2)
    else:
        m = re.fullmatch(r"(-?[\d.,]+)\s*([A-Za-z]{3}|[^\d\s\-.,]{1,3})?", t)
        if m:
            num, cur = m.group(1), m.group(2)
    if num is None:
        return None, "invalid"
    if cur is None:
        return None, "unknown"  # a bare number: no rule for a missing currency, abstain
    if cur.lower() in SYMBOLS:
        cur = SYMBOLS[cur.lower()]
    if not re.fullmatch(r"[A-Za-z]{3}", cur):
        return None, "unknown"
    value = _read_decimal(num)
    if value is None:
        return None, "invalid"
    if value <= 0:
        return value, "impossible"
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
    return s.replace("ı", "i").replace("ł", "l").replace("ø", "o").replace("æ", "ae").replace("ß", "ss")


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
    if reported == master:
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


def nonsales_kind(rec: PosRecord) -> str | None:
    """header | blank | note | subtotal | grand_total | None (looks like a sales line)."""
    r = {k: rec.reported.get(k, "").strip() for k in REPORTED_KEYS}
    content = [r[k] for k in ("customer", "part", "qty", "unit_price", "ref", "period", "country")]
    headerish = sum(1 for v in content if v.lower().rstrip(":") in HEADER_WORDS)
    if headerish >= 3:
        return "header"
    if all(FILLER.match(v) for v in (r["customer"], r["part"], r["qty"], r["unit_price"], r["ref"])):
        return "blank"
    if any(GRAND.search(r[k]) for k in ("customer", "part", "ref")):
        return "grand_total"
    if any(SUBTOTAL.match(r[k]) for k in ("customer", "part", "ref")):
        return "subtotal"
    if (not r["qty"] or FILLER.match(r["qty"])) and any(NOTE.search(r[k]) for k in ("customer", "part", "ref", "period")):
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
        if not rep[k].strip() or FILLER.match(rep[k].strip()):  # '-', 'n/a', 'none' are empty too
            defects.append(f"{k} missing")
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
    _, pe = read_period(rep["period"])
    if pe in ("impossible", "not_month"):
        defects.append(f"period {pe}: {rep['period'].strip()!r}")
    elif pe == "formatted":
        nonc.append("period")
    _, pa = read_part(rep["part"], catalogue)
    if pa == "invalid":
        defects.append(f"part unreadable: {rep['part'].strip()!r}")
    elif pa == "formatted":
        nonc.append("part")
    _, co = read_country(rep["country"])
    if co == "formatted":
        nonc.append("country")
    if lookup_failed:
        defects.append("master lookup failed")
    for k in ("period", "customer", "country", "part", "qty", "unit_price", "ref"):
        v = rep[k].strip()
        if TRUNCATED.search(v):
            defects.append(f"{k} truncated: {v!r}")
        elif NOTE_IN_VALUE.search(v):
            defects.append(f"note inside {k}: {v!r}")
    for k in ("customer", "ref"):  # the distributor's own name is never an issue
        v = rep[k]
        if v.strip() and (v != v.strip() or "  " in v):
            nonc.append(f"{k} whitespace")

    customer = "unknown" if lookup_failed else name_relation(rep["customer"], mas["customer"])
    if customer == "format_only":
        nonc.append("customer")

    ref = rep["ref"].strip()
    ref_seen = bool(ref) and norm_ref(ref) in {norm_ref(x) for x in seen_list(mas["seen_refs"])}

    pack_ok = None
    if qty and qs in ("canonical", "formatted") and mas["spq"].isdigit() and mas["moq"].isdigit():
        spq, moq = int(mas["spq"]), int(mas["moq"])
        if spq > 0:
            pack_ok = qty % spq == 0 and qty >= moq

    return Facts(rec.ok, restricted, nonsales_kind(rec), lookup_failed, defects, ref_seen,
                 pack_ok, qty, nonc, customer, rec.problems)


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
        if not f.nonsales:
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
    if act == "commit" and f.nonc:
        v("normalize", f"row_action=commit but non-canonical: {', '.join(f.nonc)}")
    if act == "normalize" and not f.nonc:
        v("normalize", "row_action=normalize but every value is canonical")
    return out


def load_catalogue(path: str | None) -> set[str] | None:
    if not path:
        return None
    import yaml

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return {p["part"] for p in data.get("parts", [])}
