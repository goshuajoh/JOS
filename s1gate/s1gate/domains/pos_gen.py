"""Procedural POS rows whose labels are known by construction (step 3, second data source).

The teacher LLM gives natural variety; this generator gives exactness and volume where the
answer is arithmetic or bookkeeping: SPQ/MOQ boundaries, refs one digit away from a seen
ref, restricted flags on otherwise clean rows, precedence between stacked problems, and the
hard pairs of the labelling guide, each built as two rows that differ in one field.

Every row is built from a clean canonical row plus one recipe of mutations, then checked
with `pos.check_labels`; a row with any violation is discarded and counted, never kept.

    s1gate procgen --pools data/pools/train/companies.json --out data/synth/procedural.jsonl
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from ..records import Record, stable_id
from . import pos


class UnknownRecipe(KeyError):
    """The recipe names a scenario or pair id this generator has no procedure for."""


COMMIT = {"row_action": "commit", "needs_human": "no", "severity": "0"}
NORMALIZE = {"row_action": "normalize", "needs_human": "no", "severity": "1"}
RENAME = {"row_action": "vendor_rename", "needs_human": "no", "severity": "2"}
RENAME_UNSURE = {"row_action": "vendor_rename", "needs_human": "yes", "severity": "3"}
PACK = {"row_action": "pack_qty_issue", "needs_human": "yes", "severity": "2"}
COMPLIANCE = {"row_action": "compliance_review", "needs_human": "yes", "severity": "4"}
DUPLICATE = {"row_action": "duplicate", "needs_human": "no", "severity": "3"}
ASK = {"row_action": "ask_human", "needs_human": "yes", "severity": "3"}
NOOP_1 = {"row_action": "no_op", "needs_human": "no", "severity": "1"}
NOOP_3 = {"row_action": "no_op", "needs_human": "no", "severity": "3"}
NOOP_4 = {"row_action": "no_op", "needs_human": "no", "severity": "4"}

FX = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "CHF": 0.88, "DKK": 6.9, "SEK": 10.6, "NOK": 10.8,
      "PLN": 4.0, "CZK": 23.0, "HUF": 360.0, "RON": 4.6, "TRY": 39.0, "AED": 3.67, "SAR": 3.75,
      "ILS": 3.7, "INR": 85.0, "SGD": 1.35, "MYR": 4.5, "THB": 36.0, "VND": 25400.0,
      "IDR": 16300.0, "CNY": 7.2, "HKD": 7.8, "TWD": 32.0, "JPY": 150.0, "KRW": 1380.0,
      "BRL": 5.5, "MXN": 18.0, "CLP": 940.0, "COP": 4100.0, "CAD": 1.37}
SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", "TRY": "₺"}
MON = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH = ["", "January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"]
COUNTRY_NAME = {"DE": "Germany", "FR": "France", "NL": "Netherlands", "BE": "Belgium", "IT": "Italy",
                "ES": "Spain", "GB": "United Kingdom", "IE": "Ireland", "AT": "Austria",
                "CH": "Switzerland", "PT": "Portugal", "DK": "Denmark", "SE": "Sweden", "NO": "Norway",
                "FI": "Finland", "PL": "Poland", "CZ": "Czechia", "HU": "Hungary", "RO": "Romania",
                "TR": "Turkey", "AE": "UAE", "SA": "Saudi Arabia", "IN": "India", "SG": "Singapore",
                "MY": "Malaysia", "TH": "Thailand", "VN": "Vietnam", "ID": "Indonesia",
                "PH": "Philippines", "CN": "China", "HK": "Hong Kong", "TW": "Taiwan", "JP": "Japan",
                "KR": "South Korea", "BR": "Brazil", "MX": "Mexico", "AR": "Argentina", "CL": "Chile",
                "CO": "Colombia", "PE": "Peru", "US": "USA", "CA": "Canada"}

# legal forms, longest first; an alias may expand or drop one
SUFFIX_EXPAND = {
    "Pvt Ltd": "Private Limited", "Pvt. Ltd.": "Private Limited", "Private Limited": "Pvt Ltd",
    "Pte Ltd": "Pte. Limited", "Pte. Ltd.": "Private Limited", "Co., Ltd.": "Company Limited",
    "Co. Ltd.": "Company Limited", "Ltd": "Limited", "Ltd.": "Limited", "Limited": "Ltd",
    "A.S.": "Anonim Sirketi", "A.Ş.": "Anonim Şirketi", "Sdn Bhd": "Sdn. Berhad",
    "Sdn. Bhd.": "Sendirian Berhad", "Inc.": "Incorporated", "Inc": "Incorporated",
    "Corp.": "Corporation", "Corporation": "Corp", "K.K.": "Kabushiki Kaisha",
    "Ltda.": "Limitada", "Ltda": "Limitada", "plc": "Public Limited Company",
    "LLC": "Limited Liability Company", "GmbH": "Gesellschaft mbH", "S.A.": "Sociedad Anonima",
    "S.p.A.": "Societa per Azioni", "S.r.l.": "Societa a responsabilita limitata",
    "B.V.": "Besloten Vennootschap", "A/S": "Aktieselskab", "AB": "Aktiebolag", "Oy": "Osakeyhtio",
}
SUFFIXES = sorted({*SUFFIX_EXPAND, "San. ve Tic. A.Ş.", "San. ve Tic. A.S.", "Ltd. Şti.",
                   "S.A. de C.V.", "Sp. z o.o.", "S.R.L.", "S.A.S.", "S.A.C.", "s.r.o.",
                   "GmbH & Co. KG", "AG", "SAS", "SARL", "N.V.", "S.L.", "PLC", "ApS", "AS",
                   "Oyj", "ehf.", "Kft.", "EOOD", "d.o.o.", "UAB", "SIA", "OÜ", "OU", "L.L.C.",
                   "FZE", "FZ-LLC", "FZCO", "W.L.L.", "JSC", "PJSC", "Co.", "SpA", "Tbk", "Ltd"},
                  key=len, reverse=True)
ABBREV = {"controls": "CTRL", "control": "CTRL", "electronics": "ELEC", "electronic": "ELEC",
          "technologies": "TECH", "technology": "TECH", "systems": "SYS", "system": "SYS",
          "industries": "IND", "industrial": "IND", "international": "INTL", "manufacturing": "MFG",
          "solutions": "SOL", "engineering": "ENGG", "components": "COMP", "automation": "AUTO",
          "devices": "DEV", "instruments": "INSTR", "communications": "COMMS", "products": "PROD",
          "enterprises": "ENT", "trading": "TRDG", "elektronik": "ELEK", "teknoloji": "TEKNO",
          "sanayi": "SAN", "ticaret": "TIC", "company": "CO", "corporation": "CORP",
          "laboratories": "LABS", "medical": "MED", "energy": "ENRG", "smart": "SMT"}
GENERIC = ["Technologies", "Technology", "Solutions", "Systems", "International", "Intl", "Group"]
INDUSTRY_ALT = ["Energy Solutions", "Power Systems", "Digital Labs", "Mobility", "Infra Projects",
                "Retail Networks", "Holdings", "Ventures", "Logistics", "Agro Tech", "Health",
                "Capital Partners", "Trading House", "Industrial Group"]
REGION_WORD = ["Asia", "Europe", "Iberia", "Nordic", "Middle East", "Americas", "India", "APAC",
               "EMEA", "Anatolia", "Benelux", "Pacific"]
BRANDS = ["Corvane", "Ostrava", "Meridian", "Arctura", "Valmora", "Quillon", "Brightmoor",
          "Tessaro", "Varnholt", "Nordvik", "Sorrento", "Zephyra", "Caldera", "Lumora", "Ironbay"]
SENSITIVE = ("drone", "uav", "defen", "surveil", "security", "radar", "tactical", "border",
             "military", "aerospace", "satellite", "camera")
ORDINARY = ("toy", "lighting", "home", "appliance", "audio", "kitchen", "garden", "furniture",
            "wearable", "fitness", "pet", "education")
HEADER_VALUES = [
    {"period": "Period", "customer": "Customer Name", "country": "Country", "part": "Part No",
     "qty": "Qty", "unit_price": "Unit Price", "ref": "Invoice No"},
    {"period": "Month", "customer": "End Customer", "country": "Ctry", "part": "Part Number",
     "qty": "Quantity", "unit_price": "Price", "ref": "Reference"},
    {"period": "PERIOD", "customer": "CUSTOMER", "country": "COUNTRY", "part": "P/N",
     "qty": "QTY", "unit_price": "UNIT PRICE", "ref": "INV NO"},
    {"period": "Date", "customer": "Sold To", "country": "Country", "part": "Material",
     "qty": "Qty.", "unit_price": "Unit_Price", "ref": "Doc No"},
]
NOTES = ["*** End of report ***", "*** continued on next page ***", "Note: prices exclude VAT",
         "--- page 2 ---", "Remark: Q4 forecast to follow", "Comment: see cover email",
         "*** END OF FILE ***", "Prices include freight"]
FILLERS = ["", "-----", "=====", "n/a", "---", ""]
UNIT_WORDS = ["pcs", " pcs", " EA", " units", "PCS", " pc"]


@dataclass(frozen=True)
class Part:
    part: str
    spq: int
    moq: int
    usd: float


@dataclass(frozen=True)
class Row:
    distributor: str
    period: str
    customer: str
    country: str
    part: str
    qty: str
    unit_price: str
    ref: str
    m_customer: str
    spq: str
    moq: str
    restricted: str = "no"
    seen_refs: str = "none"

    def text(self) -> str:
        return (f"distributor={self.distributor}; period={self.period}; customer={self.customer}; "
                f"country={self.country}; part={self.part}; qty={self.qty}; "
                f"unit_price={self.unit_price}; ref={self.ref} || master: customer={self.m_customer}; "
                f"spq={self.spq}; moq={self.moq}; restricted={self.restricted}; seen_refs={self.seen_refs}")


class Generator:
    def __init__(self, pools: dict, parts: list[Part], currencies: dict[str, list[str]] | None = None,
                 seed: int = 1):
        self.rng = random.Random(seed)
        self.seed = seed
        self.pools = {r: b for r, b in pools.items() if b.get("customers") and b.get("distributors")}
        self.regions = sorted(self.pools)
        self.parts = parts
        self.catalogue = {p.part for p in parts}
        self.currencies = currencies or {}
        self.used_refs: set[str] = set()
        self.texts: set[str] = set()
        self.rejected: Counter = Counter()

    # ------------------------------------------------------------ building blocks

    def _ref(self, style: str) -> str:
        style = style if "#" in style else "INV-######"
        for _ in range(50):
            ref = re.sub("#", lambda _: str(self.rng.randrange(10)), style)
            if ref not in self.used_refs:
                self.used_refs.add(ref)
                return ref
        return ref + str(self.rng.randrange(10, 99))

    def _near_ref(self, ref: str) -> str:
        """A ref that looks like `ref` but is a different reference."""
        digits = [i for i, c in enumerate(ref) if c.isdigit()]
        r = self.rng.random()
        if digits and r < 0.5:
            i = self.rng.choice(digits)
            new = ref[:i] + str((int(ref[i]) + self.rng.randrange(1, 10)) % 10) + ref[i + 1:]
        elif r < 0.75:
            new = ref + str(self.rng.randrange(10))
        elif "/" in ref:
            head, tail = ref.rsplit("/", 1)
            new = f"{head}/{tail}{self.rng.randrange(1, 10)}"
        else:
            new = ref[:-1] if len(ref) > 4 else ref + "1"
        return new if pos.norm_ref(new) != pos.norm_ref(ref) else ref + "7"

    def _currency(self, region: str) -> str:
        options = [c for c in self.currencies.get(region, ["USD"]) if c in FX] or ["USD"]
        return "USD" if self.rng.random() < 0.35 else self.rng.choice(options)

    @staticmethod
    def _price_value(part: Part, cur: str, rng: random.Random) -> float:
        v = part.usd * FX[cur] * rng.uniform(0.9, 1.12)
        return round(v) if v >= 1000 else round(v, 2)

    @staticmethod
    def _price_text(v: float, cur: str) -> str:
        return f"{int(v)} {cur}" if v >= 1000 else f"{v:.2f} {cur}"

    def _valid_qty(self, part: Part, scale: str | None = None) -> int:
        kmin = max(1, math.ceil(part.moq / part.spq))
        scale = scale or self.rng.choice(["small", "typical", "typical", "large"])
        if part.spq == 1:
            hi = {"small": 60, "typical": 600, "large": 5000}[scale]
            return self.rng.randint(max(part.moq, 1), max(part.moq, hi))
        span = {"small": 3, "typical": 12, "large": 60}[scale]
        return part.spq * self.rng.randint(kmin, kmin + span)

    def base(self, region: str | None = None, part: Part | None = None, customer: dict | None = None,
             history: str | None = None) -> Row:
        rng = self.rng
        region = region or rng.choice(self.regions)
        block = self.pools[region]
        dist = rng.choice(block["distributors"])
        cust = customer or rng.choice(block["customers"])
        part = part or rng.choice(self.parts)
        cur = self._currency(region)
        y, m = rng.choice([(2025, mm) for mm in range(9, 13)] + [(2026, mm) for mm in range(1, 11)] * 3)
        ref = self._ref(dist.get("ref_style", "INV-######"))
        history = history or rng.choice(["none", "none", "unrelated", "several", "similar"])
        style = dist.get("ref_style", "INV-######")
        if history == "none":
            seen = "none"
        elif history == "unrelated":
            seen = self._ref(style)
        elif history == "several":
            seen = ", ".join(self._ref(style) for _ in range(rng.randint(2, 4)))
        else:
            seen = ", ".join([self._near_ref(ref)] + [self._ref(style) for _ in range(rng.randint(0, 2))])
        return Row(distributor=dist["name"], period=f"{y}-{m:02d}", customer=cust["name"],
                   country=cust.get("country", "US"), part=part.part, qty=str(self._valid_qty(part)),
                   unit_price=self._price_text(self._price_value(part, cur, rng), cur), ref=ref,
                   m_customer=cust["name"], spq=str(part.spq), moq=str(part.moq), seen_refs=seen)

    def _part(self, pred=lambda p: True) -> Part:
        ok = [p for p in self.parts if pred(p)]
        return self.rng.choice(ok or self.parts)

    def _customer(self, keywords: tuple[str, ...]) -> tuple[str, dict]:
        hits = [(r, c) for r in self.regions for c in self.pools[r]["customers"]
                if any(k in (c.get("industry", "") + " " + c["name"]).lower() for k in keywords)]
        if hits:
            return self.rng.choice(hits)
        r = self.rng.choice(self.regions)
        return r, self.rng.choice(self.pools[r]["customers"])

    # ------------------------------------------------------------ format noise (value kept)

    def noisy_period(self, row: Row) -> Row:
        y, m = int(row.period[:4]), int(row.period[5:7])
        d = self.rng.randint(13, 28)
        styles = [f"{d:02d}.{m:02d}.{y}", f"{MON[m]}-{y % 100:02d}", f"{y}/{m:02d}",
                  f"{y}/{m:02d}/{d:02d}", f"{m:02d}/{y}", f"{MONTH[m]} {y}", f"{y}-{m:02d}-{d:02d}",
                  f"{MON[m].upper()} {y}", f"{y}{m:02d}", f"{d}/{m:02d}/{y}"]
        if m < 10:
            styles.append(f"{y}-{m}")
        return replace(row, period=self.rng.choice(styles))

    def noisy_qty(self, row: Row, q: int | None = None) -> Row:
        q = int(row.qty) if q is None else q
        styles = [f"{q}{self.rng.choice(UNIT_WORDS)}", f"{q}.00", f"{q},00", f"  {q} "]
        if q >= 1000:
            styles += [f"{q:,}", f"{q:,}".replace(",", "."), f"{q:,}".replace(",", " "),
                       f"{q:,} pcs", f"{q:,}".replace(",", ".") + ",00"]
            if q % 100 == 0:
                styles.append(f"{q / 1000:g}k")
        if q >= 100000:
            s = str(q)
            head, tail = s[:-3], s[-3:]
            groups = []
            while len(head) > 2:
                groups.insert(0, head[-2:])
                head = head[:-2]
            styles.append(",".join([head] + groups + [tail]) if head else ",".join(groups + [tail]))
        return replace(row, qty=self.rng.choice(styles))

    def noisy_price(self, row: Row) -> Row:
        num, cur = row.unit_price.rsplit(" ", 1)
        styles = [f"{num.replace('.', ',')} {cur}" if "." in num else f"{num},00 {cur}",
                  f"{num} {cur.lower()}", f"{cur} {num}", f"{num}{cur}"]
        if cur in SYMBOL:
            styles.append(f"{SYMBOL[cur]}{num}")
        return replace(row, unit_price=self.rng.choice(styles))

    def noisy_part(self, row: Row) -> Row:
        p = row.part
        mixed = "".join(c.lower() if self.rng.random() < 0.5 else c for c in p)
        styles = [p.lower(), p.replace("-", " "), p.replace("-", ""), p + " ",
                  mixed if mixed != p else p.lower()]
        return replace(row, part=self.rng.choice(styles))

    def noisy_country(self, row: Row) -> Row:
        styles = [row.country.lower()]
        if row.country in COUNTRY_NAME:
            styles.append(COUNTRY_NAME[row.country])
        return replace(row, country=self.rng.choice(styles))

    def format_only_name(self, name: str) -> str:
        styles = [name.upper(), name.lower(), name + ".", name.replace(" ", "  ", 1),
                  name.replace(".", "") if "." in name else name.upper(), " " + name]
        for _ in range(10):
            s = self.rng.choice(styles)
            if s != name and pos.name_relation(s, name) == "format_only":
                return s
        return name.upper()

    def noise(self, row: Row, k: int | None = None, fields: list[str] | None = None) -> Row:
        fields = fields or ["period", "qty", "price", "part", "country"]
        k = k or self.rng.choice([1, 1, 2, 3])
        for f in self.rng.sample(fields, k=min(k, len(fields))):
            row = {"period": self.noisy_period, "qty": self.noisy_qty, "price": self.noisy_price,
                   "part": self.noisy_part, "country": self.noisy_country}[f](row)
        return row

    # ------------------------------------------------------------ names

    @staticmethod
    def split_suffix(name: str) -> tuple[str, str]:
        for suf in SUFFIXES:
            if name.endswith(" " + suf) or name.endswith("," + suf) or name == suf:
                return name[: -len(suf)].rstrip(" ,"), suf
        return name, ""

    def clear_alias(self, name: str, kind: str | None = None) -> str | None:
        """A clearly-same-entity variant. kind: abbrev | suffix | typo | generic | None (any).

        Returns None when the requested kind is impossible for this name, so the caller can pick
        another customer instead of labelling a scenario it does not meet.
        """
        core, suf = self.split_suffix(name)
        words = core.split()
        options: dict[str, list[str]] = {"abbrev": [], "suffix": [], "typo": [], "generic": []}
        abbr = [ABBREV.get(w.lower().strip(","), None) for w in words]
        if any(abbr):
            short = " ".join(a or w.upper() for a, w in zip(abbr, words))
            options["abbrev"].append(f"{short} {suf.upper()}".strip())
        if suf in SUFFIX_EXPAND:
            options["suffix"].append(f"{core} {SUFFIX_EXPAND[suf]}")
        if suf:
            options["suffix"].append(core)  # suffix dropped
        long_words = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
        if long_words:
            i = self.rng.choice(long_words)
            w = words[i]
            j = self.rng.randrange(1, len(w) - 2)
            typo = self.rng.choice([w[:j] + w[j + 1] + w[j] + w[j + 2:], w[:j] + w[j + 1:],
                                    w[:j] + w[j] + w[j:]])
            options["typo"].append(" ".join(words[:i] + [typo] + words[i + 1:]) + (f" {suf}" if suf else ""))
        generic = [i for i, w in enumerate(words) if w in GENERIC]
        if generic and len(words) >= 3:
            i = generic[0]
            options["generic"].append(" ".join(words[:i] + words[i + 1:]) + (f" {suf}" if suf else ""))
        pool = options[kind] if kind else [o for v in options.values() for o in v]
        self.rng.shuffle(pool)
        for o in pool:
            if pos.name_relation(o, name) == "different":
                return o
        return None if kind else name + " Intl"

    def unsure_alias(self, name: str, region: str, kind: str | None = None) -> str:
        """An only-loosely-linked name. kind: sister | unrelated | branch | holding | None (any)."""
        core, suf = self.split_suffix(name)
        first = core.split()[0] if core.split() else core
        local = self.split_suffix(self.rng.choice(self.pools[region]["customers"])["name"])[1] or "Ltd"
        kind = kind or self.rng.choice(["sister", "unrelated", "branch", "holding"])
        if kind == "sister" and len(core.split()) >= 2:
            alias = f"{first} {self.rng.choice(INDUSTRY_ALT)}"
        elif kind == "branch":
            alias = f"{core} {self.rng.choice(REGION_WORD)} {local}"
        elif kind == "holding":
            alias = f"{self.rng.choice(BRANDS)} Group Holdings"
        else:
            # an invented name, never another known customer's canonical name: that would read
            # as a lookup error (ask_human), not as a rebrand of the matched customer
            alias = f"{self.rng.choice(BRANDS)} {self.rng.choice(INDUSTRY_ALT)}"
            if self.rng.random() < 0.5:
                alias += f" {local}"
        return alias if pos.name_relation(alias, name) == "different" else f"{self.rng.choice(BRANDS)} Holdings"

    # ------------------------------------------------------------ pack quantities

    def bad_qty(self, part: Part, kind: str) -> int | None:
        rng = self.rng
        spq, moq = part.spq, part.moq
        if kind == "not_multiple" and spq > 1:
            return self._valid_qty(part) + rng.randrange(1, spq)
        if kind == "below_moq" and moq > spq:
            return spq * rng.randint(1, max(1, (moq - 1) // spq))
        if kind == "both" and moq > 1 and spq > 1:
            q = rng.randrange(1, moq)
            return q if q % spq else q + 1 if (q + 1) < moq and (q + 1) % spq else None
        if kind == "near_miss" and spq > 1:
            q = self._valid_qty(part)
            delta = rng.choice([1, 2, 5, 10, 20, -1, -5, -10, -20])
            q2 = q + delta
            return q2 if q2 > 0 and q2 % spq else None
        return None

    # ------------------------------------------------------------ scenarios

    def scenario(self, sid: str) -> tuple[Row, dict] | None:
        rng = self.rng
        if sid == "C1":
            return self.base(), COMMIT
        if sid == "C2":
            p = self._part(lambda p: p.moq % p.spq == 0 and p.moq >= p.spq)
            r = self.base(part=p)
            return replace(r, qty=str(p.moq)), COMMIT
        if sid == "C3":
            return self.base(history="similar"), COMMIT
        if sid == "C4":
            p = self._part(lambda p: p.spq <= 10)
            r = self.base(part=p)
            lo = max(1, math.ceil(p.moq / p.spq))
            q = rng.choice([7, 13, 37, 61, 130, 217, 1013]) if p.spq == 1 else p.spq * rng.randint(lo, max(lo, 40))
            return replace(r, qty=str(max(q, p.moq))), COMMIT
        if sid == "C5":
            region = rng.choice(self.regions)
            longest = sorted(self.pools[region]["customers"], key=lambda c: -len(c["name"]))[:8]
            return self.base(region=region, customer=rng.choice(longest)), COMMIT
        if sid == "C6":
            region, c = self._customer(SENSITIVE)
            return self.base(region=region, customer=c), COMMIT
        if sid in ("N1", "N2", "N3", "N4", "N7"):
            field = {"N1": "period", "N2": "qty", "N3": "price", "N4": "part", "N7": "country"}[sid]
            r = self.noise(self.base(), k=1, fields=[field])
            if sid == "N7" and rng.random() < 0.5:
                r = self.noise(r, k=1, fields=["period", "price"])
            return r, NORMALIZE
        if sid == "N5":
            r = self.base()
            r = replace(r, customer=self.format_only_name(r.m_customer))
            return (self.noise(r, k=1) if rng.random() < 0.3 else r), NORMALIZE
        if sid == "N6":
            r = self.noise(self.base(), k=rng.choice([2, 3]))
            if rng.random() < 0.4:
                r = replace(r, customer=self.format_only_name(r.m_customer))
            return r, NORMALIZE
        if sid in ("V1", "V2", "V3", "V4", "V5"):
            kind = {"V1": "abbrev", "V2": "suffix", "V3": "typo", "V4": "generic"}.get(sid)
            for _ in range(30):  # find a customer whose name admits this kind of rename
                r = self.base()
                alias = self.clear_alias(r.m_customer, kind)
                if alias:
                    break
            else:
                return None
            r = replace(r, customer=alias)
            return (self.noise(r) if sid == "V5" else r), RENAME
        if sid in ("VU1", "VU2", "VU3"):
            for _ in range(30):  # a sister company shares one word of a core that has several
                region = rng.choice(self.regions)
                r = self.base(region=region)
                if sid != "VU1" or len(self.split_suffix(r.m_customer)[0].split()) >= 2:
                    break
            else:
                return None
            kind = {"VU1": "sister", "VU2": rng.choice(["unrelated", "holding"]), "VU3": "branch"}[sid]
            r = replace(r, customer=self.unsure_alias(r.m_customer, region, kind))
            return (self.noise(r, k=1) if sid == "VU3" and rng.random() < 0.4 else r), RENAME_UNSURE
        if sid in ("P1", "P2", "P3", "P4", "P5", "P6"):
            kind = {"P1": "not_multiple", "P2": "below_moq", "P3": "both", "P4": "near_miss"}.get(
                sid, rng.choice(["not_multiple", "below_moq", "near_miss"]))
            for _ in range(20):
                p = self._part()
                q = self.bad_qty(p, kind)
                if q:
                    break
            else:
                return None
            r = replace(self.base(part=p), qty=str(q))
            if sid == "P5":
                r = self.noisy_qty(r, q)
                if rng.random() < 0.5:
                    r = self.noise(r, k=1, fields=["period", "price", "part"])
            if sid == "P6":
                alias = self.clear_alias(r.m_customer, rng.choice(["abbrev", "suffix"]))
                alias = alias or self.clear_alias(r.m_customer, "suffix") or self.clear_alias(r.m_customer, "abbrev")
                if not alias:
                    return None
                r = replace(r, customer=alias)
            return r, PACK
        if sid in ("R1", "R2", "R3"):
            if sid == "R3":
                region, c = self._customer(ORDINARY)
                r = self.base(region=region, customer=c)
            else:
                r = self.base()
            if sid == "R2":
                r = self._extra_problem(r)
            return replace(r, restricted="yes"), COMPLIANCE
        if sid in ("D1", "D2", "D3", "D4"):
            r = self.base(history="none")
            ref = r.ref
            if sid == "D3":
                variants = [v for v in (ref.lower(), ref.replace("/", " / "), ref.replace("-", " - "),
                                        f" {ref}", ref[:len(ref) // 2] + " " + ref[len(ref) // 2:])
                            if v != ref and pos.norm_ref(v) == pos.norm_ref(ref)]
                ref = rng.choice(variants)
            others = [self._ref(self._style(r)) for _ in range(rng.randint(1, 4))] if sid in ("D2", "D4") else []
            seen = others + [ref]
            rng.shuffle(seen)
            r = replace(r, seen_refs=", ".join(seen))
            if sid == "D4":
                r = self._extra_problem(r, allow_dup=False)
            return r, DUPLICATE
        if sid == "A1":
            r = self.base()
            for f in rng.sample(["customer", "part", "qty", "unit_price", "ref", "period", "country"], k=rng.choice([1, 1, 2])):
                r = replace(r, **{f: rng.choice(["", " ", "-"]) if f not in ("customer", "part") else ""})
            return r, ASK
        if sid == "A2":
            r = self.base()
            kind = rng.choice(["qty0", "qtyneg", "qtyfrac", "price0", "priceneg", "pricetext", "month"])
            cur = r.unit_price.rsplit(" ", 1)[-1]
            if kind == "qty0":
                r = replace(r, qty="0")
            elif kind == "qtyneg":
                r = replace(r, qty=f"-{r.qty}")
            elif kind == "qtyfrac":
                r = replace(r, qty=rng.choice(["1.5", "2,75", "0.5", "12.25", "3.3"]))
            elif kind == "price0":
                r = replace(r, unit_price=f"0.00 {cur}")
            elif kind == "priceneg":
                r = replace(r, unit_price=f"-{r.unit_price}")
            elif kind == "pricetext":
                r = replace(r, unit_price=rng.choice(["TBD", "n/a", "see contract", "FOC", "tba"]))
            else:
                r = replace(r, period=f"{r.period[:4]}-{rng.choice(['13', '00', '14'])}")
            return r, ASK
        if sid == "A3":
            r = self.base()
            kind = rng.choice(["two_parts", "two_qty", "qty_note", "price_note", "part_note"])
            if kind == "two_parts":
                other = self._part(lambda p: p.part != r.part).part
                r = replace(r, part=f"{r.part} / {other}")
            elif kind == "two_qty":
                r = replace(r, qty=f"{r.qty}/{int(r.qty) + int(r.spq)}")
            elif kind == "qty_note":
                r = replace(r, qty=rng.choice(["see attachment", "TBC", "approx " + r.qty, r.qty + "?", "as per PO"]))
            elif kind == "price_note":
                r = replace(r, unit_price=rng.choice(["see contract", "as quoted", "price on request"]))
            else:
                r = replace(r, part=f"{r.part} (split?)")
            return r, ASK
        if sid == "A4":
            r = self.base(history="none")
            kind = rng.choice(["customer", "spqmoq", "both"])
            if kind in ("customer", "both"):
                r = replace(r, m_customer="unknown")
            if kind in ("spqmoq", "both"):
                r = replace(r, spq="?", moq="?")
            return r, ASK
        if sid == "A5":
            r = self.base()
            y = r.period[:4]
            return replace(r, period=rng.choice([f"Q{rng.randint(1, 4)} {y}", f"FY{y[2:]}",
                                                 f"H{rng.randint(1, 2)}-{y}", y, f"{y}-Q{rng.randint(1, 4)}"])), ASK
        if sid in ("O1", "O2", "O3", "O4", "O5"):
            return self._nonsales(sid)
        raise UnknownRecipe(sid)

    def _style(self, row: Row) -> str:
        for b in self.pools.values():
            for d in b["distributors"]:
                if d["name"] == row.distributor:
                    return d.get("ref_style", "INV-######")
        return "INV-######"

    def _extra_problem(self, r: Row, allow_dup: bool = True) -> Row:
        kinds = ["pack", "rename", "format"] + (["dup"] if allow_dup else [])
        kind = self.rng.choice(kinds)
        if kind == "pack":
            p = next(p for p in self.parts if p.part == r.part)
            q = self.bad_qty(p, "not_multiple") or self.bad_qty(p, "below_moq")
            return replace(r, qty=str(q)) if q else self.noise(r, k=1)
        if kind == "rename":
            return replace(r, customer=self.clear_alias(r.m_customer))
        if kind == "dup":
            return replace(r, seen_refs=r.ref if r.seen_refs == "none" else f"{r.seen_refs}, {r.ref}")
        return self.noise(r, k=1)

    def _nonsales(self, sid: str) -> tuple[Row, dict]:
        rng = self.rng
        region = rng.choice(self.regions)
        dist = rng.choice(self.pools[region]["distributors"])["name"]
        unknown = dict(m_customer="unknown", spq="?", moq="?", restricted="no", seen_refs="none")
        if sid == "O1":
            h = rng.choice(HEADER_VALUES)
            return Row(distributor=dist, **h, **unknown), NOOP_1
        if sid == "O2":
            f = lambda: rng.choice(FILLERS)  # noqa: E731
            return Row(distributor=dist, period=rng.choice(["", f()]), customer=f(), country="",
                       part=f(), qty="", unit_price="", ref="", **unknown), NOOP_1
        if sid == "O3":
            note = rng.choice(NOTES)
            where = rng.choice(["customer", "part"])
            vals = {"customer": "", "part": "", where: note}
            return Row(distributor=dist, period="", country="", qty="", unit_price="", ref="",
                       **vals, **unknown), NOOP_1
        y, m = 2026, rng.randint(1, 10)
        if sid == "O4":
            cust = rng.choice(self.pools[region]["customers"])
            total = sum(self._valid_qty(self._part()) for _ in range(rng.randint(2, 6)))
            qty = rng.choice([str(total), f"{total:,}"])
            kind = rng.random()
            if kind < 0.5:
                fields = dict(customer=cust["name"], part=rng.choice(["SUBTOTAL", "Subtotal", "SUB-TOTAL", "Sub total"]))
            elif kind < 0.8:
                fields = dict(customer=f"Total {cust['name']}", part="")
            else:
                p = self._part()
                fields = dict(customer="", part=f"Total {p.part}")
            return Row(distributor=dist, period=f"{y}-{m:02d}", country=cust.get("country", "") if fields["customer"] else "",
                       qty=qty, unit_price="", ref=rng.choice(["", "SUBTOTAL", ""]),
                       m_customer=cust["name"] if fields["customer"] == cust["name"] else "unknown",
                       spq="?", moq="?", restricted="no", seen_refs="none", **fields), NOOP_3
        total = sum(self._valid_qty(self._part(), "large") for _ in range(rng.randint(8, 30)))
        label = rng.choice(["GRAND TOTAL", "Grand Total", "Report total", "TOTAL ALL CUSTOMERS",
                            "Overall total"])
        where = rng.choice(["customer", "customer", "part"])
        vals = {"customer": "", "part": "", where: label}
        return Row(distributor=dist, period=rng.choice([f"{y}-{m:02d}", ""]), country="",
                   qty=rng.choice([str(total), f"{total:,}"]), unit_price="", ref="",
                   **vals, **unknown), NOOP_4

    # ------------------------------------------------------------ hard pairs

    def pair(self, pid: str) -> tuple[tuple[Row, dict], tuple[Row, dict]] | None:
        rng = self.rng
        if pid == "H01":
            p = self._part(lambda p: p.spq > 1)
            if p.spq <= 1:
                return None
            a = self.base(part=p)
            q = int(a.qty) + rng.randrange(1, p.spq)
            return (a, COMMIT), (replace(a, qty=str(q)), PACK)
        if pid == "H02":
            p = self._part(lambda p: p.moq > p.spq and p.moq % p.spq == 0)
            a = replace(self.base(part=p), qty=str(p.moq))
            return (a, COMMIT), (replace(a, qty=str(p.moq - p.spq)), PACK)
        if pid == "H03":
            a = self.base(history=rng.choice(["none", "unrelated", "several"]))
            seen = a.ref if a.seen_refs == "none" else f"{a.seen_refs}, {a.ref}"
            return (a, COMMIT), (replace(a, seen_refs=seen), DUPLICATE)
        if pid == "H04":
            a = self.base(history="none")
            others = [self._ref(self._style(a)) for _ in range(rng.randint(0, 2))]
            seen_a = others + [a.ref]
            rng.shuffle(seen_a)
            near = self._near_ref(a.ref)
            seen_b = [near if s == a.ref else s for s in seen_a]
            return (replace(a, seen_refs=", ".join(seen_a)), DUPLICATE), (replace(a, seen_refs=", ".join(seen_b)), COMMIT)
        if pid == "H05":
            a = self.base()
            return (a, COMMIT), (replace(a, restricted="yes"), COMPLIANCE)
        if pid == "H06":
            for _ in range(30):  # row b is an abbreviation: find a customer whose name has one
                a = self.base()
                ab = self.clear_alias(a.m_customer, "abbrev")
                if ab:
                    break
            else:
                return None
            fo = self.format_only_name(a.m_customer)
            return (replace(a, customer=fo), NORMALIZE), (replace(a, customer=ab), RENAME)
        if pid == "H07":
            region = rng.choice(self.regions)
            a = self.base(region=region)
            clear = self.clear_alias(a.m_customer, rng.choice(["abbrev", "suffix"]))
            clear = clear or self.clear_alias(a.m_customer, "suffix") or self.clear_alias(a.m_customer, "abbrev")
            if not clear:
                return None
            return (replace(a, customer=clear), RENAME), \
                   (replace(a, customer=self.unsure_alias(a.m_customer, region)), RENAME_UNSURE)
        if pid == "H08":
            region = rng.choice(self.regions)
            cust = rng.choice(self.pools[region]["customers"])
            sales = self.base(region=region, customer=cust)
            sub = replace(sales, part=rng.choice(["SUBTOTAL", "Subtotal"]), qty=f"{int(sales.qty) * rng.randint(2, 6):,}",
                          unit_price="", ref="", spq="?", moq="?", seen_refs="none")
            broken = replace(sales, **{rng.choice(["unit_price", "ref", "part"]): ""})
            return (sub, NOOP_3), (broken, ASK)
        if pid == "H09":
            p = self._part(lambda p: p.spq > 1 and p.moq <= 1000)
            a = replace(self.base(part=p), qty=str(max(p.moq, p.spq * math.ceil(1000 / p.spq))))
            q = int(a.qty)
            if q % 1000:
                q = 1000 * (q // 1000 + 1)
                if q % p.spq:
                    return None
            a = replace(a, qty=f"{q // 1000}.{q % 1000:03d}")
            return (a, NORMALIZE), (replace(a, qty=rng.choice(["1.5", "2,5", "1.25", "0.75"])), ASK)
        if pid == "H10":
            a = self.base()
            return (a, COMMIT), (self.noisy_period(a), NORMALIZE)
        if pid == "H11":
            a = self.base()
            return (a, COMMIT), (replace(a, period=f"{a.period[:4]}-{rng.choice(['13', '00'])}"), ASK)
        if pid == "H12":
            a = self.base(history="none")
            return (a, COMMIT), (replace(a, m_customer="unknown", spq="?", moq="?"), ASK)
        raise UnknownRecipe(pid)

    # ------------------------------------------------------------ drivers

    def _keep(self, row: Row, labels: dict, meta: dict) -> Record | None:
        text = row.text()
        if any(f.level == "violation" for f in pos.check_labels(text, labels, self.catalogue)):
            self.rejected[meta.get("scenario", meta.get("pair", "?"))] += 1
            return None
        return Record(id=stable_id(text, "proc"), text=text, labels=dict(labels), source="synthetic",
                      meta={"generator": "procedural", "seed": self.seed, **meta})

    def scenarios(self, counts: dict[str, int], log=print) -> list[Record]:
        out = []
        for sid, n in counts.items():
            got, tries = 0, 0
            while got < n and tries < n * 5:
                tries += 1
                try:
                    built = self.scenario(sid)
                except UnknownRecipe:
                    log(f"procgen: no procedural recipe for scenario {sid}; skipped (teacher rows only)")
                    break
                if built is None:
                    continue
                rec = self._keep(*built, {"scenario": sid})
                if rec and rec.text not in self.texts:
                    self.texts.add(rec.text)
                    out.append(rec)
                    got += 1
            if 0 < n and got < n and tries >= n * 5:
                log(f"procgen: scenario {sid}: {got} of {n} rows (the pools or parts rarely fit it)")
        return out

    def pairs(self, counts: dict[str, int], log=print) -> list[Record]:
        out = []
        for pid, n in counts.items():
            got, tries = 0, 0
            while got < n and tries < n * 5:
                tries += 1
                try:
                    built = self.pair(pid)
                except UnknownRecipe:
                    log(f"procgen: no procedural recipe for pair {pid}; skipped (teacher rows only)")
                    break
                if built is None:
                    continue
                k = f"{pid}-{got:04d}"
                (ra, la), (rb, lb) = built
                a = self._keep(ra, la, {"scenario": pid, "pair": k, "side": "a"})
                b = self._keep(rb, lb, {"scenario": pid, "pair": k, "side": "b"})
                # a pair is kept whole or not at all, so a duplicate side never orphans its partner
                if a and b and a.text != b.text and a.text not in self.texts and b.text not in self.texts:
                    self.texts |= {a.text, b.text}
                    out += [a, b]
                    got += 1
            if 0 < n and got < n and tries >= n * 5:
                log(f"procgen: pair {pid}: {got} of {n} pairs (the pools or parts rarely fit it)")
        return out


def load_parts(path: str | Path) -> list[Part]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [Part(p["part"], int(p["spq"]), int(p["moq"]), float(p["usd"])) for p in data["parts"]]


def load_currencies(path: str | Path | None) -> dict[str, list[str]]:
    if not path or not Path(path).exists():
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {r: v.get("currencies", ["USD"]) for r, v in data.get("regions", {}).items()}


def generate(pools_path: str, parts_path: str, recipe_path: str, regions_path: str | None = None,
             n_per_scenario: int = 100, pairs_per_type: int = 50, seed: int = 1,
             log=print) -> list[Record]:
    recipe = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8"))
    pools = json.loads(Path(pools_path).read_text(encoding="utf-8"))
    gen = Generator(pools, load_parts(parts_path), load_currencies(regions_path), seed=seed)
    counts = {s["id"]: max(1, round(n_per_scenario * float(s.get("weight", 1.0))))
              for s in recipe["scenarios"]}
    records = gen.scenarios(counts, log)
    records += gen.pairs({p["id"]: pairs_per_type for p in recipe.get("pairs", [])}, log)
    if gen.rejected:
        log("rejected by the rule check (rebuilt): " + ", ".join(f"{k}={v}" for k, v in sorted(gen.rejected.items())))
    return records
