# POS row gate: labelling guide (schema `pos_row_gate` v2)

This guide is for three readers: people labelling dev and eval rows, reviewers writing
corrections, and the teacher models that write synthetic rows. The option descriptions in
`configs/schema.pos.yaml` are the short form of these rules and this guide is the long form.
If the two ever disagree, fix the schema, bump its `version`, and relabel.

Two people applying this guide to the same row should pick the same three answers. If they
don't, the guide is missing a rule, so add it here.

## 1. The record

One record is one reported POS line plus the master-data entry the FSM looked up for it:

```
distributor=<name>; period=<period>; customer=<name as reported>; country=<ISO-2>; part=<part number>; qty=<quantity>; unit_price=<price> <currency>; ref=<invoice/line ref> || master: customer=<canonical name>; spq=<int>; moq=<int>; restricted=<yes|no>; seen_refs=<refs already reported, comma-separated, or none>
```

* The FSM renders the record, so the keys and their order never change. Values can be messy, empty or wrong.
* Left of `||` is what the distributor reported. Right of `||` is what the vendor knows.
* **Canonical forms**, the formats a clean row uses:
  * period: `YYYY-MM`, e.g. `2026-09`
  * qty: a plain integer with no separators, spaces or units, e.g. `2500`
  * unit_price: a dot-decimal number, a space, then an upper-case ISO currency code, e.g. `2.31 USD` or `41.90 TRY`
  * part: upper case and hyphen-separated with no spaces, e.g. `WM-S3-N8R8`
  * country: an upper-case ISO 3166 alpha-2 code, e.g. `PL`
  * customer: exactly the master `customer` value, including case, spacing and punctuation
* Master `customer=unknown`, `spq=?` or `moq=?` means the lookup failed.
* `seen_refs` lists references already reported earlier. `none` means no earlier references.

## 2. Question `row_action`

Work down this list. **The first rule that applies is the answer.**

| # | Option | Test |
|---|--------|------|
| 1 | `compliance_review` | master `restricted=yes` |
| 2 | `no_op` | the row is not a sales line |
| 3 | `ask_human` | it is a sales line, but it can't be trusted or master lookup failed |
| 4 | `duplicate` | the row's ref is already in `seen_refs` |
| 5 | `pack_qty_issue` | qty breaks SPQ or MOQ |
| 6 | `vendor_rename` | the customer name differs from master by more than formatting |
| 7 | `normalize` | some value isn't in canonical form, but every value is right |
| 8 | `commit` | none of the above |

### 2.1 `compliance_review`: master `restricted=yes`

* Only the master flag counts. A country or customer name that *sounds* sensitive with `restricted=no` is **not** compliance. Label it by the other rules.
* It beats everything: a restricted row that is also a duplicate, a pack break, a subtotal or unreadable is still `compliance_review`.

### 2.2 `no_op`: the row is not a sales line

There is nothing to ingest. The pipeline skips it. Types:

* **header**: the values are column titles (`customer=Customer Name; part=Part No; qty=Qty`), e.g. a header repeated on every page
* **blank / separator**: every reported value is empty or filler (`-----`, `=====`, `n/a`). There's no qty and no ref.
* **note**: free text with no quantity to report (`*** end of report ***`, `continued on next page`, `prices exclude VAT`)
* **subtotal**: a sum over one customer or part (`part=SUBTOTAL`, `customer=Total Acme Controls`, qty is a sum, ref empty or `SUBTOTAL`)
* **grand total**: a sum over the whole report (`customer=GRAND TOTAL`, `Report total`)

Totals and notes count in any language: `TOTAL GENERAL`, `GENEL TOPLAM`, `Gesamtsumme` and
`ARA TOPLAM` (subtotal) are total lines just as their English forms are.

Master lookups usually fail on these rows (`customer=unknown; spq=?; moq=?`). That doesn't make them `ask_human`, because rule 2 comes before rule 3.

Near miss: a row with *some* sales content (a qty and a ref, a real part) but missing other
fields is a broken sales line, so it's `ask_human`, not `no_op`.

### 2.3 `ask_human`: a sales line that can't be trusted

Any one of these is enough:

* **missing**: customer, part, qty, unit_price, ref, period or country is empty
* **impossible**: qty zero, negative, fractional (`1.5`, `2,75`) or non-numeric (`see attachment`); price zero, negative or non-numeric (`TBD`); month `13` or `00`; a period that is not a month (`Q3 2026`, `FY26`, `2026`)
* **contradictory or garbled**: two part numbers, two quantities (`1000/1200`), a note in a value field in any language (`ref=see email`, `ref=ver nota: pago pendiente`), a value cut off mid-way with or without `…` (`customer=Medisonda Tijuana S. de R.L. d`, `unit_price=1.9…`)
* **lookup failed**: master `customer=unknown`, `spq=?` or `moq=?` on an otherwise normal sales line
* **nothing else fits**: e.g. a return or credit line

### 2.4 `duplicate`: the ref has been reported before

* The row's `ref` matches an entry in `seen_refs`, ignoring letter case and extra spaces (`inv-88001/2` matches `INV-88001/2`).
* A *similar* ref is not a match: `INV-88001/12` doesn't match `INV-88001/2`, and `IE/SO/7720` doesn't match `IE/SO/7702`.
* It beats pack, rename and format issues, because the row is dropped and never committed.

### 2.5 `pack_qty_issue`: qty breaks SPQ or MOQ

* Read qty after normalizing its format, then check that **qty is a whole multiple of SPQ and qty ≥ MOQ**. If either check fails, it's a pack break.
* Examples, all with SPQ 250 and MOQ 500: `1000` and `500` pass. `1100`, `250` and `1010` fail.
* A qty exactly at MOQ that is a multiple of SPQ is fine, so the row is `commit`.
* It beats rename and format issues: `qty=1.100` (that is 1100) with SPQ 250 is `pack_qty_issue`, not `normalize`.

**Reading qty formats.** Qty always counts whole pieces, so:

* `.` or `,` followed by exactly three digits is a thousands separator: `1.000` = `1,000` = `1 000` = 1000. Indian grouping `1,00,000` = 100000.
* Trailing zero decimals are fine: `1000.00` = `1.000,00` = 1000.
* Units and suffixes are formatting: `4000pcs`, `4,000 EA`, `1.000 Stk.`, `1,000 Nos.`, `2.5k` (= 2500).
* Indian grouping may carry zero decimals too: `1,00,000.00` = 100000.
* Any other fractional qty (`1.5`, `2,75`) is impossible, so the row is `ask_human`.

### 2.6 `vendor_rename`: the customer name differs from master by more than formatting

Master matched a customer, but the reported name is a different string after ignoring case,
spacing, punctuation and accents.

* **clear**, where it is obviously the same entity: an abbreviation (`ACME CTRL LTD`), a legal suffix changed or dropped (`Ltd` → `Limited`, `Pvt Ltd` → `Private Limited`, `A.S.` → `Anonim Sirketi`), a typo or transposed letters, a translated legal form, a dropped generic word (`Co.`, `Intl`)
* **uncertain**, where the link is only loosely supported: the names share at most one distinctive word (`Shakti Energy Solutions` vs `Shakti Meters Pvt Ltd`), a completely different name for the matched customer (a possible rebrand, parent or acquirer: `Helix Group Holdings` vs `Bluefjord Audio ApS`), or a name that points to a branch or subsidiary (`Acme Controls Asia Pte Ltd` vs `Acme Controls Ltd`)

### 2.7 `normalize`: right values in the wrong format

At least one value is not in canonical form, *and* the value is right once normalized:

* period: `30.09.2026`, `Sep-26`, `2026/09`, `09/2026`, `2026-9`, `September 2026`, `2026-09-30`
* qty: separators, units or spaces, as in 2.5. The normalized qty must still pass SPQ and MOQ.
* unit_price: a decimal comma (`2,31`), lower case (`usd`), a symbol or word for the currency (`$2.31`, `2,31 Euro`, `41,90 TL`, `Rs. 105`), the currency first (`USD 2.31`), spaces or apostrophes between thousands (`5 220,00 HUF`)
* part: lower or mixed case, spaces instead of hyphens, missing hyphens (`wm-s3-n8r8`, `WM S3 N8R8`)
* country: lower case (`pl`) or the full name (`Poland`)
* customer: different **only** in case, spacing, punctuation or accents (`ACME CONTROLS LTD`, `Acme Controls Ltd.`, `Vega Beyaz Eşya A.Ş.` vs `Vega Beyaz Esya A.S.`)
* stray whitespace inside values

### 2.8 `commit`: clean

Every value is present and canonical, the customer is identical to master, qty passes SPQ and
MOQ, the ref is not in `seen_refs`, and `restricted=no`.

A customer whose real name contains words like `Total`, `Check` or `TBC` (`Total Power
Solutions Ltd`, `Check Point Systems`) is still a customer. Read the row, not the word.

These are **not** problems. Don't flag them:

* any upper-case currency code (`EUR`, `TRY`, `INR`, `SGD`, ...)
* any country code, since only the master flag matters for compliance (`NA` is Namibia, not "n/a")
* the price level, because master has no price to compare against
* `seen_refs` listing *other* refs
* a qty far above MOQ
* how the distributor writes its own name

## 3. Question `needs_human`

`yes` when the winning `row_action` needs a person:

| row_action | needs_human |
|---|---|
| `pack_qty_issue`, `compliance_review`, `ask_human` | `yes` |
| `vendor_rename`, uncertain | `yes` |
| `vendor_rename`, clear | `no` |
| `commit`, `normalize`, `duplicate`, `no_op` | `no` |

A duplicate is dropped automatically, so its other problems don't need review.

## 4. Question `severity`

How much committing the row **unchanged** would distort the POS report. When several issues
apply, take the highest.

| severity | meaning | typical rows |
|---|---|---|
| 0 | no effect | `commit` |
| 1 | cosmetic only | `normalize`; `no_op` header, blank, separator, note |
| 2 | minor number drift for one line | `pack_qty_issue`; clear `vendor_rename` |
| 3 | wrong totals for one customer or part | `duplicate`, `ask_human`, uncertain `vendor_rename`, `no_op` subtotal |
| 4 | wrong totals across the report, or compliance exposure | `compliance_review`, `no_op` grand total |

## 5. The eleven valid label combinations

| row_action | needs_human | severity | when |
|---|---|---|---|
| commit | no | 0 | clean |
| normalize | no | 1 | format only |
| vendor_rename | no | 2 | clear rename |
| vendor_rename | yes | 3 | uncertain rename |
| pack_qty_issue | yes | 2 | SPQ or MOQ break |
| compliance_review | yes | 4 | restricted |
| duplicate | no | 3 | ref already seen |
| ask_human | yes | 3 | untrustworthy sales line |
| no_op | no | 1 | header, blank, separator, note |
| no_op | no | 3 | subtotal |
| no_op | no | 4 | grand total |

A label outside this table is almost always a labelling mistake. `s1gate audit` flags them.
Synthetic generators should not write a pack break together with an uncertain rename. By the
highest-severity rule that row would be `pack_qty_issue/yes/3`, a combination too rare to teach.

## 6. Hard pairs: rows that differ in one field and flip the answer

| change | from | to |
|---|---|---|
| qty 1000 → 1100 (SPQ 250) | commit | pack_qty_issue |
| qty at MOQ → one SPQ below MOQ | commit | pack_qty_issue |
| ref added to seen_refs | commit | duplicate |
| seen ref differs by one digit | duplicate | commit |
| restricted no → yes | commit / any | compliance_review |
| customer `ACME CONTROLS LTD` → `ACME CTRL LTD` | normalize | vendor_rename (clear) |
| customer abbreviation → unrelated brand | vendor_rename/no/2 | vendor_rename/yes/3 |
| subtotal line → sales line with missing price | no_op | ask_human |
| `qty=1.000` → `qty=1.5` | normalize | ask_human |
| period `2026-09` → `30.09.2026` | commit | normalize |
| period `2026-12` → `2026-13` | commit | ask_human |
| master lookup ok → `customer=unknown` | commit | ask_human |

## 7. Checklist for labellers

1. Read master first: `restricted`, `seen_refs`, `spq` and `moq`, and whether the lookup failed.
2. Ask whether it is a sales line at all.
3. Check every field for missing, impossible or contradictory values.
4. Compare the ref with `seen_refs`.
5. Normalize qty, then check SPQ and MOQ.
6. Compare the customer name with master (formatting only, clear rename, or uncertain).
7. Check the formats.
8. Read `needs_human` and `severity` off sections 3 to 5.

For bank data, mask account numbers and counterparty names that the decision doesn't need,
and keep the labelled files inside the bank network.
