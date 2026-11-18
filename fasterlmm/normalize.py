"""
Phenotype normalisations.  Just RINT (rank-inverse-normal transform) for now -- the workhorse for non-gaussian
phenos like raw transcript counts where outliers tank the perm threshold
Canonical Blom (c=3/8) to match Victor's R helper
(workflow/scripts/rank_based_Inverse_Normal_Transformation.R) so this is a drop-in for the starlight pipeline
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

RINT_FILENAME = "pheno_rint.tsv"


def rint_columns(Y: np.ndarray, *, c: float = 3 / 8, ties: str = "average") -> np.ndarray:
    """
    Per-column Blom RINT.  out = qnorm((rank - c) / (n - 2c + 1)) with c=3/8 by default, ties.method='average'
    to match R's rank
    NaNs round-trip (not ranked, stay NaN in the output)
    Operates along axis 0, accepts 1D or 2D
    """
    Y = np.asarray(Y, dtype = np.float64)
    if Y.ndim == 1:
        return _rint_1d(Y, c, ties)
    if Y.ndim != 2:
        raise ValueError(f"rint_columns expects 1D or 2D, got shape {Y.shape}")
    out = np.full_like(Y, np.nan)
    for j in range(Y.shape[1]):
        out[:, j] = _rint_1d(Y[:, j], c, ties)
    return out


def _rint_1d(y: np.ndarray, c: float, ties: str) -> np.ndarray:
    """
    Blom RINT of one column, NaNs left untouched
    Ranks only the finite entries, maps them trough the normal quantile, scatters them back into a nan-filled
    output so the missing cells round-trip
    """
    out = np.full_like(y, np.nan, dtype = np.float64)
    mask = ~np.isnan(y)
    n = int(mask.sum())
    if n == 0:
        return out
    ranks = rankdata(y[mask], method = ties)
    out[mask] = norm.ppf((ranks - c) / (n - 2 * c + 1))
    return out


def write_rint_matrix(iid: list[str], names: list[str], Y: np.ndarray, path: str | Path) -> Path:
    """
    Write the transformed phenotype table in the input layout: Strain column, then one column per phenotype,
    tab separated, strains in input order, missing values left empty (read_phen reads them back as NaN)
    Goes trough a .tmp sibling and a rename so a reader never sees a half written file
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents = True, exist_ok = True)
    # pandas prints floats with the shortest repr that round trips, pyarrow's writer was one ulp off
    pd.DataFrame(Y, index = pd.Index(iid, name = "Strain"), columns = [str(n) for n in names]).to_csv(
        tmp, sep = "\t", na_rep = "")
    tmp.replace(path)
    return path


# TEXT PHENOTYPES -------

# words that read as missing on top of the pandas defaults (empty, NA, NaN, ...)
TEXT_NA = ["unknown", "Unknown", "UNKNOWN", "?", "n/a", "N/A", "none given", "-"]
CODEBOOK_FILENAME = "pheno_codes.tsv"

# known words, lowercase, grouped by where they sit on a no < partial < yes scale
_TIERS = (
    {"no", "n", "false", "f", "absent", "negative", "neg", "none", "non", "wild type", "wt", "low", "0"},
    {"partial", "intermediate", "inter", "some", "weak", "maybe", "mixed", "mid", "medium", "variable", "leaky"},
    {"yes", "y", "true", "t", "present", "positive", "pos", "high", "mutant", "1"},
)


def parse_levels(spec: str | None) -> dict:
    """
    --levels "col=no,partial,yes;col2=low,high" into {col: [labels low to high]}
    """
    out = {}
    for part in (spec or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--levels entry '{part}' needs the form column=low,mid,high")
        col, labs = part.split("=", 1)
        out[col.strip()] = [x.strip() for x in labs.split(",") if x.strip()]
        if len(out[col.strip()]) < 2:
            raise ValueError(f"--levels for {col.strip()} needs at least 2 labels")
    return out


def encode_text_column(col: pd.Series, name: str, order: list[str] | None = None) -> tuple[pd.Series, list[str], bool]:
    """
    Text column to numeric codes 0..L-1, returns (codes, labels in code order, ordered)
    Order comes from --levels when given (case insensitive, every value in the column must be listed), else from
    the known yes/no/partial words (synonyms like y / yes / true share a code). A column with words outside that
    vocabulary has no meaningful order, so it comes back ordered = False with alphabetical codes and goes through
    the one-vs-rest expansion instead
    """
    s = col.astype("string").str.strip()
    key = s.str.lower()
    seen = [k for k in pd.unique(key.dropna())]
    if order:
        low = [o.lower() for o in order]
        if len(set(low)) != len(low):
            raise ValueError(f"--levels for {name} lists a label twice: {order}")
        unk = [k for k in seen if k not in low]
        if unk:
            raise ValueError(f"column {name} has value(s) {unk[:5]} not in --levels {order}")
        code = {k: i for i, k in enumerate(low)}
        # drop labels that never show up so the codes stay 0..L-1 with no holes
        used = [k for k in low if k in seen]
        code = {k: i for i, k in enumerate(used)}
        labels, ordered = [order[low.index(k)] for k in used], True
    else:
        tier = {k: next((t for t, words in enumerate(_TIERS) if k in words), None) for k in seen}
        if seen and all(t is not None for t in tier.values()):
            present = sorted(set(tier.values()))
            code = {k: present.index(t) for k, t in tier.items()}
            labels = ["/".join(sorted(k for k in seen if code[k] == c)) for c in range(len(present))]
            ordered = True
        else:
            used = sorted(seen)
            code = {k: i for i, k in enumerate(used)}
            labels, ordered = used, False
    if len(labels) < 2:
        print(f"text column {name} has only {len(labels)} distinct value, it will be skipped as constant",
              file = sys.stderr, flush = True)
    codes = key.map(code).astype("float64")
    return codes, labels, ordered


def write_codebook(rows: list, path: str | Path) -> Path | None:
    """
    pheno_codes.tsv: one row per text column and level, so a beta can be read back in words
    columns are column, code, label, n_strains, how. An ordered column has effect = change in the coded value per
    allele (up is toward the last label), an unordered one has a 0/1 indicator per label, effect = change in the
    chance of that label
    """
    if not rows:
        return None
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    pd.DataFrame(rows, columns = ["column", "code", "label", "n_strains", "how"]).to_csv(path, sep = "\t", index = False)
    return path


_NAME_OK = re.compile(r"^[A-Za-z0-9._-]+$")


def _level_tag(v: float) -> str:
    """
    Level value as a filesystem safe tag, 1.0 -> 1, -1 -> m1, 0.5 -> 0.5, -0.5 -> m0.5
    """
    s = str(int(v)) if float(v) == int(v) else repr(float(v))
    return s.replace("-", "m")


def discrete_mask(Y: np.ndarray, max_levels: int = 3) -> np.ndarray:
    """
    (P,) bool, True for columns with 2 up to max_levels distinct finite values
    Constant and all-NaN columns are not discrete here, they are the guard's problem
    """
    Y = np.asarray(Y, dtype = np.float64)
    out = np.zeros(Y.shape[1], dtype = bool)
    for j in range(Y.shape[1]):
        v = Y[:, j]
        n = len(np.unique(v[~np.isnan(v)]))
        out[j] = 2 <= n <= max_levels
    return out


def _label_tag(lab: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(lab)).strip("_") or "x"


def expand_categorical(Y: np.ndarray, names: list[str], cat_names: list[str],
                       labels: dict | None = None) -> tuple[np.ndarray, list[str], list[str]]:
    """
    One-vs-rest indicators for unordered categorical columns, parent dropped and replaced in place
    L levels give L indicators named name__lvl<tag>, but 2 levels give only the higher one (the other is the exact
    complement, same F and p). NaN in the parent is NaN in every indicator so siblings share an NA pattern
    labels = {column: [text label per code]} names the indicators after the text (name__yes) instead of the code
    Returns (Y, names, indicator_names)
    """
    Y = np.asarray(Y, dtype = np.float64)
    missing = [c for c in cat_names if c not in names]
    if missing:
        raise ValueError(f"--categorical column(s) not in the pheno table: {missing}")
    cols, new_names, made = [], [], []
    for j, name in enumerate(names):
        if name not in cat_names:
            cols.append(Y[:, j])
            new_names.append(str(name))
            continue
        v = Y[:, j]
        levels = np.unique(v[~np.isnan(v)])
        if len(levels) < 2:
            raise ValueError(f"--categorical column {name} has {len(levels)} distinct value(s), need at least 2")
        keep = levels[-1:] if len(levels) == 2 else levels
        for lv in keep:
            lab = (labels or {}).get(name)
            new = f"{name}__{_label_tag(lab[int(lv)])}" if lab else f"{name}__lvl{_level_tag(lv)}"
            if not _NAME_OK.match(new):
                raise ValueError(f"indicator name {new} has characters that are not filesystem safe")
            ind = np.where(np.isnan(v), np.nan, (v == lv).astype(np.float64))
            cols.append(ind)
            new_names.append(new)
            made.append(new)
    if len(set(new_names)) != len(new_names):
        dup = sorted({n for n in new_names if new_names.count(n) > 1})
        raise ValueError(f"expanded pheno names collide with existing columns: {dup}")
    return np.column_stack(cols), new_names, made


def prepare_pheno(pheno, *, rint: bool = True, rint_discrete: bool = True, categorical: list[str] | None = None,
                  missing_code: float | None = None) -> dict:
    """
    One stop for the pheno table before alignment, edits pheno in place (Y and names), returns info
    Order is missing code -> -9 check -> categorical expansion -> RINT on everything not marked raw
    Raw (RINT skipped) columns are the categorical indicators plus, with rint_discrete off, every column with 2-3
    levels. Default flags (rint on, rint_discrete on, nothing categorical) give exactly rint_columns(Y) as before
    info = {"codebook": rows for pheno_codes.tsv, "raw": names left untransformed, "discrete": names with 2-3 levels, "expanded": indicator names}
    """
    Y = np.asarray(pheno.Y, dtype = np.float64).copy()
    names = [str(n) for n in pheno.names]
    if missing_code is not None:
        Y[Y == missing_code] = np.nan
    expanded: list[str] = []
    book = getattr(pheno, "codebook", {}) or {}
    cats = list(dict.fromkeys(list(categorical or []) + list(getattr(pheno, "auto_categorical", []) or [])))
    rows = []
    for col, labs in book.items():
        j = names.index(col)
        for k, lab in enumerate(labs):
            rows.append([col, k, lab, int(np.sum(Y[:, j] == k)), f"ordered code {k}"])
    if cats:
        Y, names, expanded = expand_categorical(Y, names, cats, labels = book)
        for r in rows:
            if r[0] in cats:
                r[4] = f"indicator column {r[0]}__{_label_tag(r[2])}" if len(book[r[0]]) > 2 or r[1] == 1 else \
                    "complement of the other label, not scanned separately"
    disc = discrete_mask(Y)
    discrete = [n for n, d in zip(names, disc) if d]
    raw_mask = np.array([n in set(expanded) for n in names], dtype = bool)
    if not rint_discrete:
        raw_mask |= disc
    # plink codes missing as -9, a raw 3-or-fewer level column holding it would eat -9 as a real (huge) level
    # with rint off nothing gets transformed, so every discrete column is effectively raw
    if missing_code is None:
        untouched = raw_mask | disc if not rint else raw_mask
        bad = [n for n, d, r, j in zip(names, disc, untouched, range(len(names)))
               if d and r and np.any(Y[:, j] == -9)]
        if bad:
            raise ValueError(f"discrete column(s) with -9 (plink missing code) would be read as a real value: "
                             f"{bad[:5]}, pass --missing-code -9 to treat it as missing")
    if rint:
        keep = ~raw_mask
        if keep.any():
            Y[:, keep] = rint_columns(Y[:, keep])
        rinted_disc = [n for n in discrete if n not in set(expanded)]
        if rint_discrete and rinted_disc:
            print(f"{len(rinted_disc)} pheno column(s) with 2-3 distinct values are RINT transformed (first: "
                  f"{rinted_disc[:3]}), pass --no-rint-discrete to leave them as raw numbers", file = sys.stderr,
                  flush = True)
    raw = [n for n, r in zip(names, raw_mask) if r]
    if raw:
        print(f"{len(raw)} pheno column(s) left untransformed (first: {raw[:3]})", file = sys.stderr, flush = True)
    pheno.Y, pheno.names = Y, names
    return {"raw": raw, "discrete": discrete, "expanded": expanded, "codebook": rows}


def degenerate_columns(Ysub: np.ndarray, min_minor: int = 10) -> np.ndarray:
    """
    (P,) bool, True where a raw discrete column on this strain subset has under 2 levels or a minor level with
    fewer than min_minor strains
    Needed because F is scale free, a constant column minus its fit leaves 1e-16 noise that scans like a real
    pheno and throws plausible hits
    """
    Ysub = np.asarray(Ysub, dtype = np.float64)
    bad = np.zeros(Ysub.shape[1], dtype = bool)
    for j in range(Ysub.shape[1]):
        v = Ysub[:, j]
        _, cnt = np.unique(v[~np.isnan(v)], return_counts = True)
        bad[j] = len(cnt) < 2 or cnt.min() < min_minor
    return bad


def rint(y: np.ndarray, *, ties_method: str = "average") -> np.ndarray:
    """
    Thin alias for rint_columns under the old name
    Kept so anything that imported rint from the previous (unused) version still resolves, can drop it once
    nothing points here
    """
    return rint_columns(y, ties = ties_method)
