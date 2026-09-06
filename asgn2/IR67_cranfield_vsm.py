#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Cranfield Ranked Retrieval with PyTerrier -- Sparse Vector Space Models
================================================================================

A modular, production-ready pipeline covering the five standard IR stages:

    1. Data ingestion          -> load_collection()      (reads cran.tar.gz)
    2. Preprocessing/indexing  -> TerrierIndexBuilder     (tokenise/stop/stem)
    3. Retrieval               -> build_systems()         (sparse VSM only)
    4. Evaluation + timing     -> Evaluator, Timer        (MAP/nDCG/P@k/...)
    5. Output                  -> write_trec_run()        (TREC results file)

MODEL SCOPE
-----------
Only sparse vector space models are used, as required by the assignment:

  * Terrier `Tf`             -- raw term frequency (sanity baseline)
  * Terrier `TF_IDF`         -- tf.idf with Robertson tf normalisation
  * Terrier `LemurTF_IDF`    -- log-tf.idf variant
  * Terrier `CoordinateMatch`-- coordination level matching baseline
  * `SmartVSM`               -- classical SMART ddd.qqq cosine VSM implemented
                                from scratch (lnc.ltc, ltc.ltc, anc.apc, lnu.ltu
                                pivoted-length normalisation, ...)
  * `RocchioExpander`        -- Rocchio pseudo-relevance feedback, the
                                vector-space-native query expansion method

No BM25/DFR/language models, no learning-to-rank, no dense or neural retrieval.

QUICK START
-----------
    pip install python-terrier pandas          # requires a JDK (>= 11) on PATH
    python cranfield_vsm.py --data ./cran.tar.gz --workdir ./work

    # skip the grid search and just run the tuned default configuration
    python cranfield_vsm.py --data ./cran.tar.gz --skip-tuning

    # score a fresh (unseen) query file, cran.qry format or "qid<TAB>text"
    python cranfield_vsm.py --data ./cran.tar.gz --test-queries ./unseen.qry

OUTPUTS (under --workdir)
-------------------------
    runs/results.txt                    TREC run for all supplied queries
    runs/results_test_queries.txt       TREC run for --test-queries (if given)
    runs/*.txt                          one run file per evaluated system
    results/*.csv                       per-stage effectiveness + timing tables
    results/best_config.json            the winning configuration
    results/query_id_map.tsv            file-.I  <->  ordinal qid mapping

NOTES ON THE CRANFIELD FILES (verified against the distributed collection)
-------------------------------------------------------------------------
  * cran_all.1400 holds 1400 records with .I/.T/.A/.B/.W fields.  Records 240,
    576 and 578 repeat a field marker; the parser therefore *appends* to the
    currently open field instead of assuming one marker per field.
  * The .W abstract normally restarts with the title text, so the title is
    de-duplicated before an explicit, tunable title boost is applied.
  * cran.qry holds 225 topics but its .I values are NOT contiguous (001, 002,
    004, ... 365) and the lines are CRLF-terminated.  cranqrel keys judgements
    on the ordinal position 1..225, so topics are renumbered by position; the
    original identifiers are preserved in results/query_id_map.tsv.
  * cranqrel grades are 1 (complete answer) .. 4 (minimal interest), plus a
    -1 code.  Gains are inverted to label = 5 - grade so that 1 -> 4 and
    4 -> 1; -1 is treated as non-relevant by default (--minus-one-relevant
    flips this).  Binary measures count label > 0 as relevant.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

# --------------------------------------------------------------------------- #
# PyTerrier is imported lazily-tolerantly: the hand-written SMART VSM half of
# the pipeline runs even on a machine without a JVM, which makes the script
# easy to debug before Terrier is available.
# --------------------------------------------------------------------------- #
try:
    import pyterrier as pt

    PT_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    pt = None
    PT_AVAILABLE = False


# =========================================================================== #
# 0. COMPATIBILITY SHIMS (PyTerrier 0.8 ... 1.x moved several symbols)
# =========================================================================== #

def _pt_symbol(name: str):
    """Return `pt.terrier.<name>` or `pt.<name>`, whichever exists."""
    if not PT_AVAILABLE:
        return None
    for holder in (getattr(pt, "terrier", None), pt):
        if holder is not None and hasattr(holder, name):
            return getattr(holder, name)
    return None


def init_pyterrier(verbose: bool = True) -> None:
    """Start the JVM exactly once, tolerating the 0.x and 1.x entry points."""
    if not PT_AVAILABLE:
        raise RuntimeError(
            "python-terrier is not installed. Run: pip install python-terrier\n"
            "A Java Development Kit (>= 11) must also be on the PATH."
        )
    started = False
    for probe in (
        lambda: pt.java.started(),          # PyTerrier >= 0.11
        lambda: pt.started(),               # PyTerrier <= 0.10
    ):
        try:
            started = bool(probe())
            break
        except Exception:
            continue
    if not started:
        for starter in (
            lambda: pt.java.init(),
            lambda: pt.init(),
        ):
            try:
                starter()
                started = True
                break
            except Exception:
                continue
    if verbose:
        ver = getattr(pt, "__version__", "unknown")
        print(f"[init] PyTerrier {ver} ready (JVM started={started})")


# Transformer base class, resolved once so `VSMTransformer` can subclass it.
if PT_AVAILABLE:
    _TRANSFORMER_BASE = (
        getattr(pt, "Transformer", None)
        or getattr(getattr(pt, "transformer", None), "TransformerBase", None)
        or object
    )
else:
    _TRANSFORMER_BASE = object


# =========================================================================== #
# 1. DATA INGESTION
# =========================================================================== #

DOC_ID_RE = re.compile(r"^\.I\s+(\d+)\s*$")
FIELD_MARKERS = {".T": "title", ".A": "author", ".B": "bib", ".W": "body"}


@dataclass
class Document:
    docno: str
    title: str
    author: str
    bib: str
    body: str


@dataclass
class Topic:
    qid: str          # ordinal identifier used for evaluation (1..N)
    file_id: str      # the raw .I value found in the query file
    text: str


def _squash(lines: Sequence[str]) -> str:
    """Join raw field lines into a single whitespace-normalised string."""
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def extract_collection(source: str, workdir: str) -> str:
    """
    Accept either `cran.tar.gz` or an already-extracted directory and return the
    directory that holds cran_all.1400 / cran.qry / cranqrel.
    """
    if os.path.isdir(source):
        return source
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Collection not found: {source}")
    if not tarfile.is_tarfile(source):
        # A bare file such as cran_all.1400 was passed: use its parent folder.
        return os.path.dirname(os.path.abspath(source)) or "."

    dest = os.path.join(workdir, "cran")
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(source, "r:*") as tar:          # auto-detects gz/bz2/xz
        for member in tar.getmembers():
            # Defensive extraction: refuse absolute paths and traversal.
            name = os.path.normpath(member.name).lstrip(os.sep)
            if name.startswith("..") or os.path.isabs(member.name):
                continue
            member.name = os.path.basename(name)      # flatten the archive
            if member.isfile():
                tar.extract(member, path=dest)
    print(f"[ingest] extracted {source} -> {dest}")
    return dest


def _locate(directory: str, *candidates: str) -> str:
    for cand in candidates:
        path = os.path.join(directory, cand)
        if os.path.isfile(path):
            return path
    # Fall back to a case-insensitive scan of the directory tree.
    wanted = {c.lower() for c in candidates}
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if fname.lower() in wanted:
                return os.path.join(root, fname)
    raise FileNotFoundError(
        f"None of {candidates} were found under {directory}"
    )


def parse_documents(path: str) -> List[Document]:
    """
    Parse the SGML-ish cran_all.1400 file.

    Field content is *appended* to whichever field is currently open, so the
    duplicated markers in records 240/576/578 do not silently drop text.
    """
    docs: List[Document] = []
    buf: Optional[Dict[str, List[str]]] = None
    docno: Optional[str] = None
    field: Optional[str] = None

    def flush() -> None:
        if docno is None or buf is None:
            return
        title = _squash(buf["title"])
        body = _squash(buf["body"])
        docs.append(
            Document(
                docno=docno,
                title=title,
                author=_squash(buf["author"]),
                bib=_squash(buf["bib"]),
                body=body,
            )
        )

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            m = DOC_ID_RE.match(line.strip())
            if m:
                flush()
                docno = str(int(m.group(1)))
                buf = {"title": [], "author": [], "bib": [], "body": []}
                field = None
                continue
            marker = line.strip()
            if marker in FIELD_MARKERS:
                field = FIELD_MARKERS[marker]
                continue
            if buf is not None and field is not None and line.strip():
                buf[field].append(line.strip())
    flush()

    if not docs:
        raise ValueError(f"No documents parsed from {path}")
    print(f"[ingest] parsed {len(docs)} documents from {os.path.basename(path)}")
    return docs


def parse_topics(path: str, qid_mode: str = "auto") -> List[Topic]:
    """
    Parse a cran.qry-style file (.I / .W) or a two-column `qid<TAB>text` file.

    qid_mode:
      "ordinal" -- number topics 1..N by position (matches cranqrel)
      "file"    -- keep the .I values verbatim
      "auto"    -- keep the file values when they are already 1..N contiguous,
                   otherwise fall back to ordinal numbering
    """
    raw = open(path, "r", encoding="utf-8", errors="replace").read()
    entries: List[Tuple[str, str]] = []

    if ".I" in raw:
        file_id: Optional[str] = None
        chunk: List[str] = []
        capture = False
        for line in raw.splitlines():
            line = line.rstrip("\r\n")
            m = DOC_ID_RE.match(line.strip())
            if m:
                if file_id is not None:
                    entries.append((file_id, _squash(chunk)))
                file_id = str(int(m.group(1)))
                chunk, capture = [], False
                continue
            marker = line.strip()
            if marker in FIELD_MARKERS:
                capture = marker == ".W"
                continue
            if file_id is not None and capture and line.strip():
                chunk.append(line.strip())
        if file_id is not None:
            entries.append((file_id, _squash(chunk)))
    else:
        for line in raw.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) == 2:
                entries.append((str(parts[0]).strip(), parts[1].strip()))

    if not entries:
        raise ValueError(f"No topics parsed from {path}")

    file_ids = [e[0] for e in entries]
    contiguous = file_ids == [str(i) for i in range(1, len(entries) + 1)]
    use_file_ids = qid_mode == "file" or (qid_mode == "auto" and contiguous)

    topics = [
        Topic(
            qid=(fid if use_file_ids else str(pos)),
            file_id=fid,
            text=text,
        )
        for pos, (fid, text) in enumerate(entries, start=1)
    ]
    scheme = "file .I values" if use_file_ids else "ordinal position"
    print(
        f"[ingest] parsed {len(topics)} topics from {os.path.basename(path)} "
        f"(qids = {scheme})"
    )
    return topics


def parse_qrels(path: str, minus_one_relevant: bool = False) -> pd.DataFrame:
    """
    Read cranqrel into a TREC-style frame with columns qid / docno / label.

    Cleverdon's grades run 1 (complete answer) .. 4 (minimal interest), so gains
    are inverted with label = 5 - grade.  Grade -1 becomes label 1 when
    `minus_one_relevant` is set and 0 (non-relevant) otherwise.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            qid, docno, grade = parts[0], parts[1], int(parts[2])
            if grade == -1:
                label = 1 if minus_one_relevant else 0
            else:
                label = max(0, 5 - grade)     # 1->4, 2->3, 3->2, 4->1
            rows.append((str(int(qid)), str(int(docno)), label))

    qrels = pd.DataFrame(rows, columns=["qid", "docno", "label"])
    qrels = qrels.sort_values(["qid", "docno"]).drop_duplicates(
        subset=["qid", "docno"], keep="last"
    ).reset_index(drop=True)
    positives = int((qrels["label"] > 0).sum())
    print(
        f"[ingest] parsed {len(qrels)} judgements over "
        f"{qrels['qid'].nunique()} topics ({positives} relevant)"
    )
    return qrels


def load_collection(
    source: str, workdir: str, qid_mode: str = "ordinal",
    minus_one_relevant: bool = False,
) -> Tuple[List[Document], List[Topic], pd.DataFrame]:
    """One-call ingestion of documents, topics and relevance judgements."""
    directory = extract_collection(source, workdir)
    docs = parse_documents(_locate(directory, "cran_all.1400", "cran.all.1400"))
    topics = parse_topics(_locate(directory, "cran.qry", "cran.qry.txt"), qid_mode)
    qrels = parse_qrels(
        _locate(directory, "cranqrel", "cranqrel.txt", "cran.rel"),
        minus_one_relevant,
    )
    return docs, topics, qrels


# =========================================================================== #
# 2. PREPROCESSING
# =========================================================================== #

# Terrier's default English stopword list (trimmed to the standard core).  Used
# by the hand-written VSM so that it mirrors what Terrier removes internally.
STOPWORDS = set("""
a about above across after afterwards again against all almost alone along
already also although always am among amongst an and another any anyhow anyone
anything anyway anywhere are around as at back be became because become becomes
becoming been before beforehand behind being below beside besides between
beyond both but by can cannot could de did do does doing done down due during
each eg eight either else elsewhere enough etc even ever every everyone
everything everywhere except few first for former formerly found four from
further get give go had has have having he hence her here hereafter hereby
herein hereupon hers herself him himself his how however i ie if in inc indeed
into is it its itself keep last latter latterly least less made make many may me
meanwhile might more moreover most mostly much must my myself namely neither
never nevertheless next no nobody none noone nor not nothing now nowhere of off
often on once one only onto or other others otherwise our ours ourselves out
over own per perhaps please put rather re same see seem seemed seeming seems
several she should since six so some somehow someone something sometime
sometimes somewhere still such take than that the their them themselves then
thence there thereafter thereby therefore therein thereupon these they this
those though three through throughout thru thus to together too toward towards
under until up upon us used using various very via was we well were what
whatever when whence whenever where whereafter whereas whereby wherein whereupon
wherever whether which while whither who whoever whole whom whose why will with
within without would yet you your yours yourself yourselves
""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+")


def clean_text(text: str) -> str:
    """
    Lower-case and strip everything that is not alphanumeric.

    Cranfield uses /slashed/ emphasis, hyphenated compounds and trailing " ."
    sentence markers; all of them are reduced to token separators.  Digits are
    kept because they carry meaning here (mach 2, naca 0012, ...).
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _resolve_stemmer() -> Tuple[Callable[[str], str], str]:
    """
    Return (stem_fn, name).  Preference order: PyStemmer, NLTK, Terrier, and
    finally a conservative built-in suffix stripper so the script never dies
    just because an optional dependency is missing.
    """
    try:
        import Stemmer  # PyStemmer

        s = Stemmer.Stemmer("english")
        return (lambda w: s.stemWord(w)), "porter (PyStemmer)"
    except Exception:
        pass
    try:
        from nltk.stem import PorterStemmer

        ps = PorterStemmer()
        if ps.stem("running") == "run":
            return ps.stem, "porter (NLTK)"
    except Exception:
        pass
    if PT_AVAILABLE:
        try:
            terrier_stemmer = _pt_symbol("TerrierStemmer")
            porter = getattr(terrier_stemmer, "porter", None)
            if porter is not None and porter.stem("running") == "run":
                return porter.stem, "porter (Terrier)"
        except Exception:
            pass

    def light_stem(word: str) -> str:
        for suffix in ("ational", "iveness", "fulness", "ousness", "ization",
                       "ations", "ically", "ingly", "edly", "ing", "ies",
                       "ied", "ies", "ers", "er", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                return word[: -len(suffix)]
        return word

    return light_stem, "light (fallback)"


STEM_FN, STEMMER_NAME = _resolve_stemmer()
_STEM_CACHE: Dict[str, str] = {}


def stem_word(word: str) -> str:
    cached = _STEM_CACHE.get(word)
    if cached is None:
        cached = STEM_FN(word)
        _STEM_CACHE[word] = cached
    return cached


def tokenise(
    text: str, remove_stopwords: bool = True, stem: bool = True,
    min_length: int = 2,
) -> List[str]:
    """Full analysis chain: clean -> tokenise -> stop -> stem."""
    tokens = TOKEN_RE.findall(clean_text(text))
    out: List[str] = []
    for tok in tokens:
        if len(tok) < min_length and not tok.isdigit():
            continue
        if remove_stopwords and tok in STOPWORDS:
            continue
        out.append(stem_word(tok) if stem else tok)
    return out


@dataclass
class PreprocConfig:
    """Everything that changes what actually goes into the index."""
    stem: bool = True
    stopwords: bool = True
    title_weight: int = 2          # how many times the title is repeated
    include_author: bool = False
    include_bib: bool = False
    dedupe_title: bool = True      # drop the title copy that opens .W

    def key(self) -> str:
        return (
            f"stem{int(self.stem)}_stop{int(self.stopwords)}"
            f"_tw{self.title_weight}_a{int(self.include_author)}"
            f"_b{int(self.include_bib)}_d{int(self.dedupe_title)}"
        )

    def label(self) -> str:
        bits = [
            "stem" if self.stem else "nostem",
            "stop" if self.stopwords else "nostop",
            f"title x{self.title_weight}",
        ]
        if self.include_author:
            bits.append("+author")
        if self.include_bib:
            bits.append("+bib")
        return ", ".join(bits)


def compose_document_text(doc: Document, cfg: PreprocConfig) -> str:
    """
    Build the indexable surface form of a record.

    The .W abstract usually repeats the title verbatim; that copy is removed
    first so that `title_weight` is the single knob controlling title emphasis.
    """
    body = doc.body
    if cfg.dedupe_title and doc.title:
        clean_body, clean_title = clean_text(body), clean_text(doc.title)
        if clean_title and clean_body.startswith(clean_title):
            # Re-slice the raw body proportionally to the matched prefix.
            approx = len(doc.title)
            body = body[approx:].lstrip(" .,;")
    parts: List[str] = []
    if doc.title:
        parts.extend([doc.title] * max(0, cfg.title_weight))
    if cfg.include_author and doc.author:
        parts.append(doc.author)
    if cfg.include_bib and doc.bib:
        parts.append(doc.bib)
    parts.append(body)
    return clean_text(" ".join(p for p in parts if p))


def topics_to_frame(topics: Sequence[Topic]) -> pd.DataFrame:
    """PyTerrier topic frame; queries are sanitised for the Terrier parser."""
    return pd.DataFrame(
        {
            "qid": [t.qid for t in topics],
            "query": [clean_text(t.text) for t in topics],
        }
    )


# =========================================================================== #
# 3a. INDEXING (Terrier)
# =========================================================================== #

class TerrierIndexBuilder:
    """Builds (and caches) one Terrier index per preprocessing configuration."""

    def __init__(self, index_root: str):
        self.index_root = index_root
        os.makedirs(index_root, exist_ok=True)
        self._cache: Dict[str, Tuple[object, float]] = {}

    @staticmethod
    def _doc_iter(docs: Sequence[Document], cfg: PreprocConfig) -> Iterator[dict]:
        for doc in docs:
            yield {"docno": doc.docno, "text": compose_document_text(doc, cfg)}

    def _make_indexer(self, path: str, cfg: PreprocConfig):
        IterDictIndexer = _pt_symbol("IterDictIndexer")
        if IterDictIndexer is None:
            raise RuntimeError("PyTerrier IterDictIndexer is unavailable")
        stemmer = "porter" if cfg.stem else "none"
        stopwords = "terrier" if cfg.stopwords else "none"
        try:
            return IterDictIndexer(
                path, meta={"docno": 8}, overwrite=True,
                stemmer=stemmer, stopwords=stopwords,
            )
        except TypeError:
            # Older PyTerrier: configure the pipeline through the property.
            indexer = IterDictIndexer(path, meta={"docno": 8}, overwrite=True)
            pipeline = ",".join(
                x for x in (
                    "Stopwords" if cfg.stopwords else None,
                    "PorterStemmer" if cfg.stem else None,
                ) if x
            )
            if hasattr(indexer, "setProperty"):
                indexer.setProperty("termpipelines", pipeline)
            return indexer

    def build(
        self, docs: Sequence[Document], cfg: PreprocConfig, force: bool = False
    ) -> Tuple[object, float]:
        """
        Build (or reuse) an index for `cfg`.

        Windows + JVM interact badly here: Terrier writes each posting file as
        `data_N.xxx` and renames to `data.xxx` at the end.  If the JVM has not
        yet released a handle on files from the *previous* build, the rename
        fails with a `java.io.IOException: Rename ... source file is still
        open`.  The defence is (i) a Python-side gc + brief sleep before every
        build to encourage handle release, and (ii) retry with a unique suffix
        if the rename still fails.
        """
        import gc as _gc

        key = cfg.key()
        if key in self._cache and not force:
            return self._cache[key]

        def _attempt(subpath: str) -> Tuple[object, float]:
            if os.path.exists(subpath):
                shutil.rmtree(subpath, ignore_errors=True)
            os.makedirs(subpath, exist_ok=True)
            indexer = self._make_indexer(subpath, cfg)
            start = time.perf_counter()
            ref = indexer.index(self._doc_iter(docs, cfg))
            elapsed = time.perf_counter() - start
            IndexFactory = _pt_symbol("IndexFactory")
            index = ref
            if not hasattr(ref, "getCollectionStatistics") and IndexFactory is not None:
                index = IndexFactory.of(ref)
            return index, elapsed

        # Give the JVM a moment to release any file handles from a prior build.
        _gc.collect()
        time.sleep(0.15)

        base_path = os.path.join(self.index_root, f"idx_{key}").replace("\\", "/")
        try:
            index, elapsed = _attempt(base_path)
        except Exception as exc:
            print(f"[index] build for {cfg.label()!r} failed ({type(exc).__name__}): "
                  f"retrying with a fresh directory ...")
            _gc.collect()
            time.sleep(0.5)
            # Retry with a unique suffix so Windows never sees stale handles
            # on the target files.
            suffix = time.strftime("%H%M%S")
            retry_path = f"{base_path}__{suffix}"
            index, elapsed = _attempt(retry_path)

        try:
            stats = index.getCollectionStatistics()
            print(
                f"[index] {cfg.label():<38} {elapsed:6.2f}s  "
                f"docs={stats.getNumberOfDocuments()} "
                f"terms={stats.getNumberOfUniqueTerms()} "
                f"postings={stats.getNumberOfPointers()}"
            )
        except Exception:
            print(f"[index] {cfg.label():<38} {elapsed:6.2f}s")

        self._cache[key] = (index, elapsed)
        return index, elapsed


def terrier_retriever(index, wmodel: str, num_results: int = 1000,
                      controls: Optional[dict] = None):
    """Instantiate a Terrier retriever for a sparse weighting model."""
    Retriever = _pt_symbol("Retriever") or _pt_symbol("BatchRetrieve")
    if Retriever is None:
        raise RuntimeError("Neither pt.terrier.Retriever nor pt.BatchRetrieve exists")
    kwargs = {"wmodel": wmodel, "num_results": num_results}
    if controls:
        kwargs["controls"] = {k: str(v) for k, v in controls.items()}
    try:
        return Retriever(index, **kwargs)
    except TypeError:
        kwargs.pop("num_results", None)
        return Retriever(index, **kwargs)


# =========================================================================== #
# 3b. RETRIEVAL -- classical SMART vector space model (implemented in full)
# =========================================================================== #

class SmartVSM:
    """
    A textbook cosine vector space model using SMART ddd.qqq notation.

        tf   : n=raw  l=1+log(tf)  a=0.5+0.5*tf/max_tf  b=boolean  L=log-average
        df   : n=none t=log(N/df)  p=max(0, log((N-df)/df))
        norm : n=none c=cosine     u=pivoted unique-term length

    The index is a plain in-memory inverted file, which is more than fast
    enough for 1400 documents and keeps every weighting decision explicit.
    """

    def __init__(
        self,
        docnos: Sequence[str],
        doc_tokens: Sequence[Sequence[str]],
        doc_scheme: str = "lnc",
        qry_scheme: str = "ltc",
        pivot_slope: float = 0.20,
    ):
        assert len(doc_scheme) == 3 and len(qry_scheme) == 3
        self.docnos = list(docnos)
        self.doc_scheme = doc_scheme
        self.qry_scheme = qry_scheme
        self.pivot_slope = pivot_slope
        self.N = len(self.docnos)

        # ---- document frequencies ------------------------------------------
        self.df: Dict[str, int] = Counter()
        counters: List[Counter] = []
        for tokens in doc_tokens:
            counter = Counter(tokens)
            counters.append(counter)
            for term in counter:
                self.df[term] += 1

        self.idf_doc = {t: self._idf(doc_scheme[1], df, self.N)
                        for t, df in self.df.items()}
        self.idf_qry = {t: self._idf(qry_scheme[1], df, self.N)
                        for t, df in self.df.items()}

        # ---- unnormalised document weights ---------------------------------
        raw_vectors: List[Dict[str, float]] = []
        unique_counts: List[int] = []
        for counter in counters:
            if not counter:
                raw_vectors.append({})
                unique_counts.append(0)
                continue
            max_tf = max(counter.values())
            avg_tf = sum(counter.values()) / len(counter)
            vec = {}
            for term, tf in counter.items():
                w = self._tf(doc_scheme[0], tf, max_tf, avg_tf) * self.idf_doc[term]
                if w != 0.0:
                    vec[term] = w
            raw_vectors.append(vec)
            unique_counts.append(len(counter))

        self.pivot = (sum(unique_counts) / len(unique_counts)) if unique_counts else 1.0

        # ---- normalisation + inverted file ---------------------------------
        self.doc_vectors: List[Dict[str, float]] = []
        self.postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for idx, vec in enumerate(raw_vectors):
            norm = self._norm_factor(doc_scheme[2], vec, unique_counts[idx])
            normalised = {t: w / norm for t, w in vec.items()} if norm else vec
            self.doc_vectors.append(normalised)
            for term, w in normalised.items():
                self.postings[term].append((idx, w))

    # ------------------------------------------------------------------ #
    # weighting primitives
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tf(letter: str, tf: float, max_tf: float, avg_tf: float) -> float:
        if tf <= 0:
            return 0.0
        if letter == "n":
            return float(tf)
        if letter == "l":
            return 1.0 + math.log(tf)
        if letter == "a":
            return 0.5 + 0.5 * tf / max_tf if max_tf else 0.0
        if letter == "b":
            return 1.0
        if letter == "L":
            return (1.0 + math.log(tf)) / (1.0 + math.log(avg_tf if avg_tf > 0 else 1.0))
        raise ValueError(f"unknown tf letter {letter!r}")

    @staticmethod
    def _idf(letter: str, df: int, N: int) -> float:
        if letter == "n":
            return 1.0
        if df <= 0:
            return 0.0
        if letter == "t":
            return math.log(N / df)
        if letter == "p":
            return max(0.0, math.log((N - df) / df)) if df < N else 0.0
        raise ValueError(f"unknown idf letter {letter!r}")

    def _norm_factor(self, letter: str, vec: Dict[str, float], unique: int) -> float:
        if letter == "n" or not vec:
            return 1.0
        if letter == "c":
            return math.sqrt(sum(w * w for w in vec.values())) or 1.0
        if letter == "u":
            s = self.pivot_slope
            return ((1.0 - s) * self.pivot + s * unique) or 1.0
        raise ValueError(f"unknown normalisation letter {letter!r}")

    # ------------------------------------------------------------------ #
    # querying
    # ------------------------------------------------------------------ #
    def query_vector(self, tokens: Sequence[str]) -> Dict[str, float]:
        counter = Counter(t for t in tokens if t in self.df)
        if not counter:
            return {}
        max_tf = max(counter.values())
        avg_tf = sum(counter.values()) / len(counter)
        vec = {
            term: self._tf(self.qry_scheme[0], tf, max_tf, avg_tf) * self.idf_qry[term]
            for term, tf in counter.items()
        }
        vec = {t: w for t, w in vec.items() if w != 0.0}
        norm = self._norm_factor(self.qry_scheme[2], vec, len(vec))
        return {t: w / norm for t, w in vec.items()} if norm else vec

    def score_vector(self, qvec: Dict[str, float], k: int = 1000
                     ) -> List[Tuple[str, float]]:
        """Term-at-a-time accumulation over the inverted file."""
        if not qvec:
            return []
        acc: Dict[int, float] = defaultdict(float)
        for term, qw in qvec.items():
            for doc_idx, dw in self.postings.get(term, ()):
                acc[doc_idx] += qw * dw
        ranked = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.docnos[i], s) for i, s in ranked if s > 0.0]

    def search(self, tokens: Sequence[str], k: int = 1000) -> List[Tuple[str, float]]:
        return self.score_vector(self.query_vector(tokens), k)


@dataclass
class RocchioConfig:
    """Rocchio pseudo-relevance feedback -- the VSM-native expansion method."""
    alpha: float = 1.0
    beta: float = 0.75
    gamma: float = 0.15
    fb_docs: int = 10
    fb_terms: int = 20
    neg_docs: int = 10
    iterations: int = 1        # >1 = multi-pass Rocchio (re-expand from expanded ranking)
    first_pass_k: int = 0      # 0 = auto (use k); larger widens candidate pool for feedback

    def label(self) -> str:
        base = (f"Rocchio(a={self.alpha},b={self.beta},g={self.gamma},"
                f"k={self.fb_docs},m={self.fb_terms}")
        if self.iterations > 1:
            base += f",it={self.iterations}"
        if self.first_pass_k:
            base += f",pool={self.first_pass_k}"
        return base + ")"


class RocchioExpander:
    """
    Query-vector shift with pseudo-relevance feedback.

    Extends the textbook Rocchio in two small but effective ways:
      * `first_pass_k` widens the initial ranking depth from which feedback docs
        are drawn -- deeper pools lift recall before the second pass.
      * `iterations > 1` re-runs the expansion on the newly ranked list, which
        often adds a further 1-2 MAP points on small, technical collections.
    """

    def __init__(self, vsm: SmartVSM, cfg: RocchioConfig):
        self.vsm = vsm
        self.cfg = cfg
        self._index_of = {d: i for i, d in enumerate(vsm.docnos)}

    # ------------------------------------------------------------------ #
    def _expand(self, qvec: Dict[str, float],
                ranking: List[Tuple[str, float]]) -> Dict[str, float]:
        """Rocchio centroid shift given a current query vector and ranking."""
        cfg = self.cfg
        expanded: Dict[str, float] = {t: cfg.alpha * w for t, w in qvec.items()}

        positives = ranking[: cfg.fb_docs]
        if positives and cfg.beta:
            share = cfg.beta / len(positives)
            for docno, _ in positives:
                for term, w in self.vsm.doc_vectors[self._index_of[docno]].items():
                    expanded[term] = expanded.get(term, 0.0) + share * w

        negatives = ranking[-cfg.neg_docs:] if cfg.neg_docs else []
        neg_set = {d for d, _ in positives}
        negatives = [d for d in negatives if d[0] not in neg_set]
        if negatives and cfg.gamma:
            share = cfg.gamma / len(negatives)
            for docno, _ in negatives:
                for term, w in self.vsm.doc_vectors[self._index_of[docno]].items():
                    expanded[term] = expanded.get(term, 0.0) - share * w

        expanded = {t: w for t, w in expanded.items() if w > 0.0}
        if cfg.fb_terms and len(expanded) > cfg.fb_terms:
            top = dict(sorted(expanded.items(), key=lambda kv: -kv[1])[: cfg.fb_terms])
            # Protect original query terms so the expansion never drops them.
            for term in qvec:
                if term in expanded:
                    top.setdefault(term, expanded[term])
            expanded = top

        norm = math.sqrt(sum(w * w for w in expanded.values())) or 1.0
        return {t: w / norm for t, w in expanded.items()}

    # ------------------------------------------------------------------ #
    def search(self, tokens: Sequence[str], k: int = 1000) -> List[Tuple[str, float]]:
        cfg = self.cfg
        qvec = self.vsm.query_vector(tokens)
        if not qvec:
            return []

        # A wider candidate pool improves the feedback centroid; the final
        # ranking still returns the requested k documents.
        pool_k = max(k, cfg.first_pass_k or k, cfg.fb_docs + cfg.neg_docs)
        ranking = self.vsm.score_vector(qvec, pool_k)
        if len(ranking) < 2:
            return ranking[:k]

        current_q = qvec
        for _ in range(max(1, cfg.iterations)):
            current_q = self._expand(current_q, ranking)
            ranking = self.vsm.score_vector(current_q, pool_k)
        return ranking[:k]


class VSMTransformer(_TRANSFORMER_BASE):
    """
    Wraps SmartVSM / RocchioExpander in the PyTerrier transformer interface so
    the hand-written model can be evaluated side by side with Terrier's.
    """

    def __init__(self, engine, analyser: Callable[[str], List[str]],
                 num_results: int = 1000, name: str = "SmartVSM"):
        try:
            super().__init__()
        except Exception:
            pass
        self.engine = engine
        self.analyser = analyser
        self.num_results = num_results
        self.name = name

    def transform(self, topics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for qid, query in zip(topics["qid"], topics["query"]):
            hits = self.engine.search(self.analyser(query), self.num_results)
            for rank, (docno, score) in enumerate(hits):
                rows.append((str(qid), str(docno), rank, float(score), query))
        return pd.DataFrame(rows, columns=["qid", "docno", "rank", "score", "query"])

    __call__ = transform


def build_vsm(
    docs: Sequence[Document], cfg: PreprocConfig, doc_scheme: str = "lnc",
    qry_scheme: str = "ltc", pivot_slope: float = 0.20,
) -> Tuple[SmartVSM, Callable[[str], List[str]], float]:
    """Construct the in-memory VSM and report its indexing time."""
    analyser = lambda text: tokenise(text, cfg.stopwords, cfg.stem)
    start = time.perf_counter()
    tokenised = [analyser(compose_document_text(d, cfg)) for d in docs]
    vsm = SmartVSM(
        [d.docno for d in docs], tokenised, doc_scheme, qry_scheme, pivot_slope
    )
    elapsed = time.perf_counter() - start
    print(
        f"[index] SmartVSM {doc_scheme}.{qry_scheme:<6} {elapsed:6.2f}s  "
        f"terms={len(vsm.df)}"
    )
    return vsm, analyser, elapsed


# =========================================================================== #
# 4. EVALUATION AND TIMING
# =========================================================================== #

METRICS = ["map", "ndcg", "ndcg_cut_10", "P_5", "P_10", "recall_100", "recip_rank"]


def timed_search(system, topics: pd.DataFrame, repeats: int = 1
                 ) -> Tuple[pd.DataFrame, float, float]:
    """
    Execute a retrieval system and return (results, best_wall_seconds,
    mean_ms_per_query).  The best of `repeats` runs is reported so that JIT
    warm-up and GC noise do not dominate the measurement.
    """
    best: Optional[float] = None
    results: Optional[pd.DataFrame] = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        out = system.transform(topics) if hasattr(system, "transform") else system(topics)
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best, results = elapsed, out
    n = max(1, len(topics))
    return results, float(best), float(best) / n * 1000.0


class Evaluator:
    """
    Effectiveness scoring.  Uses `pt.Experiment` when PyTerrier is usable and
    falls back to an equivalent built-in implementation of the trec_eval
    definitions otherwise.
    """

    def __init__(self, qrels: pd.DataFrame, mode: str = "auto"):
        self.qrels = qrels
        self.mode = mode if mode in {"auto", "pyterrier", "builtin"} else "auto"
        self._by_qid: Dict[str, Dict[str, int]] = defaultdict(dict)
        for qid, docno, label in qrels.itertuples(index=False):
            self._by_qid[str(qid)][str(docno)] = int(label)

    # ---------------------------------------------------------------- #
    def evaluate(self, runs: Dict[str, pd.DataFrame], topics: pd.DataFrame
                 ) -> pd.DataFrame:
        if self.mode in {"auto", "pyterrier"} and PT_AVAILABLE:
            try:
                names = list(runs.keys())
                frame = pt.Experiment(
                    [runs[n] for n in names], topics, self.qrels,
                    eval_metrics=METRICS, names=names,
                )
                return frame.rename(columns={"name": "system"})
            except Exception as exc:
                if self.mode == "pyterrier":
                    raise
                print(f"[eval] pt.Experiment unavailable ({exc}); using built-in")
                self.mode = "builtin"
        rows = [dict(system=name, **self._score_run(df, topics))
                for name, df in runs.items()]
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- #
    def _score_run(self, run: pd.DataFrame, topics: pd.DataFrame) -> Dict[str, float]:
        qids = [str(q) for q in topics["qid"]]
        grouped: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        if len(run):
            for qid, docno, score in zip(run["qid"], run["docno"], run["score"]):
                grouped[str(qid)].append((str(docno), float(score)))

        totals = Counter()
        for qid in qids:
            judged = self._by_qid.get(qid, {})
            ranked = sorted(grouped.get(qid, []), key=lambda kv: -kv[1])
            labels = [judged.get(docno, 0) for docno, _ in ranked]
            n_rel = sum(1 for v in judged.values() if v > 0)

            hits = 0
            ap = 0.0
            for i, label in enumerate(labels, start=1):
                if label > 0:
                    hits += 1
                    ap += hits / i
            totals["map"] += ap / n_rel if n_rel else 0.0

            rr = 0.0
            for i, label in enumerate(labels, start=1):
                if label > 0:
                    rr = 1.0 / i
                    break
            totals["recip_rank"] += rr

            totals["P_5"] += sum(1 for v in labels[:5] if v > 0) / 5.0
            totals["P_10"] += sum(1 for v in labels[:10] if v > 0) / 10.0
            totals["recall_100"] += (
                sum(1 for v in labels[:100] if v > 0) / n_rel if n_rel else 0.0
            )

            ideal = sorted((v for v in judged.values() if v > 0), reverse=True)
            totals["ndcg"] += self._ndcg(labels, ideal, None)
            totals["ndcg_cut_10"] += self._ndcg(labels, ideal, 10)

        n = max(1, len(qids))
        return {m: totals[m] / n for m in METRICS}

    @staticmethod
    def _ndcg(labels: Sequence[int], ideal: Sequence[int], cut: Optional[int]) -> float:
        if cut is not None:
            labels, ideal = labels[:cut], ideal[:cut]
        dcg = sum(l / math.log2(i + 1) for i, l in enumerate(labels, start=1) if l > 0)
        idcg = sum(l / math.log2(i + 1) for i, l in enumerate(ideal, start=1) if l > 0)
        return dcg / idcg if idcg else 0.0


def split_topics(topics: pd.DataFrame, held_out_every: int = 3
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministic train/validation split (every third topic is held out).

    All tuning decisions are taken on the training half only; the held-out half
    provides an honest estimate of how the configuration generalises to the
    unseen evaluation queries.
    """
    def bucket(qid: str) -> bool:
        try:
            return int(qid) % held_out_every == 0
        except ValueError:
            return abs(hash(qid)) % held_out_every == 0

    mask = topics["qid"].map(bucket)
    return topics[~mask].reset_index(drop=True), topics[mask].reset_index(drop=True)


# =========================================================================== #
# 5. OUTPUT
# =========================================================================== #

def write_trec_run(
    results: pd.DataFrame, path: str, run_tag: str = "VSM",
    top_k: int = 1000, qid_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Export a ranking in the six-column TREC format expected by trec_eval:

        qid  Q0  docno  rank  score  run_tag

    `qid_map` optionally rewrites the ordinal identifiers back to the .I values
    of the source query file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    frame = results.copy()
    frame["qid"] = frame["qid"].astype(str)
    frame["docno"] = frame["docno"].astype(str)
    frame = frame.sort_values(["qid", "score"], ascending=[True, False])

    lines = 0
    with open(path, "w", encoding="utf-8") as fh:
        for qid, group in frame.groupby("qid", sort=True):
            out_qid = qid_map.get(qid, qid) if qid_map else qid
            for rank, (docno, score) in enumerate(
                zip(group["docno"].head(top_k), group["score"].head(top_k)), start=1
            ):
                fh.write(f"{out_qid} Q0 {docno} {rank} {score:.6f} {run_tag}\n")
                lines += 1
    print(f"[output] wrote {lines} result lines -> {path}")
    return path


def save_table(frame: pd.DataFrame, path: str, title: str = "") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    frame.to_csv(path, index=False)
    if title:
        print(f"\n=== {title} ===")
        with pd.option_context("display.width", 200, "display.max_columns", 50):
            print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"[output] table -> {path}")


# =========================================================================== #
# 6. EXPERIMENT DRIVER
# =========================================================================== #

@dataclass
class ExperimentPaths:
    workdir: str

    @staticmethod
    def _p(*parts: str) -> str:
        """Join and normalise to forward slashes (keeps Java bridge happy)."""
        return os.path.join(*parts).replace("\\", "/")

    @property
    def indexes(self) -> str:
        return self._p(self.workdir, "indexes")

    @property
    def runs(self) -> str:
        return self._p(self.workdir, "runs")

    @property
    def results(self) -> str:
        return self._p(self.workdir, "results")

    def prepare(self) -> None:
        for d in (self.indexes, self.runs, self.results):
            os.makedirs(d, exist_ok=True)


def _row(system: str, family: str, detail: str, metrics: Dict[str, float],
         index_s: float, search_s: float, ms_per_q: float) -> Dict[str, object]:
    row = {"system": system, "family": family, "configuration": detail}
    row.update({m: round(float(metrics.get(m, 0.0)), 4) for m in METRICS})
    row["index_time_s"] = round(index_s, 3)
    row["search_time_s"] = round(search_s, 3)
    row["ms_per_query"] = round(ms_per_q, 2)
    return row


def _metrics_of(evaluator: Evaluator, run: pd.DataFrame, topics: pd.DataFrame,
                name: str) -> Dict[str, float]:
    frame = evaluator.evaluate({name: run}, topics)
    return frame.iloc[0].to_dict()


def stage_preprocessing(
    docs, builder, train, evaluator, quick: bool
) -> Tuple[PreprocConfig, pd.DataFrame]:
    """Stage 1 -- which preprocessing chain feeds the best TF_IDF ranking?"""
    print("\n########## Stage 1: preprocessing / indexing configurations ##########")
    candidates = [
        PreprocConfig(stem=False, stopwords=False, title_weight=1),
        PreprocConfig(stem=False, stopwords=True, title_weight=1),
        PreprocConfig(stem=True, stopwords=False, title_weight=1),
        PreprocConfig(stem=True, stopwords=True, title_weight=1),
    ]
    if not quick:
        candidates += [
            PreprocConfig(stem=True, stopwords=True, title_weight=0),
            PreprocConfig(stem=True, stopwords=True, title_weight=2),
            PreprocConfig(stem=True, stopwords=True, title_weight=3),
            PreprocConfig(stem=True, stopwords=True, title_weight=2,
                          include_author=True, include_bib=True),
            PreprocConfig(stem=True, stopwords=True, title_weight=2,
                          dedupe_title=False),
            # Prior experiments showed dedupe_title=False + boosted title tends
            # to win because .W already repeats the title once, so title
            # weight 2 (or 3) with dedupe off effectively boosts titles further.
            PreprocConfig(stem=True, stopwords=True, title_weight=3,
                          dedupe_title=False),
        ]

    rows = []
    best_cfg, best_score = candidates[-1], -1.0
    for cfg in candidates:
        index, index_s = builder.build(docs, cfg)
        system = terrier_retriever(index, "TF_IDF")
        run, search_s, ms = timed_search(system, train)
        metrics = _metrics_of(evaluator, run, train, "TF_IDF")
        rows.append(_row(f"TF_IDF [{cfg.key()}]", "preprocessing", cfg.label(),
                         metrics, index_s, search_s, ms))
        if metrics["map"] > best_score:
            best_cfg, best_score = cfg, metrics["map"]

    frame = pd.DataFrame(rows).sort_values("map", ascending=False)
    print(f"\n[stage 1] best preprocessing: {best_cfg.label()}  (MAP={best_score:.4f})")
    return best_cfg, frame


def stage_models(docs, builder, cfg, train, evaluator, quick: bool
                 ) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Stage 2 -- compare sparse weighting models on the winning index."""
    print("\n########## Stage 2: sparse vector space models ##########")
    index, index_s = builder.build(docs, cfg)
    rows, systems = [], {}

    for wmodel in ("Tf", "TF_IDF", "LemurTF_IDF", "CoordinateMatch"):
        try:
            system = terrier_retriever(index, wmodel)
            run, search_s, ms = timed_search(system, train)
            metrics = _metrics_of(evaluator, run, train, wmodel)
        except Exception as exc:
            print(f"[stage 2] {wmodel} unavailable: {exc}")
            continue
        rows.append(_row(f"Terrier {wmodel}", "terrier", cfg.label(),
                         metrics, index_s, search_s, ms))
        systems[f"Terrier {wmodel}"] = system

    schemes = [("lnc", "ltc"), ("ltc", "ltc"), ("anc", "atc"), ("lnu", "ltu")]
    if not quick:
        schemes += [("ltc", "ltn"), ("anc", "ltc"), ("Lnc", "ltc"), ("bnc", "ltc")]
    for doc_scheme, qry_scheme in schemes:
        vsm, analyser, vsm_index_s = build_vsm(docs, cfg, doc_scheme, qry_scheme)
        name = f"SmartVSM {doc_scheme}.{qry_scheme}"
        system = VSMTransformer(vsm, analyser, name=name)
        run, search_s, ms = timed_search(system, train)
        metrics = _metrics_of(evaluator, run, train, name)
        rows.append(_row(name, "smart-vsm", f"{cfg.label()}; {doc_scheme}.{qry_scheme}",
                         metrics, vsm_index_s, search_s, ms))
        systems[name] = system

    frame = pd.DataFrame(rows).sort_values("map", ascending=False)
    return frame, systems


def stage_parameters(docs, builder, cfg, train, evaluator, quick: bool) -> pd.DataFrame:
    """Stage 3 -- parameter sweeps: Terrier's `c`, and the pivot slope."""
    print("\n########## Stage 3: parameter tuning ##########")
    index, index_s = builder.build(docs, cfg)
    rows = []

    c_values = [0.2, 0.5, 1.0] if quick else [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    for c in c_values:
        system = terrier_retriever(index, "TF_IDF", controls={"c": c})
        run, search_s, ms = timed_search(system, train)
        metrics = _metrics_of(evaluator, run, train, f"TF_IDF c={c}")
        rows.append(_row(f"Terrier TF_IDF (c={c})", "terrier-tuning", f"c={c}",
                         metrics, index_s, search_s, ms))

    slopes = [0.1, 0.2, 0.3] if quick else [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    for slope in slopes:
        vsm, analyser, vsm_index_s = build_vsm(docs, cfg, "lnu", "ltu", slope)
        name = f"SmartVSM lnu.ltu (s={slope})"
        run, search_s, ms = timed_search(VSMTransformer(vsm, analyser, name=name), train)
        metrics = _metrics_of(evaluator, run, train, name)
        rows.append(_row(name, "vsm-tuning", f"pivot slope={slope}",
                         metrics, vsm_index_s, search_s, ms))

    return pd.DataFrame(rows).sort_values("map", ascending=False)


def stage_feedback(docs, cfg, base_scheme, train, evaluator, quick: bool
                   ) -> Tuple[pd.DataFrame, Optional[RocchioConfig]]:
    """Stage 4 -- Rocchio pseudo-relevance feedback on the best VSM."""
    print("\n########## Stage 4: Rocchio pseudo-relevance feedback ##########")
    doc_scheme, qry_scheme, slope = base_scheme
    vsm, analyser, index_s = build_vsm(docs, cfg, doc_scheme, qry_scheme, slope)

    grid = [
        RocchioConfig(1.0, 0.5, 0.0, 5, 15, 0),
        RocchioConfig(1.0, 0.75, 0.15, 10, 20, 10),
    ]
    if not quick:
        grid += [
            RocchioConfig(1.0, 0.3, 0.0, 5, 10, 0),
            RocchioConfig(1.0, 0.5, 0.1, 10, 25, 10),
            RocchioConfig(1.0, 0.75, 0.0, 8, 30, 0),
            RocchioConfig(1.0, 1.0, 0.15, 10, 30, 10),
            RocchioConfig(1.2, 0.6, 0.2, 15, 40, 15),
            RocchioConfig(1.0, 0.4, 0.05, 20, 50, 20),
        ]

    rows, best_cfg, best_score = [], None, -1.0
    for rc in grid:
        name = rc.label()
        system = VSMTransformer(RocchioExpander(vsm, rc), analyser, name=name)
        run, search_s, ms = timed_search(system, train)
        metrics = _metrics_of(evaluator, run, train, name)
        rows.append(_row(f"SmartVSM + {name}", "rocchio", name,
                         metrics, index_s, search_s, ms))
        if metrics["map"] > best_score:
            best_cfg, best_score = rc, metrics["map"]

    return pd.DataFrame(rows).sort_values("map", ascending=False), best_cfg


def stage_best_combination(
    docs, cfg, candidate_schemes, train, evaluator, quick: bool
) -> Tuple[pd.DataFrame, Tuple[str, str, float], Optional[RocchioConfig]]:
    """
    Stage 5 -- exhaustive Rocchio grid on the top-N base VSMs.

    The earlier stages tune scheme and pivot slope separately, so the winning
    base VSM from Stage 3 (e.g. `lnu.ltu` with s=0.3) never gets paired with
    Rocchio in Stage 4 -- Stage 4 is anchored to whichever scheme happened to
    lead Stage 2 (e.g. `Lnc.ltc`).  This stage closes that gap by running a
    wide Rocchio + multi-pass grid on the top candidate schemes and returning
    the single best (scheme, Rocchio) combination for the shipped submission.
    """
    print("\n########## Stage 5: best combined (base VSM + wide Rocchio grid) ##########")

    # Wider Rocchio grid.  Stage 4's small grid consistently preferred gamma=0
    # and higher fb_terms, so this grid concentrates there and adds multi-pass
    # variants known to help on Cranfield-scale collections.
    grid = [
        RocchioConfig(alpha=1.0, beta=0.6,  gamma=0.0, fb_docs=5,  fb_terms=25),
        RocchioConfig(alpha=1.0, beta=0.75, gamma=0.0, fb_docs=8,  fb_terms=30),
        RocchioConfig(alpha=1.0, beta=0.75, gamma=0.0, fb_docs=10, fb_terms=50),
        RocchioConfig(alpha=1.0, beta=0.9,  gamma=0.0, fb_docs=8,  fb_terms=40),
    ]
    if not quick:
        grid += [
            RocchioConfig(alpha=1.0, beta=0.5,  gamma=0.0, fb_docs=5,  fb_terms=20),
            RocchioConfig(alpha=1.0, beta=0.75, gamma=0.0, fb_docs=12, fb_terms=75),
            RocchioConfig(alpha=1.0, beta=1.0,  gamma=0.0, fb_docs=8,  fb_terms=50),
            RocchioConfig(alpha=1.0, beta=1.0,  gamma=0.0, fb_docs=10, fb_terms=75),
            RocchioConfig(alpha=1.0, beta=1.25, gamma=0.0, fb_docs=8,  fb_terms=40),
            # Multi-pass variants -- Rocchio iterated on the expanded ranking.
            RocchioConfig(alpha=1.0, beta=0.75, gamma=0.0, fb_docs=8,  fb_terms=30,
                          iterations=2),
            RocchioConfig(alpha=1.0, beta=0.75, gamma=0.0, fb_docs=10, fb_terms=50,
                          iterations=2),
            # Widened candidate pool -- draw feedback from a deeper first pass.
            RocchioConfig(alpha=1.0, beta=0.75, gamma=0.0, fb_docs=8,  fb_terms=30,
                          first_pass_k=2000),
        ]

    rows: List[Dict[str, object]] = []
    best_key: Optional[Tuple[str, str, float]] = None
    best_rc: Optional[RocchioConfig] = None
    best_score = -1.0

    for scheme in candidate_schemes:
        doc_scheme, qry_scheme, slope = scheme
        vsm, analyser, index_s = build_vsm(docs, cfg, doc_scheme, qry_scheme, slope)

        # Baseline (no feedback) for this scheme so the table shows the lift.
        baseline_name = f"SmartVSM {doc_scheme}.{qry_scheme} (s={slope})"
        run, search_s, ms = timed_search(
            VSMTransformer(vsm, analyser, name=baseline_name), train,
        )
        metrics = _metrics_of(evaluator, run, train, baseline_name)
        rows.append(_row(baseline_name, "stage5-base",
                         f"{doc_scheme}.{qry_scheme}; s={slope}",
                         metrics, index_s, search_s, ms))

        for rc in grid:
            name = f"{doc_scheme}.{qry_scheme}(s={slope}) + {rc.label()}"
            system = VSMTransformer(RocchioExpander(vsm, rc), analyser, name=name)
            run, search_s, ms = timed_search(system, train)
            metrics = _metrics_of(evaluator, run, train, name)
            rows.append(_row(f"SmartVSM {name}", "stage5-combined", name,
                             metrics, index_s, search_s, ms))
            if metrics["map"] > best_score:
                best_key, best_rc, best_score = scheme, rc, metrics["map"]

    if best_key is not None and best_rc is not None:
        print(f"\n[stage 5] best combination: "
              f"{best_key[0]}.{best_key[1]} (s={best_key[2]}) + {best_rc.label()}  "
              f"(MAP={best_score:.4f})")
    return pd.DataFrame(rows).sort_values("map", ascending=False), best_key, best_rc


def run_experiments(args: argparse.Namespace) -> None:
    paths = ExperimentPaths(args.workdir)
    paths.prepare()

    # ---------------- ingestion ----------------------------------------- #
    docs, topics, qrels = load_collection(
        args.data, args.workdir, qid_mode="ordinal",
        minus_one_relevant=args.minus_one_relevant,
    )
    topic_frame = topics_to_frame(topics)
    qid_map = {t.qid: t.file_id for t in topics}
    pd.DataFrame(
        {"qid": list(qid_map.keys()), "file_id": list(qid_map.values())}
    ).to_csv(os.path.join(paths.results, "query_id_map.tsv"), sep="\t", index=False)

    # Keep only topics that actually carry judgements when evaluating.
    judged = set(qrels["qid"].unique())
    eval_topics = topic_frame[topic_frame["qid"].isin(judged)].reset_index(drop=True)
    print(f"[setup] {len(eval_topics)}/{len(topic_frame)} topics have judgements")
    print(f"[setup] stemmer: {STEMMER_NAME}")

    train, held_out = split_topics(eval_topics, args.held_out_every)
    print(f"[setup] tuning on {len(train)} topics, holding out {len(held_out)}")

    evaluator = Evaluator(qrels, mode=args.evaluator)
    init_pyterrier()
    builder = TerrierIndexBuilder(paths.indexes)

    # ---------------- staged tuning -------------------------------------- #
    if args.skip_tuning:
        best_cfg = PreprocConfig(stem=True, stopwords=True, title_weight=2)
        best_scheme = ("lnu", "ltu", 0.30)
        best_rocchio = RocchioConfig(1.0, 0.75, 0.0, 8, 30)
        # A pre-tuned combination for the direct-submission path.
        best_combo_scheme = best_scheme
        best_combo_rocchio = RocchioConfig(1.0, 0.75, 0.0, 8, 30, iterations=2)
        all_tables: Dict[str, pd.DataFrame] = {}
    else:
        best_cfg, t1 = stage_preprocessing(docs, builder, train, evaluator, args.quick)
        t2, _systems = stage_models(docs, builder, best_cfg, train, evaluator, args.quick)
        t3 = stage_parameters(docs, builder, best_cfg, train, evaluator, args.quick)

        # Read the winning SMART scheme (and slope) off the tuning tables.
        best_scheme = ("lnu", "ltu", 0.20)
        vsm_rows = t2[t2["family"] == "smart-vsm"]
        if len(vsm_rows):
            top = vsm_rows.iloc[0]["system"].split()[-1]
            if "." in top:
                d, q = top.split(".")
                best_scheme = (d, q, best_scheme[2])
        slope_rows = t3[t3["family"] == "vsm-tuning"]
        if len(slope_rows) and best_scheme[0] == "lnu":
            best_scheme = (
                best_scheme[0], best_scheme[1],
                float(slope_rows.iloc[0]["configuration"].split("=")[-1]),
            )

        t4, best_rocchio = stage_feedback(
            docs, best_cfg, best_scheme, train, evaluator, args.quick
        )

        # Stage 5 -- gather top base VSMs from Stages 2 and 3 and pair each
        # with a wide Rocchio grid.  This closes the gap where Stage 4 only
        # runs feedback on top of the Stage-2 winner.
        candidate_schemes: List[Tuple[str, str, float]] = []
        seen: set = set()
        # (i) Best SMART scheme from Stage 2 with the default slope.
        for _, row in vsm_rows.head(2).iterrows():
            top = row["system"].split()[-1]
            if "." in top:
                d, q = top.split(".")
                key = (d, q, 0.20)
                if key not in seen:
                    candidate_schemes.append(key)
                    seen.add(key)
        # (ii) Best pivoted-length scheme with the best-tuned slope from Stage 3.
        if len(slope_rows):
            top_slope = float(slope_rows.iloc[0]["configuration"].split("=")[-1])
            key = ("lnu", "ltu", top_slope)
            if key not in seen:
                candidate_schemes.append(key)
                seen.add(key)
        # (iii) Failsafe -- always try the tuned pivoted-length as well.
        for fallback in [("lnu", "ltu", 0.30), ("Lnc", "ltc", 0.20)]:
            if fallback not in seen:
                candidate_schemes.append(fallback)
                seen.add(fallback)

        t5, best_combo_scheme, best_combo_rocchio = stage_best_combination(
            docs, best_cfg, candidate_schemes[:3], train, evaluator, args.quick,
        )

        all_tables = {
            "stage1_preprocessing": t1,
            "stage2_models": t2,
            "stage3_parameters": t3,
            "stage4_rocchio": t4,
            "stage5_best_combination": t5,
        }
        for name, table in all_tables.items():
            save_table(table, os.path.join(paths.results, f"{name}.csv"),
                       title=name.replace("_", " "))

    # ---------------- final systems -------------------------------------- #
    print("\n########## Final comparison (held-out + full topic set) ##########")
    index, index_s = builder.build(docs, best_cfg)
    doc_scheme, qry_scheme, slope = best_scheme
    vsm, analyser, vsm_index_s = build_vsm(docs, best_cfg, doc_scheme, qry_scheme, slope)

    finalists = {
        "Terrier TF_IDF": (terrier_retriever(index, "TF_IDF"), index_s),
        f"SmartVSM {doc_scheme}.{qry_scheme}": (
            VSMTransformer(vsm, analyser, num_results=args.top_k), vsm_index_s
        ),
    }
    if best_rocchio is not None:
        finalists[f"SmartVSM + {best_rocchio.label()}"] = (
            VSMTransformer(RocchioExpander(vsm, best_rocchio), analyser,
                           num_results=args.top_k, name="rocchio"),
            vsm_index_s,
        )

    # Stage 5's winning combination: the tuned base VSM paired with its best
    # Rocchio configuration.  This may reuse the same VSM instance or build a
    # different one depending on which scheme won Stage 5.
    if best_combo_scheme is not None and best_combo_rocchio is not None:
        cd, cq, cs = best_combo_scheme
        if best_combo_scheme == best_scheme:
            combo_vsm, combo_analyser, combo_idx_s = vsm, analyser, vsm_index_s
        else:
            combo_vsm, combo_analyser, combo_idx_s = build_vsm(
                docs, best_cfg, cd, cq, cs
            )
        combo_name = f"SmartVSM {cd}.{cq}(s={cs}) + {best_combo_rocchio.label()}"
        finalists[combo_name] = (
            VSMTransformer(
                RocchioExpander(combo_vsm, best_combo_rocchio), combo_analyser,
                num_results=args.top_k, name="combo",
            ),
            combo_idx_s,
        )

    final_rows, full_runs = [], {}
    for name, (system, idx_s) in finalists.items():
        run_h, s_h, ms_h = timed_search(system, held_out, repeats=args.repeats)
        m_h = _metrics_of(evaluator, run_h, held_out, name)
        final_rows.append(_row(name, "final/held-out", f"{best_cfg.label()}",
                               m_h, idx_s, s_h, ms_h))

        run_f, s_f, ms_f = timed_search(system, eval_topics, repeats=args.repeats)
        m_f = _metrics_of(evaluator, run_f, eval_topics, name)
        final_rows.append(_row(name, "final/all-topics", f"{best_cfg.label()}",
                               m_f, idx_s, s_f, ms_f))
        full_runs[name] = run_f

        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        write_trec_run(run_f, os.path.join(paths.runs, f"{slug}.txt"),
                       run_tag=slug, top_k=args.top_k)

    final_table = pd.DataFrame(final_rows)
    save_table(final_table, os.path.join(paths.results, "final_comparison.csv"),
               title="final comparison")

    # The submission run: the best system over every supplied topic.
    best_name = max(
        full_runs,
        key=lambda n: float(
            final_table[(final_table["system"] == n)
                        & (final_table["family"] == "final/held-out")]["map"].iloc[0]
        ),
    )
    print(f"\n[final] selected system: {best_name}")
    write_trec_run(full_runs[best_name], os.path.join(paths.runs, "results.txt"),
                   run_tag="CRAN_VSM", top_k=args.top_k)

    best_config = {
        "selected_system": best_name,
        "preprocessing": asdict(best_cfg),
        "smart_scheme": {"doc": doc_scheme, "query": qry_scheme, "pivot_slope": slope},
        "rocchio": asdict(best_rocchio) if best_rocchio else None,
        "combined_winner": (
            {
                "scheme": {
                    "doc": best_combo_scheme[0],
                    "query": best_combo_scheme[1],
                    "pivot_slope": best_combo_scheme[2],
                },
                "rocchio": asdict(best_combo_rocchio),
            }
            if best_combo_scheme is not None and best_combo_rocchio is not None
            else None
        ),
        "stemmer": STEMMER_NAME,
        "top_k": args.top_k,
        "index_time_terrier_s": round(index_s, 3),
        "index_time_vsm_s": round(vsm_index_s, 3),
    }
    with open(os.path.join(paths.results, "best_config.json"), "w") as fh:
        json.dump(best_config, fh, indent=2)
    print(f"[output] configuration -> {paths.results}/best_config.json")

    # ---------------- unseen queries ------------------------------------- #
    if args.test_queries:
        unseen = parse_topics(args.test_queries, qid_mode=args.qid_mode)
        unseen_frame = topics_to_frame(unseen)
        system = finalists[best_name][0]
        run, search_s, ms = timed_search(system, unseen_frame, repeats=args.repeats)
        write_trec_run(run, os.path.join(paths.runs, "results_test_queries.txt"),
                       run_tag="CRAN_VSM", top_k=args.top_k)
        # Safety copy keyed on the original .I values of the query file.
        write_trec_run(
            run, os.path.join(paths.runs, "results_test_queries_fileids.txt"),
            run_tag="CRAN_VSM", top_k=args.top_k,
            qid_map={t.qid: t.file_id for t in unseen},
        )
        print(f"[final] scored {len(unseen_frame)} unseen queries in "
              f"{search_s:.3f}s ({ms:.2f} ms/query)")


# =========================================================================== #
# 7. CLI
# =========================================================================== #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cranfield ranked retrieval with sparse vector space models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="cran.tar.gz",
                   help="cran.tar.gz archive or a directory holding the files")
    p.add_argument("--workdir", default="./work",
                   help="where indexes, runs and result tables are written")
    p.add_argument("--test-queries", default=None,
                   help="extra (unseen) query file to score with the best system")
    p.add_argument("--qid-mode", default="auto", choices=["auto", "file", "ordinal"],
                   help="qid numbering used for --test-queries")
    p.add_argument("--top-k", type=int, default=1000,
                   help="number of documents retrieved per query")
    p.add_argument("--repeats", type=int, default=3,
                   help="timing repetitions for the final systems")
    p.add_argument("--held-out-every", type=int, default=3,
                   help="hold out every n-th topic as a validation set")
    p.add_argument("--evaluator", default="auto",
                   choices=["auto", "pyterrier", "builtin"])
    p.add_argument("--quick", action="store_true",
                   help="small grids for a fast smoke test")
    p.add_argument("--skip-tuning", action="store_true",
                   help="use the pre-tuned configuration and only produce runs")
    p.add_argument("--minus-one-relevant", action="store_true",
                   help="treat cranqrel grade -1 as (weakly) relevant")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # ------------------------------------------------------------------ #
    # Windows / Java bridge fix: the Terrier JVM requires clean absolute  #
    # paths with forward slashes.  Relative paths (./work) and mixed-     #
    # slash strings (.\var\./work) both cause the JVM to fail silently    #
    # and write nothing.  Resolve everything to an absolute POSIX-style   #
    # string here, once, before any other code touches the paths.         #
    # ------------------------------------------------------------------ #
    args.workdir = os.path.abspath(args.workdir).replace("\\", "/")
    args.data    = os.path.abspath(args.data).replace("\\", "/")
    if args.test_queries:
        args.test_queries = os.path.abspath(args.test_queries).replace("\\", "/")
    print(f"[paths] workdir : {args.workdir}")
    print(f"[paths] data    : {args.data}")

    started = time.perf_counter()
    try:
        run_experiments(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(f"\n[done] total wall time {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
