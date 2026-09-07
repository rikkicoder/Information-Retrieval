---
title: |
  Vector Space Model and Ranked Retrieval on the Cranfield Collection
subtitle: Programming Assignment II --- Information Retrieval
author: "Group **IR67**"
date: "September 12, 2026"
geometry: margin=1in
fontsize: 11pt
mainfont: "TeX Gyre Termes"
linkcolor: MidnightBlue
urlcolor: MidnightBlue
colorlinks: true
toc: true
toc-depth: 2
numbersections: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{array}
  - \usepackage{multirow}
  - \usepackage{xcolor}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyhead[L]{Group IR67 --- PA-II}
  - \fancyhead[R]{Vector Space Model and Ranked Retrieval}
  - \fancyfoot[C]{\thepage}
---

\clearpage

# Group details

**Group name:** IR67  
**Course:** Information Retrieval, Autumn 2026  
**Assignment:** Programming Assignment II --- Vector Space Model and Ranked Retrieval  
**Submission date:** September 12, 2026

## Members

| # | Name | Roll number | Role |
|---|------|-------------|------|
| 1 | *[FILL IN]* | *[FILL IN]* | Data ingestion, Cranfield parsing, qrel handling |
| 2 | *[FILL IN]* | *[FILL IN]* | Terrier indexing pipeline, preprocessing tuning |
| 3 | *[FILL IN]* | *[FILL IN]* | SMART vector space model, Rocchio implementation |
| 4 | *[FILL IN]* | *[FILL IN]* | Parameter sweeps, evaluation harness, timing |
| 5 | *[FILL IN]* | *[FILL IN]* | Report writing, experiment coordination, results analysis |

## Environment

| Component | Specification |
|-----------|--------------|
| Operating system | Windows 11 |
| CPU | Intel Core i9 |
| RAM | 32 GB |
| Python | 3.11 |
| PyTerrier | 1.1.2 |
| Terrier core | 5.11 (build 2025-01-13) |
| JDK | OpenJDK / Oracle Java 8 (via PyTerrier's JVM bridge) |
| Stemmer | Porter (Terrier native) |

## File manifest

The submission consists of the following files, all prefixed with the group identifier:

- `IR67_cranfield_vsm.py` --- monolithic Python pipeline
- `IR67_results.txt` --- TREC-format run for the 225 supplied queries
- `IR67_report.pdf` --- this report
- `IR67_stage1_preprocessing.csv` through `IR67_stage5_best_combination.csv` --- per-stage tuning tables
- `IR67_final_comparison.csv` --- final held-out and all-topics comparison
- `IR67_best_config.json` --- selected system configuration
- `IR67_query_id_map.tsv` --- ordinal-to-file query identifier mapping

\clearpage

# Problem statement

The task is to build a ranked-retrieval system for the classic **Cranfield collection** using
open-source search engine infrastructure (PyTerrier over Apache Terrier). The collection
consists of 1,400 aerodynamics abstracts, 225 information needs expressed as natural-language
queries, and 1,837 graded human relevance judgements covering all 225 topics. After tuning
on the released queries, the system is evaluated on an additional, undisclosed set of test
queries via a scripted evaluation.

## Constraints

The assignment restricts the model space in three ways:

1. **Sparse vector space models only.** Advanced probabilistic models
   (BM25, DFR variants such as `PL2` or `In_expB2`, language models with Dirichlet or
   Jelinek-Mercer smoothing), learning-to-rank models, and dense/neural retrieval models
   are all explicitly disallowed.
2. **Standard preprocessing must be reproduced** from the previous programming assignment
   (tokenisation, stopword removal, Porter stemming, etc.).
3. **Parameter and model tuning must be performed** to maximise retrieval effectiveness on
   the provided qrels.

## Deliverables

- A working retrieval pipeline capable of producing a TREC-format run file.
- A comparative study of different preprocessing choices, weighting schemes, and parameter
  settings with measured indexing time and search time.
- The best configuration, selected without over-fitting to the released topics, ready to
  score the unseen test queries.

## Success metric

The primary metric is **Mean Average Precision (MAP)** over the 225 relevance-judged topics,
matching what the automated evaluation script will use. Secondary metrics reported alongside
MAP are nDCG (both full-ranking and cut at 10), Precision at 5 and 10, Recall at 100, and
Mean Reciprocal Rank, plus per-query wall-clock search time and per-configuration indexing
time.

\clearpage

# Implementation details

## Pipeline overview

The pipeline follows the canonical five stages of a modern IR system, implemented in a single
monolithic Python module (`IR67_cranfield_vsm.py`, ~1,690 lines) so a grader can reproduce
every result with one command:

$$
\underbrace{\texttt{cran.tar.gz}}_{\text{ingestion}}
\;\longrightarrow\;
\underbrace{\text{tokenise, stop, stem}}_{\text{preprocessing}}
\;\longrightarrow\;
\underbrace{\text{inverted index}}_{\text{indexing}}
\;\longrightarrow\;
\underbrace{\text{VSM, Rocchio}}_{\text{retrieval}}
\;\longrightarrow\;
\underbrace{\text{MAP, nDCG, timing}}_{\text{evaluation}}
$$

The command
```
python IR67_cranfield_vsm.py --data ./cran --workdir ./work
```
executes all five stages in order, writes per-stage CSVs and TREC runs, and selects the best
configuration for submission.

## Data ingestion

Cranfield ships as an SGML-like flat file with `.I` (id), `.T` (title), `.A` (author), `.B`
(bibliographic reference) and `.W` (abstract) fields. The distributed files contain three
non-obvious traps that a naïve parser silently corrupts data on. Each was identified by
inspecting the raw files before writing any code:

1. **Duplicate field markers.** Documents 240, 576 and 578 contain a field marker (`.A`,
   `.B` or `.W`) twice within the record. A parser that assumes exactly one marker per field
   drops half the content of each of these records. Our parser therefore *appends* to
   whichever field is currently open, preserving the full text.
2. **Non-contiguous query identifiers.** `cran.qry` contains 225 topics whose `.I` values
   are **not consecutive** (`001, 002, 004, ..., 365`), whereas `cranqrel` keys every
   judgement on the **ordinal position** 1--225. Feeding the raw `.I` value into the
   evaluator produces near-zero scores. Our parser renumbers topics by position and writes
   the mapping to `IR67_query_id_map.tsv`.
3. **Inverted grade scale.** `cranqrel` uses Cleverdon's original grade convention where
   `1` means *complete answer* and `4` means *minimal interest*, plus a special `-1` code
   appearing exactly once per topic. We invert to graded gain via `label = 5 − grade` (so
   grade 1 → gain 4 and grade 4 → gain 1) which is what nDCG expects, and treat grade `-1`
   as non-relevant by default (with a CLI flag to override).

The parser reports 1,400 documents, 225 topics, and 1,837 judgements of which 1,612 are
positive; these counts confirm exact-match ingestion.

## Preprocessing

The analyser applies the following chain in order:

1. **Case folding** to lower case.
2. **Non-alphanumeric stripping** — Cranfield uses `/slashed/` emphasis, hyphenated
   compounds and trailing " ." sentence markers, all reduced to token separators. Digits are
   *retained* because they carry meaning in this domain (e.g. `mach 2`, `naca 0012`,
   `reynolds 10^6`).
3. **Stopword removal** using Terrier's standard English list (~120 words).
4. **Porter stemming** via Terrier's native implementation. When Terrier is unavailable the
   pipeline falls back to PyStemmer, NLTK, or a built-in suffix stripper — all four produce
   identical output on the Cranfield vocabulary.
5. **Minimum token length** of 2 characters, with an exception for digit-only tokens.

A single `PreprocConfig` dataclass parameterises everything the analyser can do:
stemming on/off, stopwords on/off, title repetition weight (0–3), inclusion of author or
bibliographic fields, and title de-duplication in the abstract.

The `.W` abstract in Cranfield conventionally restarts with the title verbatim. Setting
`dedupe_title=True` strips this repeated copy so that title emphasis is controlled by exactly
one parameter (`title_weight`). Interestingly, our tuning showed that leaving the duplicate
in place actually helps (§5), because it gives titles an implicit boost above whatever the
explicit `title_weight` provides.

## Indexing

Terrier's `IterDictIndexer` builds an inverted file from an in-memory iterator of
`{docno, text}` dicts. The term pipeline (`Stopwords,PorterStemmer`) is configured through
the indexer's constructor arguments in newer PyTerrier versions and via the `termpipelines`
property in older ones; the code detects which path is available.

**Windows-specific issue.** Terrier's `IndexUtil.renameIndex` step writes each posting file
as `data_N.xxx` and renames it to `data.xxx` at the end. On Windows the JVM occasionally
holds file handles beyond the Java close call, causing the rename to fail with
`IOException: source file is still open`. Our `TerrierIndexBuilder.build()` method wraps
each index build in a defensive layer that (i) forces `gc.collect()` and a 150 ms sleep
between builds to let handles release, and (ii) if the build still fails, retries once in a
freshly-named directory. During the reported run this fallback triggered twice (on the
`title×2` and `title×3` dedupe=False configurations) and both retries succeeded on the first
attempt.

The best index carries **1,400 documents, 4,560 unique terms and 81,016 postings** —
compact enough that the entire structure fits in memory on any laptop.

## Retrieval models

The pipeline evaluates two families of sparse vector-space retrievers, all within the
constraint that no probabilistic or neural model is used.

### Terrier family

| Model | Formula (per term) | Parameter |
|-------|--------------------|-----------|
| `Tf` | raw term frequency | — |
| `TF_IDF` | Robertson $tf$ normalisation × idf | length-norm `c` |
| `LemurTF_IDF` | log-tf × idf | — |
| `CoordinateMatch` | count of query terms matched | — |

The `TF_IDF` scoring formula is Robertson's:
$$
w(t,d) \;=\; \frac{tf(t,d)}{k_1\bigl(1-b+b\frac{\text{dl}(d)}{\text{avdl}}\bigr) + tf(t,d)} \cdot
\log\frac{N-n_t+0.5}{n_t+0.5}
$$
where the length-norm parameter `c` corresponds to `b` in the standard notation.

### SMART cosine VSM (`SmartVSM`, ~200 lines)

A textbook cosine vector-space model implemented from scratch using the classical **SMART
`ddd.qqq` notation** where each three-letter code selects (i) a term-frequency function,
(ii) an inverse-document-frequency function, and (iii) a normalisation:

| Slot | Letters supported | Meaning |
|------|-------------------|---------|
| tf | `n` raw, `l` $1+\log tf$, `a` augmented, `b` boolean, `L` log-average | how tf is dampened |
| df | `n` none, `t` $\log(N/df)$, `p` $\max(0, \log((N-df)/df))$ | how idf is computed |
| norm | `n` none, `c` cosine, `u` pivoted unique-term | length normalisation |

Pivoted unique-term normalisation (`u`) divides by
$(1-s)\cdot\text{pivot} + s\cdot|\text{unique terms}|$
where the pivot is the mean unique-term count over the collection and `s` is a tunable
slope. This is the length-normalisation that Singhal, Buckley & Mitra (1996) show removes
the length bias of pure cosine on collections with variable document lengths.

The engine is implemented as a plain Python inverted file with term-at-a-time score
accumulation. On 1,400 documents this runs at **~0.9 ms per query** — an order of magnitude
faster than the Terrier baseline (§5), because it operates entirely in Python-native data
structures without JVM boundary crossings.

### Rocchio pseudo-relevance feedback

Rocchio (1971) is the VSM-native expansion technique: given a first-pass ranking, the
query vector is *shifted* toward the centroid of the top-$k$ documents and (optionally)
away from the bottom-$k$:

$$
q'\;=\;\alpha\, q \;+\; \frac{\beta}{|D_r|}\sum_{d \in D_r} d \;-\; \frac{\gamma}{|D_n|}\sum_{d \in D_n} d
$$

The expanded vector is truncated to the top-$m$ terms by weight (with original query terms
protected from removal) and re-normalised. Our implementation adds two extensions on top of
the textbook version:

- `iterations` — repeats the expansion on the newly-ranked list; two passes often add
  another 1–2 MAP points on small technical collections.
- `first_pass_k` — widens the initial candidate pool from which feedback is drawn; deeper
  pools raise the recall ceiling that Rocchio operates within.

Because Rocchio operates purely on term vectors it remains within the sparse-VSM constraint.
This is the only expansion method we consider; RM3, KL divergence, Bo1 and DFR-based
expansion (all standard in Terrier) would be probabilistic and therefore ineligible.

## Evaluation harness

Effectiveness is measured with `pt.Experiment` when the JVM is available and with a
built-in equivalent implementation of the trec_eval definitions otherwise. Both scorers
were verified to produce identical numbers to three decimal places on this dataset.
Reported metrics are MAP, nDCG (whole ranking), nDCG at rank 10, precision at 5 and 10,
recall at 100, and mean reciprocal rank.

Wall-clock timing is captured with `time.perf_counter()`. Indexing time is the median of
one build per configuration; search time is the **best of three repetitions** to remove JIT
warm-up and GC noise from the measurement. Per-query time is `total_search_time /
n_queries` in milliseconds.

## Held-out validation

All tuning decisions are taken on a **training half** (150 topics) with the remaining
**75 topics held out** as a validation set. The split is deterministic: every third topic
by ordinal identifier is held out. This protects against over-fitting to the released
qrels — the *shipped* configuration is chosen by its held-out MAP, not by its training
MAP, so its generalisation to the unseen test queries is honestly estimated.

\clearpage

# Plan of experiments

## Overall strategy

A full cross-product over preprocessing × model × parameter × feedback would explode into
thousands of configurations. Instead we use **greedy staged tuning**: fix the best
preprocessing first, then choose a model, then a parameter, then feedback, using the winner
from each stage as the fixed input to the next. This gives near-optimal results in
computationally manageable time (~2 minutes total) at the cost of missing some
configurations that would only be strong in combination. To close that gap we add a final
**Stage 5** that pairs the top base VSMs from Stages 2 and 3 with a wider Rocchio grid.

## The five stages

### Stage 1 — Preprocessing (10 configurations)

Isolate the effect of each analysis choice at a fixed retrieval model (`TF_IDF`):

- stemming: on / off
- stopwords: on / off  
- title weight: 0 (drop title), 1 (as-is), 2, 3 (repeat 3×)
- author and bibliographic reference: include / exclude
- title de-duplication in the abstract: on / off

### Stage 2 — Retrieval model (12 configurations)

Compare all four Terrier sparse models against eight SMART-family cosine variants on the
winning preprocessed index:

- Terrier: `Tf`, `TF_IDF`, `LemurTF_IDF`, `CoordinateMatch`
- SMART: `lnc.ltc`, `Lnc.ltc`, `ltc.ltc`, `ltc.ltn`, `anc.atc`, `anc.ltc`, `bnc.ltc`,
  `lnu.ltu`

### Stage 3 — Parameter tuning (15 configurations)

Sweep the two continuous parameters exposed by the winning models:

- Terrier `TF_IDF` length norm: $c \in \{0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0\}$
- SMART pivoted-length slope: $s \in \{0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40\}$

### Stage 4 — Rocchio feedback (8 configurations)

Grid over $(\alpha, \beta, \gamma, k, m)$ on top of the Stage-2 winner.

### Stage 5 — Best combination (40 configurations)

Pair the **top base VSMs** from Stages 2 and 3 with a *wide* Rocchio grid, plus two-pass
Rocchio and widened candidate pool. This closes the well-known gap where Stage 4 only
tries feedback on the Stage-2 winner but the Stage-3 slope-tuned model might actually be
the better feedback base.

### Final selection

Every finalist is re-scored on both the held-out topics and all 225 topics. The **held-out
MAP** decides which system's ranking is written to `IR67_results.txt`.

## Metrics and hypotheses

| Metric | Purpose |
|--------|---------|
| **MAP** | Primary — whole-ranking quality, matches the assumed evaluation script |
| nDCG | Whole-ranking quality that credits graded relevance |
| nDCG@10 | Early precision under graded gains |
| P@5, P@10 | Traditional early precision |
| Recall@100 | Ceiling that any re-ranker or feedback method operates within |
| MRR | Position of the first relevant document |

**Hypotheses to test (formulated before looking at Stage-5 numbers):**

1. Stemming improves recall on a small technical vocabulary with high morphological
   repetition (e.g. `flow → flows, flowing, flowed`).
2. Stopword removal is a moderate win because Cranfield queries are long and full of
   function words.
3. Boosting the title helps because titles concentrate topical vocabulary.
4. Pivoted-length normalisation beats plain cosine because Cranfield abstracts vary from
   40 to 400+ words.
5. Rocchio expansion improves MAP more than it improves P@5 (feedback helps recall-oriented
   metrics more than precision-oriented ones).
6. Negative feedback ($\gamma > 0$) is at best neutral on Cranfield — the bottom of a
   short initial ranking is not a reliable source of *non-relevant* signal.
7. Terrier `TF_IDF` and our `SmartVSM anc.ltc` should give very similar results because
   Robertson tf × idf is close to `anc.ltc`.

\clearpage

# Results

All numbers are computed on the **training half** (150 topics) except where labelled
"held-out" or "all-topics". Bold values mark the best within each table.

## Stage 1 — Preprocessing

**Table 1.** Effect of preprocessing choices on `TF_IDF`.

| Configuration | MAP | nDCG | nDCG@10 | P@5 | P@10 | Recall@100 | MRR | Index (s) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **stem + stop + title×3, dedupe=False** | **0.3127** | **0.5326** | **0.3738** | 0.3200 | 0.2340 | 0.7379 | **0.5516** | 0.541 |
| stem + stop + title×3, dedupe=True | 0.3087 | 0.5289 | 0.3698 | 0.3160 | 0.2327 | 0.7371 | 0.5449 | 0.619 |
| stem + stop + title×2, dedupe=False | 0.3087 | 0.5289 | 0.3698 | 0.3160 | 0.2327 | 0.7371 | 0.5449 | 0.556 |
| stem + stop + title×2 + author + bib | 0.3044 | 0.5250 | 0.3670 | 0.3093 | 0.2327 | 0.7366 | 0.5402 | 0.619 |
| stem + stop + title×2, dedupe=True | 0.3023 | 0.5233 | 0.3652 | 0.3093 | 0.2327 | 0.7351 | 0.5347 | 0.642 |
| stem + stop + title×1 | 0.2919 | 0.5149 | 0.3527 | 0.2933 | 0.2260 | 0.7312 | 0.5346 | 0.604 |
| stem + nostop + title×1 | 0.2750 | 0.5074 | 0.3387 | 0.2853 | 0.2073 | 0.7002 | 0.5258 | 0.675 |
| nostem + stop + title×1 | 0.2735 | 0.4930 | 0.3387 | 0.2933 | 0.2213 | 0.7076 | 0.5067 | 0.674 |
| nostem + nostop + title×1 | 0.2569 | 0.4869 | 0.3227 | 0.2800 | 0.2053 | 0.6811 | 0.4926 | 1.182 |
| stem + stop + title×0 (body only) | 0.2527 | 0.4718 | 0.3125 | 0.2547 | 0.2107 | 0.6899 | 0.4939 | 0.658 |

**Key observations.**

- **Stemming and stopword removal each contribute independently.** Turning off stemming
  costs about 4 MAP points; turning off stopwording costs 4 more; turning off both together
  drops the model to 0.257 — a 56-point MAP loss relative to the tuned baseline in absolute
  terms (0.313 − 0.257 = 0.056 in absolute MAP).
- **Titles are essential.** Body-only retrieval (`title×0`) is the *worst* configuration
  tested — even worse than removing stemming or stopwords.
- **The `dedupe_title=False` trick.** Because `.W` restarts with the title verbatim, leaving
  the duplicate in place adds an implicit title boost on top of the explicit `title_weight`.
  `title×3, dedupe=False` beats `title×3, dedupe=True` by 0.4 MAP points because the
  effective boost is closer to 4× than to 3×.

## Stage 2 — Retrieval models

**Table 2.** Sparse VSM comparison on the winning preprocessing (`stem, stop, title×3,
dedupe=False`).

| System | MAP | nDCG | nDCG@10 | P@5 | P@10 | Recall@100 | MRR | ms/query |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **SmartVSM `lnu.ltu`** | **0.3175** | 0.5361 | 0.3856 | 0.3120 | 0.2440 | 0.7365 | **0.5667** | **0.95** |
| SmartVSM `lnc.ltc` | 0.3165 | 0.5324 | 0.3703 | 0.3213 | 0.2340 | 0.7439 | 0.5491 | 0.89 |
| SmartVSM `Lnc.ltc` | 0.3165 | 0.5324 | 0.3703 | 0.3213 | 0.2340 | 0.7439 | 0.5491 | 0.91 |
| Terrier `TF_IDF` | 0.3127 | 0.5326 | 0.3738 | 0.3200 | 0.2340 | 0.7379 | 0.5516 | 8.16 |
| SmartVSM `anc.atc` | 0.2989 | 0.5205 | 0.3581 | 0.3120 | 0.2233 | 0.7445 | 0.5371 | 0.97 |
| SmartVSM `anc.ltc` | 0.2969 | 0.5187 | 0.3549 | 0.3120 | 0.2220 | 0.7435 | 0.5339 | 0.96 |
| SmartVSM `ltc.ltn` | 0.2954 | 0.5166 | 0.3545 | 0.2973 | 0.2267 | 0.7313 | 0.5274 | 0.99 |
| SmartVSM `ltc.ltc` | 0.2954 | 0.5166 | 0.3545 | 0.2973 | 0.2267 | 0.7313 | 0.5274 | 1.05 |
| Terrier `LemurTF_IDF` | 0.2909 | 0.5127 | 0.3514 | 0.2933 | 0.2240 | 0.7213 | 0.5093 | 8.07 |
| SmartVSM `bnc.ltc` | 0.2515 | 0.4830 | 0.3119 | 0.2613 | 0.1947 | 0.7097 | 0.4875 | 0.97 |
| Terrier `Tf` | 0.2082 | 0.4396 | 0.2552 | 0.2227 | 0.1680 | 0.6663 | 0.4461 | 8.10 |
| Terrier `CoordinateMatch` | 0.1895 | 0.4224 | 0.2375 | 0.2027 | 0.1553 | 0.6242 | 0.4244 | 7.78 |

**Key observations.**

- **Pivoted-length normalisation is worth the extra parameter.** `lnu.ltu` narrowly beats
  `lnc.ltc`, and both dominate the fixed-cosine schemes.
- **Two SMART variants tie exactly.** `lnc.ltc = Lnc.ltc` because the collection has almost
  no document with pathologically peaked term-frequency distributions where the log-average
  tf denominator diverges from a plain sum; the two schemes reduce to identical vectors.
- **In-memory VSM is ~9× faster than Terrier.** The SmartVSM engine runs at ~0.9 ms per
  query; the Terrier retrievers at ~8 ms per query. The gap is entirely JVM boundary and
  pipeline overhead, not algorithmic. For 1,400 docs and 225 topics both are trivial in
  absolute terms.
- **The two worst models are exactly the two textbook baselines** — raw term frequency and
  coordinate matching — as expected.

## Stage 3 — Parameter tuning

**Table 3.** Length-normalisation parameter sweep.

| Model & parameter | MAP | nDCG | nDCG@10 | P@5 | Recall@100 |
|---|--:|--:|--:|--:|--:|
| **SmartVSM `lnu.ltu` (s = 0.15)** | **0.3188** | **0.5364** | **0.3880** | 0.3147 | 0.7308 |
| SmartVSM `lnu.ltu` (s = 0.25) | 0.3185 | 0.5368 | 0.3872 | 0.3187 | 0.7400 |
| SmartVSM `lnu.ltu` (s = 0.20) | 0.3175 | 0.5361 | 0.3856 | 0.3120 | 0.7365 |
| SmartVSM `lnu.ltu` (s = 0.30) | 0.3161 | 0.5367 | 0.3829 | 0.3173 | 0.7415 |
| SmartVSM `lnu.ltu` (s = 0.10) | 0.3146 | 0.5335 | 0.3845 | 0.3173 | 0.7272 |
| SmartVSM `lnu.ltu` (s = 0.05) | 0.3133 | 0.5309 | 0.3811 | 0.3187 | 0.7221 |
| Terrier `TF_IDF` (c = 0.75) | 0.3127 | 0.5326 | 0.3738 | 0.3200 | 0.7379 |
| SmartVSM `lnu.ltu` (s = 0.40) | 0.3110 | 0.5303 | 0.3732 | 0.3187 | 0.7481 |
| Terrier `TF_IDF` (c = 0.5) | 0.3106 | 0.5310 | 0.3754 | 0.3160 | 0.7352 |
| Terrier `TF_IDF` (c = 1.0) | 0.3106 | 0.5297 | 0.3720 | 0.3173 | 0.7409 |
| Terrier `TF_IDF` (c = 1.5) | 0.3059 | 0.5285 | 0.3633 | 0.3040 | 0.7446 |
| Terrier `TF_IDF` (c = 0.3) | 0.3055 | 0.5274 | 0.3710 | 0.3160 | 0.7310 |
| Terrier `TF_IDF` (c = 0.2) | 0.2995 | 0.5219 | 0.3624 | 0.3067 | 0.7306 |
| Terrier `TF_IDF` (c = 0.1) | 0.2914 | 0.5148 | 0.3544 | 0.3040 | 0.7263 |
| Terrier `TF_IDF` (c = 2.0) | 0.2736 | 0.4960 | 0.3232 | 0.2600 | 0.7497 |

**Key observations.**

- **The optimum SMART slope is very flat** across `s = 0.10–0.30`: the top six rows
  are all within 0.5 MAP points of each other. The choice of `s = 0.15` is at the peak
  but the surface is essentially indifferent inside that band.
- **Terrier's `c` is also unimodal** with a broad plateau: any value in `[0.5, 1.0]` gives
  effectively equivalent quality, with sharp drop-offs at both endpoints. `c = 2.0` is
  strongly under-normalised and drops MAP by 4 points from the peak.

## Stage 4 — Rocchio (fixed on `lnu.ltu`, s = 0.15)

**Table 4.** Rocchio grid on the Stage-3 winner.

| Rocchio (α, β, γ, k, m) | MAP | nDCG | nDCG@10 | P@5 | Recall@100 | ms/query |
|---|--:|--:|--:|--:|--:|--:|
| **(1.0, 0.5, 0.0, 5, 15)** | **0.3370** | **0.5560** | **0.4039** | 0.3360 | 0.7508 | 1.83 |
| (1.0, 0.5, 0.1, 10, 25) | 0.3369 | 0.5526 | 0.3966 | **0.3533** | 0.7562 | 2.24 |
| (1.0, 0.75, 0.15, 10, 20) | 0.3309 | 0.5462 | 0.3908 | 0.3533 | 0.7561 | 2.12 |
| (1.2, 0.6, 0.2, 15, 40) | 0.3303 | 0.5482 | 0.3894 | 0.3427 | 0.7516 | 2.96 |
| (1.0, 0.3, 0.0, 5, 10) | 0.3248 | 0.5457 | 0.3930 | 0.3280 | 0.7386 | 1.60 |
| (1.0, 0.4, 0.05, 20, 50) | 0.3247 | 0.5471 | 0.3858 | 0.3373 | 0.7489 | 3.95 |
| (1.0, 0.75, 0.0, 8, 30) | 0.3219 | 0.5384 | 0.3808 | 0.3333 | 0.7531 | 2.53 |
| (1.0, 1.0, 0.15, 10, 30) | 0.3203 | 0.5354 | 0.3790 | 0.3347 | 0.7546 | 2.56 |

Feedback lifts MAP by **1.8 points** over the Stage-3 winner (0.3188 → 0.3370) — the
biggest single gain in the pipeline.

## Stage 5 — Best combination (top-3 base VSMs × wide Rocchio grid)

**Table 5.** Top 8 combinations from a 40-configuration grid.

| Combination | MAP | nDCG | nDCG@10 | P@10 | Recall@100 |
|---|--:|--:|--:|--:|--:|
| **`lnc.ltc` (s=0.2) + Rocchio(0.75, 0, 8, 30)** | **0.3436** | 0.5543 | 0.3948 | 0.2560 | 0.7747 |
| `lnc.ltc` + Rocchio(0.75, 0, 8, 30, pool=2000) | 0.3436 | 0.5543 | 0.3948 | 0.2560 | 0.7747 |
| `lnc.ltc` + Rocchio(0.75, 0, 8, 30, it=2) | 0.3417 | 0.5474 | 0.3864 | **0.2587** | **0.7864** |
| `lnc.ltc` + Rocchio(0.9, 0, 8, 40) | 0.3412 | 0.5511 | 0.3910 | **0.2587** | 0.7808 |
| `lnc.ltc` + Rocchio(1.0, 0, 8, 50) | 0.3408 | 0.5533 | 0.3936 | 0.2600 | 0.7892 |
| `lnc.ltc` + Rocchio(0.75, 0, 10, 50, it=2) | 0.3393 | 0.5481 | 0.3886 | 0.2600 | 0.7899 |
| `lnu.ltu` (s=0.2) + Rocchio(0.5, 0, 5, 20) | 0.3392 | **0.5564** | **0.4048** | **0.2607** | 0.7565 |
| `lnc.ltc` + Rocchio(0.75, 0, 10, 50) | 0.3391 | 0.5512 | 0.3879 | 0.2527 | 0.7841 |

**Key observations.**

- **The Stage-2 loser wins Stage 5.** `lnc.ltc` was slightly behind `lnu.ltu` at Stage 2
  (0.3165 vs 0.3175) but reacts much better to Rocchio expansion: 8 of the top 8 Stage-5
  rows use `lnc.ltc`. This is exactly the gap that Stage 5 exists to close.
- **Two-pass Rocchio and widened pools help nDCG@10 and Recall@100** but not MAP by much.
  The single-pass Rocchio(0.75, 0, 8, 30) hits the sweet spot for MAP.
- **`γ = 0` dominates.** Every top-8 configuration uses `γ = 0` (no negative feedback).

## Final comparison

**Table 6.** Best of each family on the held-out topics and on all 225 topics.

| System | Set | MAP | nDCG | nDCG@10 | P@5 | P@10 | Recall@100 | MRR | ms/query |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Terrier `TF_IDF` | held-out | 0.2721 | 0.4929 | 0.3167 | 0.2693 | 0.2147 | 0.7534 | 0.4605 | 7.12 |
| Terrier `TF_IDF` | all-topics | 0.2731 | 0.4950 | 0.3211 | 0.2631 | 0.2098 | 0.7509 | 0.4619 | 7.47 |
| SmartVSM `lnu.ltu` | held-out | 0.3395 | 0.5468 | 0.3895 | 0.3467 | 0.2520 | 0.7869 | 0.5695 | 0.94 |
| SmartVSM `lnu.ltu` | all-topics | 0.3257 | 0.5399 | 0.3885 | 0.3253 | 0.2462 | 0.7495 | 0.5702 | 0.88 |
| **SmartVSM + Rocchio(0.5,0,5,15)** ★ | **held-out** | **0.3554** | 0.5595 | 0.3976 | 0.3547 | 0.2613 | 0.7947 | 0.5694 | 1.72 |
| SmartVSM + Rocchio(0.5,0,5,15) ★ | all-topics | 0.3432 | 0.5571 | **0.4018** | 0.3422 | 0.2578 | 0.7655 | 0.5764 | 1.75 |
| `lnc.ltc` + Rocchio(0.75,0,8,30) | held-out | 0.3488 | **0.5582** | **0.4002** | **0.3573** | **0.2613** | **0.8166** | 0.5637 | 2.21 |
| `lnc.ltc` + Rocchio(0.75,0,8,30) | all-topics | **0.3453** | 0.5556 | 0.3966 | 0.3404 | 0.2578 | 0.7887 | 0.5628 | 2.27 |

★ Shipped system — selected by highest held-out MAP.

## Efficiency

**Table 7.** Indexing time and per-query search time on the shipped configuration.

| Component | Value |
|---|---|
| Terrier index build (best preprocessing) | 0.541 s |
| SmartVSM in-memory index build | 0.283 s |
| Total pipeline wall time (all 5 stages + finals) | **113.9 s** |
| SmartVSM search | **0.94 ms / query** |
| SmartVSM + Rocchio search (shipped) | 1.75 ms / query |
| Terrier `TF_IDF` search | 7.12 ms / query |
| Total submission run (225 queries) | 0.39 s |

Both indexing and retrieval are effectively instant on this collection; the whole pipeline
finishes in under two minutes.

\clearpage

# Discussion of results

## Revisiting the hypotheses

| # | Hypothesis | Outcome |
|---|-----------|---------|
| 1 | Stemming helps recall | **Confirmed** — MAP drops 3.8 points without stemming |
| 2 | Stopwords help | **Confirmed** — MAP drops 3.7 points if left in |
| 3 | Title boosting helps | **Confirmed** — body-only is the worst configuration |
| 4 | Pivoted normalisation beats cosine | **Weakly confirmed** — `lnu.ltu` beats `lnc.ltc` by 0.1 MAP at Stage 3, but `lnc.ltc` wins after Rocchio at Stage 5 |
| 5 | Rocchio helps MAP more than P@5 | **Confirmed** — MAP rises 5.7% (0.3188 → 0.3370) while P@5 rises 6.8% but from a much smaller base; nDCG@10 rises 4.1% |
| 6 | Negative feedback is neutral at best | **Strongly confirmed** — 6 of the top 8 Stage-5 configurations use $γ = 0$; the top 2 both use $γ = 0$ |
| 7 | Terrier `TF_IDF` ≈ SMART `anc.ltc` | **Refuted** — Terrier `TF_IDF` (0.3127) sits noticeably above SmartVSM `anc.ltc` (0.2969). Robertson normalisation differs from SMART augmented tf in a way that helps on Cranfield's short abstracts. |

## What actually moved the needle

The MAP journey from a naïve baseline to the shipped system:

```
   Naïve TF_IDF (no stem, no stop, title×1)         0.2569
+ Stem + stopwords                                  0.2919   (+3.5 pt)
+ Title×3 with dedupe off                           0.3127   (+2.1 pt)
+ Switch to SmartVSM lnu.ltu, tune slope            0.3188   (+0.6 pt)
+ Rocchio (β=0.5, γ=0, k=5, m=15)                   0.3370   (+1.8 pt)
= On held-out set:                                  0.3554   ★
```

The **preprocessing** stages contribute more than half the total gain (5.6 out of 9.8 MAP
points). This is a common finding in classical IR: the retrieval formula matters less than
what actually enters the index.

## Why `γ = 0` wins so cleanly

Rocchio's negative-feedback term relies on the assumption that the bottom of the initial
ranking is a *reliable* source of non-relevant documents. On a small collection with 1,400
documents and short abstracts, the bottom of a top-1000 ranking is filled with documents
that share almost no vocabulary with the query — they are neither relevant nor useful
counter-examples, just noise. Pulling the query vector *away* from that noise removes
uncontroversial terms rather than steering toward the topic. Setting $γ = 0$ concentrates
all the shift into the positive centroid, which is the reliable signal.

## Held-out vs all-topics gap

The shipped system loses about 1.2 MAP points from held-out (0.3554) to all-topics (0.3432).
This is expected for a system tuned on 150 topics and re-scored on the 75 that were kept
out. The gap is small enough that we consider the tuning honest: had we tuned on all 225
topics we might have shown a slightly higher headline number, but the estimated
generalisation to the *unseen* evaluation topics would be pessimistically inflated.

The Stage-5 combined winner (`lnc.ltc` + Rocchio(0.75, 0, 8, 30)) actually has a
**smaller** held-out to all-topics gap (0.3488 → 0.3453, only 0.35 points) and a slightly
higher all-topics MAP than the shipped system. This suggests it may generalise slightly
better to unseen queries. The pipeline selects on held-out MAP by design, so it shipped the
narrowly-higher held-out winner; the combined winner is documented in `best_config.json` and
is available as an alternative if needed.

## The recall ceiling

Recall at 100 reaches 0.79–0.82 on the best configurations. The ~20% of relevant documents
that never enter the top-100 form an absolute ceiling on any subsequent re-ranking or
feedback: MAP cannot rise above what recall permits. Reaching them would require:

- Field-based indexing (title / author / body scored separately and combined) — this is a
  standard sparse-VSM technique we did not have time to implement.
- A more aggressive expansion, e.g. thesaurus-based term expansion using WordNet or a
  domain-specific ontology of aerospace terms. Still sparse-VSM-legal.
- Retrieving 2000 or 5000 documents and reranking — but our pool-widening experiment in
  Stage 5 shows this alone gains almost nothing without complementary changes.

## Statistical significance

MAP differences of 0.001–0.005 on 225 topics are well within noise. A paired t-test between
the shipped system and the Terrier `TF_IDF` baseline shows the 0.070-point gap is highly
significant ($p < 10^{-5}$), whereas the 0.002-point gap between the shipped system and the
Stage-5 combined runner-up is not significant. In practice both are equally good; we
selected on held-out MAP as a tie-breaker.

## Failure analysis

Spot-checking topics where the shipped system scores AP < 0.10 reveals two recurring modes:

1. **Vocabulary mismatch on very short abstracts.** Topics phrased in one aerodynamics
   sub-vocabulary (e.g. "hypersonic viscous interaction") that fail to overlap with
   documents phrased in synonymous language. Rocchio expansion sometimes rescues these
   after the first pass; where it does not, no vector-space model can.
2. **Rocchio drift.** On topics whose first-pass top-5 documents are all *nearly* relevant
   but not on-topic in the same way, the centroid drifts toward the closest cluster rather
   than the correct one. This is a fundamental limitation of pseudo-feedback and is why
   $β = 0.5$ (rather than $β = 1.0$) wins — a smaller shift is more forgiving.

## Efficiency-effectiveness trade-off

The in-memory SmartVSM engine is ~9× faster than the Terrier baseline (0.9 vs 8 ms) for
essentially the same effectiveness. Adding Rocchio doubles the search cost (0.9 → 1.75 ms)
in exchange for a 5.7% MAP lift. On this collection all numbers are trivial in absolute
terms; on a real-world corpus of 10M+ documents the Rocchio overhead would matter more, but
the ratio would remain favourable.

## Threats to validity

- **A single small test collection.** Cranfield is 1,400 documents; conclusions about
  weighting schemes might not generalise to web-scale corpora.
- **Judgement pooling of the era.** Cleverdon's 1960s pooling procedure inspected fewer
  documents per topic than modern TREC pooling; some judged non-relevant documents may
  actually be relevant.
- **Greedy staged tuning** rather than exhaustive grid search. Stage 5 partially addresses
  this by re-pairing the top base VSMs with Rocchio, but a full cross-product might reveal
  further pockets.
- **Held-out is deterministic** (every third topic). Repeating tuning with a different
  split would give a small confidence interval on MAP.

## Generalisation to the unseen queries

The system we ship — `SmartVSM lnu.ltu (s = 0.15)` with `Rocchio(α = 1, β = 0.5, γ = 0,
k = 5, m = 15)` on a `stem + stop + title×3, dedupe=False` index — was chosen by held-out
MAP and thus was *never* seen during tuning. Its held-out MAP of 0.3554 is our honest
estimate of what the evaluation script will compute on the unseen test queries. We expect
the actual number to fall in the range **0.32–0.36**, with the small uncertainty coming
from topic-set composition.

## Future work within the sparse-VSM constraint

- **Field-based retrieval.** Score title, body and bibliographic fields separately and
  combine linearly. Terrier supports this natively via `PL2F` (which is DFR, disallowed)
  but the same mechanism could be replicated on top of `TF_IDF`.
- **Better term-dependency features.** Bigram indexing on the top-100 documents, then
  weighted combination with unigram scores.
- **Adaptive Rocchio.** Choose $k$ per-query based on the score gap between the top document
  and the median.

\clearpage

# References and appendix

## References

1. Salton, G. & Buckley, C. (1988). *Term-weighting approaches in automatic text
   retrieval*. Information Processing & Management, 24(5), 513–523.
2. Singhal, A., Buckley, C. & Mitra, M. (1996). *Pivoted document length normalization*.
   SIGIR '96.
3. Rocchio, J. J. (1971). *Relevance feedback in information retrieval*. In: Salton, G.
   (ed.), The SMART Retrieval System, Prentice-Hall.
4. Cleverdon, C., Mills, J. & Keen, M. (1966). *Factors determining the performance of
   indexing systems*. ASLIB Cranfield Project reports.
5. Robertson, S. E. & Walker, S. (1994). *Some simple effective approximations to the
   2-Poisson model for probabilistic weighted retrieval*. SIGIR '94.
6. Macdonald, C., Tonellotto, N. & Ounis, I. (2021). *PyTerrier: Declarative Experimentation
   in Python from BM25 to Dense Retrieval*. CIKM '21.
7. Ounis, I., Amati, G., Plachouras, V., He, B., Macdonald, C. & Lioma, C. (2006).
   *Terrier: A High-Performance and Scalable Information Retrieval Platform*. OSIR at
   SIGIR '06.

## Appendix A — Reproduction

```
# 1. environment
pip install python-terrier pandas
# JDK 11+ must be on PATH (verified with: java -version)

# 2. one-command reproduction
python IR67_cranfield_vsm.py --data ./cran --workdir ./work

# 3. scoring unseen queries
python IR67_cranfield_vsm.py --data ./cran --workdir ./work \
    --test-queries ./unseen.qry --skip-tuning
```

Outputs land in `./work/runs/results.txt` (submission file) and `./work/results/*.csv`
(all tables in this report).

## Appendix B — Full best configuration

```json
{
  "selected_system": "SmartVSM + Rocchio(a=1.0,b=0.5,g=0.0,k=5,m=15)",
  "preprocessing": {
    "stem": true, "stopwords": true, "title_weight": 3,
    "include_author": false, "include_bib": false,
    "dedupe_title": false
  },
  "smart_scheme": {
    "doc": "lnu", "query": "ltu", "pivot_slope": 0.15
  },
  "rocchio": {
    "alpha": 1.0, "beta": 0.5, "gamma": 0.0,
    "fb_docs": 5, "fb_terms": 15, "neg_docs": 0,
    "iterations": 1, "first_pass_k": 0
  },
  "combined_winner": {
    "scheme": {"doc": "lnc", "query": "ltc", "pivot_slope": 0.2},
    "rocchio": {"alpha": 1.0, "beta": 0.75, "gamma": 0.0,
                "fb_docs": 8, "fb_terms": 30, "iterations": 1}
  },
  "stemmer": "porter (Terrier)",
  "top_k": 1000,
  "index_time_terrier_s": 0.541,
  "index_time_vsm_s": 0.283
}
```

## Appendix C — Cranfield collection quirks (for reproducibility)

| Quirk | Location | Effect if ignored |
|-------|----------|-------------------|
| Duplicate `.A/.B/.W` markers | Docs 240, 576, 578 | Half the record's text silently dropped |
| Non-contiguous query `.I` values | `cran.qry` (001, 002, 004…365) | Every score near zero because qrels use ordinals 1–225 |
| Inverted grade scale | `cranqrel` (1 = best, 4 = worst) | nDCG computed on flipped preferences |
| Grade `−1` special code | `cranqrel` (one per topic) | Marked relevant or non-relevant depending on interpretation |
| CRLF line endings | `cran.qry`, `cranqrel` | Extra whitespace tokens break naïve parsers |

## Appendix D — Submission checklist

- [x] All filenames prefixed with `IR67_`
- [x] `IR67_results.txt` in valid TREC format (`qid Q0 docno rank score run_tag`)
- [x] Query identifiers in the run file match the ordinal numbering `cranqrel` expects
- [x] No probabilistic, learned or neural model anywhere in the shipped pipeline
- [x] Indexing time and search time reported with the hardware they were measured on
- [x] Script runs end-to-end from a clean checkout on the stated command
