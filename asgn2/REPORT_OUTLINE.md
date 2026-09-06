# Report Outline — Vector Space Model and Ranked Retrieval (Cranfield)

Target length ≈ 10 pages. Suggested page budget in brackets. Render with
`pandoc REPORT.md -o <group>_report.pdf --toc -V geometry:margin=1in`.

---

## 1. Group details [0.25 p]

- Group name / number (and use it as the prefix of every submitted filename).
- Members: name, roll number, and the part each person owned (ingestion,
  indexing, model implementation, tuning, evaluation, write-up).
- Environment: OS, CPU/RAM, Python version, `python-terrier` version, Terrier
  core version, JDK version. Timings are meaningless without this.
- File manifest: `cranfield_vsm.py`, `runs/results.txt`, `results/*.csv`,
  `results/best_config.json`, this report.

## 2. Problem statement [0.5 p]

- Task: ranked retrieval over the Cranfield collection (1400 aerodynamics
  abstracts, 225 topics, 1836 graded judgements), evaluated against the
  supplied relevance assessments and then against an unseen query set.
- Scope constraint: sparse vector space models only — no BM25/DFR/language
  models, no learning-to-rank, no dense or neural retrieval. State explicitly
  which models this admits and why each one you used qualifies.
- What "maximise performance" means here: name the primary metric you optimise
  (MAP is the usual choice) and the secondary ones you report.
- Efficiency requirement: indexing time and search time must be measured.

## 3. Implementation details [2.5 p]

### 3.1 Pipeline overview
One diagram: `cran.tar.gz → parse → analyse → index → retrieve → rank → evaluate`,
annotated with the module or function that owns each box.

### 3.2 Data ingestion
- Archive handling; the `.I/.T/.A/.B/.W` record grammar.
- Collection quirks you had to defend against, with evidence:
  - records **240, 576, 578** repeat a field marker, so the parser appends to
    the currently open field rather than assuming one marker per field;
  - the `.W` abstract restarts with the title, which is de-duplicated so title
    emphasis is controlled by one explicit parameter;
  - `cran.qry` holds 225 topics whose `.I` values are **non-contiguous**
    (001, 002, 004 … 365) with CRLF line endings, while `cranqrel` keys
    judgements on the **ordinal position** 1–225 — describe the renumbering and
    the `query_id_map.tsv` you emit. Getting this wrong silently destroys every
    score, so it deserves a paragraph.
- Qrel semantics: Cleverdon grades 1 (complete answer) … 4 (minimal interest)
  plus a −1 code; justify the inversion `label = 5 − grade` for graded nDCG and
  the treatment of −1 (225 such rows, one per topic).

### 3.3 Preprocessing
Case folding, non-alphanumeric stripping (`/slashed/` emphasis, hyphens,
sentence-final ` .`), digit retention (mach 2, naca 0012), stopword removal,
Porter stemming, minimum token length. Note where Terrier's internal term
pipeline and your own analyser must agree, and how you kept them aligned.

### 3.4 Indexing
`IterDictIndexer` configuration, the `meta` field, the term-pipeline settings
per configuration, and the resulting statistics (documents, unique terms,
pointers, on-disk size) for the chosen index.

### 3.5 Retrieval models
For each model: the scoring formula, its parameters, and one sentence on why it
is a sparse VSM.
- Terrier `Tf`, `TF_IDF` (Robertson tf normalisation, parameter `c`),
  `LemurTF_IDF`, `CoordinateMatch`.
- Own SMART `ddd.qqq` cosine implementation: tf letters `n/l/a/b/L`, idf letters
  `n/t/p`, normalisation `n/c/u`; give the pivoted unique-term normalisation
  denominator `(1−s)·pivot + s·|unique terms|` and explain the pivot.
- Rocchio pseudo-relevance feedback: `q' = αq + (β/|D_r|)Σd − (γ/|D_n|)Σd`,
  the truncation to the top-*m* terms, and why Rocchio (not RM3/Bo1) is the
  vector-space-legal expansion method.

### 3.6 Evaluation harness
`pt.Experiment` metrics, the equivalent built-in scorer used as a cross-check,
and how timing is measured (best of *n* repetitions, wall clock, ms/query).

## 4. Plan of experiments [1.5 p]

- **Methodology first**: the deterministic train / held-out split (every third
  topic held out), and the rule that *all* tuning decisions are taken on the
  training half only. Say plainly that this is what protects you from
  over-fitting to the 225 released topics when the unseen set arrives.
- Staged (greedy) search rather than a full cross-product, with the reason:
  a full grid over preprocessing × model × parameters × feedback is
  combinatorially wasteful on a collection this small.
  - Stage 1 — preprocessing: stemming × stopwords × title weight (0–3) ×
    author/bibliography inclusion, held at TF_IDF.
  - Stage 2 — model family on the winning index.
  - Stage 3 — parameters: Terrier `c ∈ [0.1, 2.0]`; pivot slope `s ∈ [0.05, 0.4]`.
  - Stage 4 — Rocchio grid over α, β, γ, feedback documents, expansion terms.
  - Stage 5 — final: re-score the finalists on the held-out half and the full
    topic set, export runs.
- Metrics and why each is reported: MAP (primary, whole-ranking), nDCG and
  nDCG@10 (use the graded judgements the collection provides), P@5 / P@10
  (early precision), Recall@100 (recall ceiling for the feedback stage), MRR.
- Hypotheses to test — write them *before* the numbers, then check them in §6.
  e.g. "stemming helps recall on a small, morphologically repetitive technical
  vocabulary"; "title boosting adds little because `.W` already repeats the
  title"; "pivoted normalisation beats plain cosine because Cranfield abstracts
  vary in length"; "Rocchio helps MAP more than P@5".

## 5. Results [3 p]

Tables straight from `results/*.csv`; bold the best value per column.

- **Table 1** — preprocessing configurations × metrics (+ indexing time).
- **Table 2** — model comparison on the fixed best index (+ ms/query).
- **Table 3** — parameter sweeps: `c` and pivot slope. Add a line plot of MAP
  against each parameter; the shape of the curve is the argument, not the peak.
- **Table 4** — Rocchio grid.
- **Table 5** — final: training vs held-out vs all topics, for every finalist.
  The training-to-held-out drop is the number a grader looks for.
- **Table 6** — efficiency: index build time, index size, total search time,
  mean ms/query, for Terrier and for the in-memory VSM.
- Figures: MAP-vs-parameter curves; an interpolated precision–recall curve for
  the finalists; optionally a per-topic MAP delta plot (best system minus
  baseline) to show where the gains come from.
- State the exact command line that reproduces each table.

## 6. Discussion of results [2 p]

- Revisit each hypothesis from §4: confirmed, refuted, or inconclusive.
- Effectiveness: which weighting component mattered most (tf damping, idf,
  length normalisation)? Cranfield queries are long and verbose — say what that
  implies for the query-side scheme.
- Statistical care: differences of one or two MAP points on 225 topics are
  usually not significant. Report a paired t-test or a bootstrap interval
  between your best system and the TF_IDF baseline, or explicitly state that
  you did not test and treat small gaps as ties.
- Failure analysis: two or three topics where the best system does badly, with
  the likely cause (vocabulary mismatch, very short abstract, feedback drift
  after a bad first pass).
- Efficiency/effectiveness trade-off: what Rocchio's second pass costs per
  query, and whether it is worth it.
- Threats to validity: single small test collection, judgement pooling of the
  era, −1 grade handling, greedy rather than exhaustive tuning, one split.
- Generalisation to the unseen queries: which configuration you shipped in
  `results.txt` and why you trust it — this is the paragraph that justifies your
  submission.
- Future work within the sparse-VSM constraint: better length normalisation,
  term-dependency features, thesaurus-based expansion.

## 7. References and appendix [0.25 p]

Salton & Buckley (1988) on term weighting; Singhal, Buckley & Mitra (1996) on
pivoted normalisation; Rocchio (1971); the Cranfield/Cleverdon reports;
PyTerrier and Terrier documentation. Appendix: full parameter grids, the
`best_config.json` dump, and reproduction instructions.

---

### Checklist before submitting

- [ ] Every filename prefixed with the group name.
- [ ] `results.txt` is valid TREC format: `qid Q0 docno rank score run_tag`.
- [ ] Query identifiers in the run file match the numbering the evaluation
      script expects (ordinal, per `cranqrel`) — mention the mapping file.
- [ ] No probabilistic, learned or neural model anywhere in the shipped
      pipeline; the constraint is stated in the report.
- [ ] Indexing and search times reported with the hardware they were measured on.
- [ ] The script runs end-to-end from a clean checkout on the stated command.
