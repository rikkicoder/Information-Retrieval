# How to compile the report to PDF

The report is written in Markdown (`IR67_report.md`) and converts to a clean 10-page PDF
with **pandoc + a LaTeX engine**. Pick whichever route is easiest for you.

---

## Option 1 — Direct PDF via pandoc + xelatex (recommended)

**One-line install on Windows:**

```powershell
winget install --id JohnMacFarlane.Pandoc
winget install --id MiKTeX.MiKTeX
```

**Compile:**

```powershell
pandoc IR67_report.md -o IR67_report.pdf --pdf-engine=xelatex --toc
```

The first run of `xelatex` will prompt MiKTeX to fetch a few missing packages — click
"Install" each time. That happens once; subsequent runs are instant.

---

## Option 2 — Direct PDF via pandoc + pdflatex

If MiKTeX complains about `TeX Gyre Termes` or fonts, remove the `mainfont:` line from the
YAML block at the top of the report and use pdflatex instead:

```powershell
pandoc IR67_report.md -o IR67_report.pdf --toc
```

---

## Option 3 — Overleaf (no local install)

1. Create a new Overleaf project.
2. Convert to LaTeX first:
   ```powershell
   pandoc IR67_report.md -o IR67_report.tex --standalone --toc
   ```
3. Upload `IR67_report.tex` to Overleaf, click **Compile**.

---

## Option 4 — Google Docs / Word (fallback)

If none of the above works:

```powershell
pandoc IR67_report.md -o IR67_report.docx --toc
```

Open `IR67_report.docx` in Word or Google Docs, review formatting, then export to PDF via
File → Save As PDF.

---

## Before compiling — fill in the five member names

Open `IR67_report.md` in any text editor and find the members table at the top:

```
| # | Name | Roll number | Role |
|---|------|-------------|------|
| 1 | *[FILL IN]* | *[FILL IN]* | Data ingestion, ...
| 2 | *[FILL IN]* | *[FILL IN]* | Terrier indexing, ...
```

Replace each `*[FILL IN]*` with the actual name and roll number. Everything else in the
report is pre-filled with your real experimental results.

---

## Sanity check

The finished PDF should be **9–10 pages** with:

- Cover page + TOC
- 1 page: Group details
- 1 page: Problem statement
- 2–3 pages: Implementation details
- 1–2 pages: Plan of experiments
- 2–3 pages: Results (with 7 tables)
- 2 pages: Discussion
- 1 page: References + appendices

If it comes out shorter than 8 pages, check that the tables rendered correctly — long tables
sometimes get truncated without warning.
