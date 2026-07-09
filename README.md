# Probabilistic Earley Parser

A probabilistic parser for [Context-Free Grammars](https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar) (PCFGs) based on [Earley's algorithm](https://en.wikipedia.org/wiki/Earley_parser). Given a weighted grammar and a list of tokenized sentences, it reconstructs the **highest-probability parse tree** of each sentence and prints it as a bracketed S-expression, together with the total weight of that parse.

Authors: **Harry Qi**, **Zike Hu**

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [File Formats](#file-formats)
- [Examples](#examples)
- [Performance Notes](#performance-notes)
- [Acknowledgements](#acknowledgements)

---

## Overview

Earley's algorithm is a chart-based parsing algorithm that accepts any context-free grammar (CFG) and runs in **O(n³)** time in the worst case, and linear time on unambiguous grammars. This project extends a plain (unweighted) Earley recognizer into a **probabilistic parser**:

- Each grammar rule carries a probability `p`. The parser converts it to a weight `w = -log2(p)`, so that the *most probable* parse is the one with the *minimum total weight*.
- During chart construction, every item accumulates the weight of the rules and sub-parses it is built from, and keeps **backpointers** to the child items that produced it.
- After the chart is complete, the parser picks the completed root item with the smallest weight and follows the backpointers to reconstruct the best parse tree.

Three implementations are provided:

| File | Description |
|------|-------------|
| `src/recognize.py` | The original **unweighted recognizer**. It only answers the recognition question — is the sentence grammatical? — and does not reconstruct a tree. |
| `src/parse.py` | The **basic probabilistic parser**. Adds weights, backpointers, minimum-weight root selection, and S-expression reconstruction. |
| `src/parse2.py` | An **optimized probabilistic parser**. Same output as `parse.py`, but with extra indexing in the chart to avoid the expensive linear scan during the *attach* step, making it substantially faster on large grammars. |

---

## How It Works

The chart is a list of **columns** (one per token position). Each column holds an **agenda** — a FIFO queue with duplicate detection. Parsing proceeds column by column, applying three operations to each item:

1. **PREDICT** — if the next symbol after the dot is a nonterminal, seed new items for each of its expansions.
2. **SCAN** — if the next symbol is a terminal matching the input token, advance the dot and move the new item into the next column.
3. **ATTACH** (a.k.a. *complete*) — if an item is complete (the dot is at the end), find earlier items waiting for its left-hand side and advance their dots.

### Probabilistic extensions (in `src/parse.py`)

- `Item` stores a `weight` (the sum of the rule weights along the path) and a `backPointer` list recording the child item that filled each slot in the right-hand side.
- When two items with the same `(rule, dot_position, start_position)` are pushed, the agenda keeps the one with the **smaller weight** and retires the other (via a `flag`), so the chart only ever retains the best derivation of each state.
- After parsing, `accepted()` collects every complete root item (a `ROOT` item that started at position 0 and sits in the final column), and `output()` selects the minimum-weight one and recursively prints the tree.

### Optimization in `src/parse2.py`

The basic `_attach` step originally scans an entire earlier column to find *customers* (items whose next symbol matches the completed item's LHS). `parse2.py` augments the `Agenda` with a `_lhs_symbol_index` mapping each LHS to the set of items waiting on it, so the lookup is direct. A `_next_symbol_index` is also maintained for the same purpose. This eliminates the dominant cost of the algorithm on large grammars.

---

## Repository Structure

```
.
├── README.md
├── .gitignore                 # Ignores macOS metadata (._*, .DS_Store) and editor files
├── src/                       # Earley parser implementations (Python 3)
│   ├── recognize.py           #   Unweighted recognizer (baseline)
│   ├── parse.py               #   Probabilistic parser (basic)
│   └── parse2.py              #   Probabilistic parser (optimized)
├── scripts/                   # Helper utilities (Perl)
│   ├── prettyprint            #   Pretty-prints bracketed parse trees from stdin
│   └── checkvocab             #   Checks sentence tokens against a grammar
├── grammars/                  # PCFG grammar files (*.gr)
│   ├── papa.gr                #   Tiny toy grammar ("Papa ate the caviar")
│   ├── arith.gr               #   Grammar for arithmetic expressions
│   ├── permissive.gr          #   Minimal recursive grammar (A -> A A | x)
│   ├── permissive2.gr         #   Slightly larger recursive grammar
│   └── wallstreet.gr          #   Large PCFG from the Wall Street Journal treebank
├── sentences/                 # Tokenized input sentences (*.sen)
│   ├── papa.sen
│   ├── papaTest.sen           #   A long, deeply-ambiguous test sentence
│   ├── arith.sen
│   ├── arithtest.sen
│   ├── permissive.sen
│   ├── wallstreet.sen
│   └── wallstreetTest.sen
├── reference/                 # Reference parse outputs (*.par)
│   ├── arith.par              #   Expected output for arith.sen
│   └── wallstreet.par         #   Expected output for wallstreet.sen
└── docs/                      # Project instructions & background reading
    ├── INSTRUCTIONS.md
    ├── INSTRUCTIONS.html
    ├── NLP-HW4.pdf
    ├── hw-parse.pdf
    ├── treebank-manual.pdf
    └── treebank-notation.pdf
```

---

## Installation

The parsers are written in **Python 3** and depend only on the [`tqdm`](https://github.com/tqdm/tqdm) progress-bar library.

```bash
pip install tqdm
```

### Recommended: use PyPy

Earley parsing is CPU-intensive, especially on the `wallstreet.gr` grammar. For a significant speedup, run the parser under [PyPy](https://www.pypy.org/):

```bash
pypy src/parse.py  grammars/papa.gr sentences/papa.sen
pypy src/parse2.py grammars/wallstreet.gr sentences/wallstreet.sen
```

The helper scripts `scripts/prettyprint` and `scripts/checkvocab` are written in Perl and require a `perl` interpreter on your `PATH`. Make them executable first:

```bash
chmod +x scripts/prettyprint scripts/checkvocab
```

> **Note:** All example commands below assume you are running from the **repository root** directory.

---

## Usage

All three parsers/recognizer share the same command-line interface:

```
python src/<script>.py GRAMMAR SENTENCES [-s START_SYMBOL] [-v | -q] [--progress]
```

| Argument | Description |
|----------|-------------|
| `GRAMMAR` | Path to a `.gr` file containing a PCFG (e.g. `grammars/papa.gr`). |
| `SENTENCES` | Path to a `.sen` file with one tokenized sentence per line (e.g. `sentences/papa.sen`). |
| `-s, --start_symbol` | Start symbol of the grammar (default: `ROOT`). |
| `-v, --verbose` | Verbose logging output (debug-level). |
| `-q, --quiet` | Quiet logging output (warnings only). |
| `--progress` | Display a `tqdm` progress bar while parsing. |

### Recognizer

Prints whether each sentence is accepted or rejected:

```bash
python src/recognize.py grammars/papa.gr sentences/papa.sen
python src/recognize.py -v grammars/papa.gr sentences/papa.sen      # verbose
```

### Parser

For each sentence, prints the best parse tree as a one-line S-expression, followed by its total weight (a value of `-log2` probability, so **smaller is better**). Ungrammatical sentences print `NONE`:

```bash
python src/parse.py  grammars/papa.gr sentences/papa.sen
python src/parse2.py grammars/wallstreet.gr sentences/wallstreet.sen --progress
```

### Pretty-printing the trees

Pipe the parser output through `scripts/prettyprint` to get an indented, readable tree:

```bash
python src/parse.py grammars/papa.gr sentences/papa.sen | scripts/prettyprint
```

### Checking vocabulary coverage

Before parsing, you can check that every token in your sentence file actually appears in the grammar:

```bash
scripts/checkvocab grammars/wallstreet.gr sentences/wallstreet.sen
```

It warns about any out-of-vocabulary tokens, which would otherwise lead to failed parses.

---

## File Formats

### Grammar files (`grammars/*.gr`)

Tab-separated, one rule per line:

```
<probability>	<lhs>	<rhs symbols...>
```

- `<probability>` is a normalized probability for that expansion of `<lhs>` (the probabilities of all rules sharing an LHS should sum to 1).
- `<lhs>` is a nonterminal symbol.
- `<rhs>` is a space-separated sequence of terminals and/or nonterminals.
- Lines beginning with `#` (or text after `#`) are treated as comments and ignored, as are blank lines.

Example (from `grammars/papa.gr`):

```
1	ROOT	S
0.8	NP	Det N
0.1	NP	NP PP
0.7	VP	V NP
0.5	N	caviar
```

### Sentence files (`sentences/*.sen`)

Plain text, one **whitespace-tokenized** sentence per line. Blank lines are skipped.

Example (from `sentences/papa.sen`):

```
Papa ate the caviar
Papa ate the caviar with a spoon
the caviar ate Papa with a spoon
```

### Parse output (`reference/*.par`)

Each accepted sentence produces two lines: the bracketed parse tree, then its weight. Rejected sentences produce `NONE`. See `reference/arith.par` and `reference/wallstreet.par` for reference outputs.

---

## Examples

### A small grammar

```bash
$ python src/parse.py grammars/papa.gr sentences/papa.sen | scripts/prettyprint
(ROOT (S (NP (N Papa)) (VP (V ate) (NP (Det the) (N caviar)))))
...
```

The second line of each pair is the weight (e.g. `8.0`), where smaller means a more probable parse.

### Arithmetic expressions

```bash
$ python src/parse.py grammars/arith.gr sentences/arith.sen | scripts/prettyprint
```

yields properly nested trees such as:

```
(ROOT (EXPR (TERM (TERM (FACTOR (Num 3)))
                  *
                  (FACTOR (Num 5)))))
```

### A real-world grammar

The `grammars/wallstreet.gr` grammar has over 10,000 rules extracted from the Wall Street Journal treebank. This is where `src/parse2.py` and PyPy make a real difference:

```bash
pypy src/parse2.py grammars/wallstreet.gr sentences/wallstreet.sen --progress | scripts/prettyprint
```

---

## Performance Notes

- The worst-case time complexity of Earley parsing is **O(n³)** in the sentence length *n*, and it can be far worse in practice on highly ambiguous grammars because the number of distinct items grows quickly.
- `src/parse.py` performs a **linear scan** of earlier columns during every ATTACH, which dominates the runtime on large grammars. `src/parse2.py` replaces this with **hash-indexed lookups** (`_next_symbol_index`, `_lhs_symbol_index`), giving a large constant-factor speedup.
- On `wallstreet.gr`, prefer `src/parse2.py` over `src/parse.py`, and run it under **PyPy** rather than CPython.
- The `--progress` flag shows how far the parser has progressed through the chart columns; `-v` additionally reports the number of PREDICT, SCAN, and ATTACH operations per sentence.
- Highly ambiguous sentences (e.g. the long one in `sentences/papaTest.sen`) exercise the parser's handling of exponentially many parses and are a good way to compare the two implementations.

---

## Acknowledgements

The baseline `src/recognize.py` was originally written by **Arya McCarthy, Alexandra DeLucia, and Jason Eisner** and released to the public domain. The `scripts/prettyprint` and `scripts/checkvocab` utilities were authored by Jason Eisner. The probabilistic parsing logic, backpointer-based tree reconstruction, and chart optimizations in `src/parse.py` and `src/parse2.py` are the work of the present authors.
