# Legacy Word (`.doc`) extraction

This describes the current working-tree implementation, not a guarantee that every Word file is fully represented as Markdown.

## Dependencies and route

The preferred route for Word 97–2003 binaries is:

```text
.doc → LibreOffice native DOCX conversion → python-docx → source text
```

Install the Python dependency from this checkout:

```bash
python -m pip install -e '.[docx]'
```

LibreOffice is a separate system/user-local application; Python extras do not install it. Install it using your platform's supported installer and expose `soffice` or `libreoffice` on the CLI's `PATH`. Verify the environment that actually launches `olw`:

```bash
command -v soffice || command -v libreoffice
soffice --headless --version
python -c 'import docx; print(docx.__version__)'
```

If only the `libreoffice` executable exists, use that name for the version check. Desktop applications may inherit a different `PATH` from your interactive shell.

For an extraction-only check, use a disposable vault:

```bash
olw ingest /path/to/disposable-vault --url /path/to/document.doc --preview --json
```

`--url` also accepts supported local document paths. Preview performs extraction but does not write source notes or call synthesis. Remove `--preview` and use `--skip-synthesis` to persist the extracted source without building generated notes. Inspect diagnostics before doing so.

## Why not convert through PDF?

A PDF intermediate re-derives text from page layout. In the tested legacy Word sample, that route changed curly apostrophes and paragraph wrapping. In an authored EN/CN control on a host without CJK fonts, it also lost Chinese text. The native DOCX route preserves the stored text rather than relying on rendered glyphs.

If either LibreOffice or python-docx is unavailable, the dispatcher may attempt the optional LiteParse compatibility route. It attaches a warning about punctuation/layout loss. That fallback is not equivalent fidelity, and its own `.doc` conversion can still require LibreOffice. Conversion failure, encryption and unsupported Word formats do **not** trigger this compatibility fallback.

## Admission, isolation and bounds

The converter:

- Inspects the compound-file signature and Word stream before conversion. Recognized encryption, non-Word files and Word 6/95 formats are rejected.
- Runs on a private copy, with a fresh LibreOffice user profile and a forced `MS Word 97` import filter. The filter matters: automatic detection can import corrupt binary bytes as plain text while exiting successfully.
- Uses headless mode, no stdin, bounded stdout/stderr and a wall-clock deadline. A timeout terminates the process group.
- Rejects nonzero exit, missing/empty output and output without a DOCX `word/document.xml` part.
- Removes temporary copies/profiles and does not modify the caller's document.

Relevant configuration defaults:

| Setting | Default | Scope |
|---|---:|---|
| `MAX_DOCUMENT_BYTES` | 50,000,000 | Document acquisition and converted DOCX size |
| `PARSER_TIMEOUT_SECONDS` | 120 | Native parser process deadline |
| `MAX_PARSER_STDOUT_BYTES` | 1,000,000 | Captured process stdout |
| `MAX_PARSER_STDERR_BYTES` | 16,384 | Captured process stderr |

The converter also checks the declared size of `word/document.xml` against a bounded expansion factor. These checks are **not an OS sandbox or a universal memory/decompression bound for every document part**. Use an appropriately isolated execution environment for untrusted documents.

## What is preserved—and what is not

The text extractor preserves paragraph/table ordering, headings and supported list markers, including Unicode punctuation and Chinese text in the exercised controls.

The native conversion path reports omitted images, embedded OLE objects, text boxes, footnotes, endnotes, comments and text-bearing headers/footers where detected. It does not extract their full content. A successful body-text extraction therefore does not imply complete document fidelity.

Fresh-source and freshly extracted body-duplicate CLI events include nonempty `extraction_diagnostics`; human-readable output prints the same caveats as extraction notes. For example:

```json
{
  "status": "succeeded",
  "extraction_diagnostics": [
    "libreoffice-docx: 1 embedded OLE object(s) not extracted; content not verified"
  ]
}
```

This is an illustrative event fragment, not a full NDJSON run. `succeeded` means the extraction operation returned usable body text; it does not certify the omitted content or any later synthesis.

Known limits:

- Word 6/95 and earlier formats are unsupported.
- A Word stream stored in the OLE mini stream is not classified by the pre-inspector; the strict LibreOffice filter remains the final format check.
- LibreOffice's own Word-97 export can lose table-cell boundaries when the source ends in a table without a final paragraph mark. Text survival is not table-layout fidelity.
- Complex fields, typography, images and full Word revision/rendering semantics are not certified by plain-text controls.

## Verification scope

The exercised environment used **LibreOffice 26.8.1.1**. For the public `sample2.doc` fixture, all **43 defined reference text segments** were present both in an independent MS-DOC piece-table decoding and in the current production extraction. Both source curly apostrophes survived; the input binary remained unchanged. This is a captured-input text-fidelity result, not a claim that every binary feature was extracted.

A separate fresh public-URL CLI preview succeeded, returned the document's actual heading as its title, exposed the omitted-content diagnostics, and left its disposable vault unchanged. The NDJSON preview reports counts and diagnostics, not the complete extracted text; its byte count alone is not the fidelity test.

Real integration tests are in [`tests/integration/test_libreoffice_doc_cli.py`](../tests/integration/test_libreoffice_doc_cli.py). They exercise an authored EN/CN Word-97 round trip and corrupt-OLE rejection. Unit/CLI controls cover dependency failures, output checks, document order and diagnostic propagation:

```bash
python -m pytest tests/new/test_office_documents.py tests/new/test_docx_document_order.py tests/integration/test_libreoffice_doc_cli.py tests/new/test_ingest_extraction_diagnostics.py -q -rs
```

Check the skip summary: the real integration tests skip explicitly when their optional dependencies are absent. These document checks do not certify the separate LLM support reviewer or complete a repository-wide release gate.
