# BayesCaTrack artifact workflow

Long-running BayesCaTrack experiments should run in this public code repository. The private paper repository should consume frozen result artifacts and should not rerun the benchmark.

## Motion-stratified ACCAD artifact

For a GitHub-hosted sample run, dispatch the **ACCAD Motion-Stratified Benchmark** workflow and leave `download_sample=true`. It uploads an artifact named:

```text
bayescatrack-accad-motion-stratified-<run-number>-<sha>
```

For the full private ACCAD/AMASS checkout, dispatch the **Full-Data ACCAD Motion-Stratified Benchmark** workflow. This workflow runs on any self-hosted runner attached to the repository:

```text
self-hosted
```

Before running it, configure the repository or environment secret:

```text
ACCAD_DATA_ROOT=/path/to/ACCAD/on/the/self-hosted/runner
```

An absolute path is recommended because GitHub Actions checks out the repository into a runner work directory that can vary by machine and service configuration. A relative path is still accepted; the workflow resolves it relative to the checked-out repository directory before validating that it exists.

The full-data workflow does not expose the full-data path through workflow inputs, which avoids putting local dataset paths into public workflow dispatch metadata. It also packages only sanitized paper-facing outputs and does not upload raw AMASS/ACCAD `.npz` files or copied motion-bin segment files.

The full-data workflow uploads an artifact named:

```text
bayescatrack-accad-motion-stratified-full-<run-number>-<sha>
```

The sanitized artifact contains aggregate result tables, summary JSON/Markdown files, benchmark summaries, transition metrics, SVG plots, validation metadata, and a redacted `run_manifest.json`.

`run_manifest.json` records the source SHA, workflow run metadata, config hash/content, selected runtime package versions, and output file hashes. For full-data artifacts, local path-like fields are redacted before upload. Keep this manifest beside every paper result snapshot.

## Repository boundary

The paper repository should:

1. download one selected artifact bundle from this workflow,
2. unpack it under `results/accad-motion-stratified/<snapshot>/`,
3. generate publication figures from the CSV/JSON files, and
4. commit the frozen result snapshot plus figure scripts.

It should not execute the AMASS/ACCAD filtering benchmark.
