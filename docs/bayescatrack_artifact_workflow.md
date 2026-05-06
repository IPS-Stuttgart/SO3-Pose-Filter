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

The full-data workflow supports four data-source modes:

```text
owncloud_webdav     Download from the password-protected ownCloud/WebDAV share with rclone, then reuse it from a persistent runner cache.
secret_path         Use ACCAD_DATA_ROOT as a local path on the selected self-hosted runner.
download_and_cache  Download an archive/.npz from dataset_url or ACCAD_DATA_URL, then reuse it from a persistent runner cache.
cached_only         Use the persistent runner cache and fail if it has not been populated before.
```

The default mode is `owncloud_webdav`. Configure these repository or environment secrets:

```text
ACCAD_DATA_WEBDAV_URL=<ownCloud public WebDAV endpoint>
ACCAD_DATA_KEY=<ownCloud public-share token/user>
ACCAD_DATA_PASSWORD=<ownCloud public-share password>
```

The workflow installs `rclone`, then populates the cache with:

```bash
rclone copy :webdav: ./ACCAD_DATA \
  --webdav-url "$ACCAD_DATA_WEBDAV_URL" \
  --webdav-vendor owncloud \
  --webdav-user "$ACCAD_DATA_KEY" \
  --webdav-pass "$(rclone obscure "$ACCAD_DATA_PASSWORD")"
```

For `secret_path`, configure the repository or environment secret:

```text
ACCAD_DATA_ROOT=/path/to/ACCAD/on/the/self-hosted/runner
```

For `download_and_cache`, either provide `dataset_url` when manually dispatching the workflow or configure the repository/environment secret:

```text
ACCAD_DATA_URL=https://example.invalid/path/to/accad-or-amass-archive.zip
```

The dataset is cached under:

```text
$RUNNER_TOOL_CACHE/<cache_subdir>
```

or, if `RUNNER_TOOL_CACHE` is unavailable, under `$RUNNER_TEMP/<cache_subdir>`. On a persistent self-hosted runner this avoids re-downloading the full dataset on later runs. Use `force_redownload=true` to replace the cached copy.

An absolute local path is recommended for `ACCAD_DATA_ROOT` because GitHub Actions checks out the repository into a runner work directory that can vary by machine and service configuration. A relative path is still accepted; the workflow resolves it relative to the checked-out repository directory before validating that it exists.

The full-data workflow does not upload raw AMASS/ACCAD `.npz` files or copied motion-bin segment files. It packages only sanitized paper-facing outputs.
The default full-data config uses balanced motion selection, so the paper-facing artifact should contain low-, medium-, and high-motion ACCAD windows when enough candidates are available.

The full-data workflow uploads an artifact named:

```text
bayescatrack-accad-motion-stratified-full-<run-number>-<sha>
```

The sanitized artifact contains aggregate result tables, statistical summaries, robustness summaries, particle-collapse diagnostics, transition-vs-tracking diagnostics, benchmark summaries, transition metrics, SVG plots, validation metadata, and a redacted `run_manifest.json`.

`run_manifest.json` records the source SHA, workflow run metadata, config hash/content, selected runtime package versions, and output file hashes. For full-data artifacts, local path-like fields are redacted before upload. Keep this manifest beside every paper result snapshot.

## Repository boundary

The paper repository should:

1. download one selected artifact bundle from this workflow,
2. unpack it under `results/accad-motion-stratified/<snapshot>/`,
3. generate publication figures from the CSV/JSON files, and
4. commit the frozen result snapshot plus figure scripts.

It should not execute the AMASS/ACCAD filtering benchmark.
