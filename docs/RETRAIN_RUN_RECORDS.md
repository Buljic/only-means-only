# Retrain run records, twelve arms

Per-arm provenance behind the departure disclosure of the manuscript. The manuscript and
supplement report the departure classes and their claim consequences; this file holds the run-level
record itself, so that the supplement does not have to.

Source: the manifests and metrics sidecars of the collected arms. Weights are the confirmatory
scored weights (DoveNet epoch 25, Harmonizer epoch 60), first 16 hex characters of sha256.
`Seg wall` is the manifest `wall_seconds`, which is the last invocation or segment only for resumed
arms. `Launch commit` is `git.repo.commit` at arm launch and `prov commit` is `provenance.commit` at
invocation start. For resumed DoveNet arms and Harmonizer seed 0 `leak_out`, the finish time equals
the manifest `resumed_utc` and the manifest file modification time. Departure identifiers D01 to D14
label the departures recorded for each arm in the project deviation ruling.

Machine names are generic (`local workstation`, `laptop A`, `laptop B`). GPU models appear only where
they were recorded in the run notes.

## Run records

| Arm | Run root | rng_seed | Start (manifest) | Finish (manifest) | Seg wall (s) | Steps / budget | Resume records | Weights sha256 |
|---|---|---|---|---|---|---|---|---|
| Arm  |  Run root  |  rng\_seed  |  Start (manifest)  |  Finish (manifest)  |  Seg wall (s)  |  Steps / budget  |  Resume \ records  |  Weights sha256 |
| dovenet leak\_in s0  |  20260925T161553Z  |  987369860  |  2026-09-25T16:15:53Z  |  2026-09-29T00:18:11Z  |  288,138.3  |  1,580,650 / 1,580,650  |  0  |  14573bfc61862a29 |
| dovenet leak\_out s0  |  20260925T161553Z  |  987369860  |  2026-09-29T00:18:11Z  |  2026-10-02T23:40:58Z  |  102,379.8  |  1,580,650 / 1,580,650  |  1  |  3801401f19853a18 |
| dovenet leak\_in s1  |  20261003T053442Z  |  527887430  |  2026-10-03T05:34:42Z  |  2026-10-06T02:49:56Z  |  21,974.6  |  1,580,650 / 1,580,650  |  1  |  89200e47cf9259f0 |
| dovenet leak\_out s1  |  20261003T053442Z  |  527887430  |  2026-10-05T20:39:23Z  |  2026-10-07T12:24:17Z  |  143,094.0  |  1,580,650 / 1,580,650  |  0  |  67d214b84c814e92 |
| dovenet leak\_in s2  |  20260925T161553Z  |  792586843  |  2026-09-25T16:15:53Z  |  2026-09-29T00:18:11Z  |  288,138.3  |  1,580,650 / 1,580,650  |  0  |  ce867759468d8aff |
| dovenet leak\_out s2  |  20260925T161553Z  |  792586843  |  2026-09-29T00:18:11Z  |  2026-10-02T23:40:58Z  |  102,379.8  |  1,580,650 / 1,580,650  |  1  |  dbb95abfb8f9c76a |
| harmonizer leak\_in s0  |  20260925T174257Z  |  790756038  |  2026-09-25T17:43:00Z  |  2026-09-28T08:27:33Z  |  225,873.5  |  474,180 / 474,180  |  0  |  53550949cecf47a3 |
| harmonizer leak\_out s0  |  20260928T201758Z  |  790756038  |  2026-09-28T20:18:02Z  |  2026-10-03T07:41:39Z  |  131,217.6  |  474,180 / 474,180  |  1  |  ca4abf2a160a5975 |
| harmonizer leak\_in s1  |  20261002T1835-seed1-return  |  411961215  |  2026-09-26T17:33:18Z  |  2026-09-28T05:44:21Z  |  130,262.6  |  474,180 / 474,180  |  0  |  2f40db089fb97f8d |
| harmonizer leak\_out s1  |  20261002T1835-seed1-return  |  411961215  |  2026-09-28T13:00:15Z  |  2026-09-30T07:33:06Z  |  36,794.5  |  474,180 / 474,180  |  3  |  011e4b7971012172 |
| harmonizer leak\_in s2  |  20261002T2335-seed2-return  |  2067707940  |  2026-09-27T06:43:15Z  |  2026-09-30T03:15:51Z  |  246,755.5  |  474,180 / 474,180  |  0  |  d790712675c5a2c2 |
| harmonizer leak\_out s2  |  20261002T2335-seed2-return  |  2067707940  |  2026-09-30T03:16:54Z  |  2026-10-02T15:07:44Z  |  169,517.7  |  474,180 / 474,180  |  1 (log)  |  edb3905f93542037 |

## Environment and departures

| Arm | Machine (recorded) | torch / python | Workers | Launch commit | Prov commit | Departures |
|---|---|---|---|---|---|---|
| Arm  |  Machine (recorded)  |  torch / python  |  Workers  |  Launch commit  |  Prov commit  |  Departures |
| dovenet leak\_in s0  |  \raggedright local workstation (`.venvs/dovenet-cuda`); GPU model not in manifest  |  2.3.1+cu121 / 3.12.14  |  3  |  de0c21e  |  de0c21e  |  D08, D09, D10, D13, D14 |
| dovenet leak\_out s0  |  local workstation  |  2.3.1+cu121 / 3.12.14  |  3  |  91796ab  |  de0c21e  |  D01, D08 to D11, D13, D14 |
| dovenet leak\_in s1  |  local workstation  |  2.3.1+cu121 / 3.12.14  |  3  |  b5b1285  |  b5b1285  |  D02, D03, D08 to D11, D13 |
| dovenet leak\_out s1  |  local workstation  |  2.3.1+cu121 / 3.12.14  |  3  |  0eee1ff  |  b5b1285  |  D03, D08, D09, D13 |
| dovenet leak\_in s2  |  local workstation  |  2.3.1+cu121 / 3.12.14  |  3  |  de0c21e  |  de0c21e  |  D08, D09, D10, D13, D14 |
| dovenet leak\_out s2  |  local workstation  |  2.3.1+cu121 / 3.12.14  |  3  |  91796ab  |  de0c21e  |  D01, D08 to D11, D13, D14 |
| harmonizer leak\_in s0  |  local workstation (`.venvs/harmonizer-cuda`)  |  2.3.1+cu121 / 3.12.14  |  3  |  1479f5f  |  1479f5f  |  D08, D09, D10, D13, D14 |
| harmonizer leak\_out s0  |  local workstation  |  2.3.1+cu121 / 3.12.14  |  3  |  91796ab  |  91796ab  |  D04, D05, D08 to D11, D13, D14 |
| harmonizer leak\_in s1  |  laptop A, RTX 5050 Laptop, driver 596.08 (notes)  |  2.7.1+cu128 / 3.12.14  |  6  |  de0c21e  |  de0c21e  |  D08, D09, D12, D13, D14 |
| harmonizer leak\_out s1  |  laptop A  |  2.7.1+cu128 / 3.12.14  |  6  |  de0c21e  |  de0c21e  |  D06, D08, D09, D11 to D14 |
| harmonizer leak\_in s2  |  laptop B, RTX 3050 Laptop 4 GB (notes)  |  2.3.1+cu121 / 3.12.3  |  6  |  91796ab  |  91796ab  |  D08, D09, D12, D13, D14 |
| harmonizer leak\_out s2  |  laptop B  |  2.3.1+cu121 / 3.12.3  |  6  |  91796ab  |  91796ab  |  D07, D08, D09, D12 to D14 |

Constant across all twelve arms, read from every manifest: the evaluation and training list hashes
equal the registered ones, the confirmatory epoch equals the final epoch, and the scored row counts
are 5,581 and 1,874. The `is_pre_registered` flag is a driver-side flag; it is not evidence that the
departures did not occur.
