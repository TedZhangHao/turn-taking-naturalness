# Turn-Taking Naturalness Benchmark

This directory is reserved for the released TurnNat Perturbation Benchmark.

Figshare artifact package:

`https://figshare.com/s/65e5bf5290085220ee88`

Expected layout after downloading:

```text
data/benchmark/
  manifests/test2.csv
  early_entry/
  late_response/
  shift_instead_of_hold/
  hold_instead_of_shift/
  excessive_backchannel/
```

The benchmark contains 1,000 paired natural/perturbed examples: five timing
perturbation types with 200 pairs each.

- `early_entry`
- `late_response`
- `shift_instead_of_hold`
- `hold_instead_of_shift`
- `excessive_backchannel`

Use `data/benchmark/manifests/test2.csv` as the default evaluation manifest when
running the released TurnNat FVAD checkpoint from the same artifact package.
