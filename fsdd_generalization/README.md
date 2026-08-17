# FSDD speaker-disjoint generalization experiment

This experiment moves beyond the tiny overfitting capstone. It downloads a pinned FSDD archive, verifies SHA256 and the 3,000-file audio contract, then splits by speaker:

- train: george, jackson, lucas, nicolas (2,000 files);
- dev: theo (500 files);
- test: yweweler (500 files).

No speaker appears in more than one split. Feature statistics, augmentation, and gradient updates use only train. Epoch selection uses only dev. Test is evaluated after the best dev checkpoint has been fixed.

```powershell
uv run python -m fsdd_generalization.data
uv run python -m fsdd_generalization.run_experiment --epochs 24
```

Use `--skip-baseline` for a quicker augmented-only run. It writes separate
`*_augmented_only` artifacts so it cannot overwrite the published comparison.

The follow-up six-fold speaker analysis is:

```powershell
uv run python -m fsdd_generalization.loso
```

It rotates every FSDD speaker through outer test, uses one different speaker as
dev, trains two candidates without providing test recordings, and constructs
outer-test features only after dev-only selection. This is an exploratory
within-dataset resampling estimate—not a new untouched benchmark—because the
preceding lesson already used and inspected FSDD.

Each example concatenates 1–4 real clips from the same speaker with short silences, so CTC must learn genuine multi-token alignment, including repeated adjacent digits. The default run compares clean-only training with one deterministic gain-and-noise augmented copy per training sequence. It saves a manifest, metrics JSON, and the augmented streaming CTC checkpoint under `artifacts/`.

The downloaded WAV files stay under `.local_data/` and are intentionally ignored by Git. FSDD is CC BY-SA 4.0; keep `DATA_SOURCES.md` and upstream attribution when redistributing derived artifacts.
