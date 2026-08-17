# Preregistered AudioMNIST external evaluation

The immutable protocol was committed before audio download, and the final FSDD
checkpoint was committed before any AudioMNIST inference. See
`AUDIOMNIST_EXTERNAL_PROTOCOL.md` and `AUDIOMNIST_EXTERNAL_REPORT.md`.

```powershell
uv run python -m external_evaluation.data
uv run python -m external_evaluation.final_fit
```

The first external score has already been published. The evaluator refuses to
overwrite it:

```powershell
uv run python -m external_evaluation.evaluate
```

Use the saved JSON for learning and analysis. AudioMNIST is now contacted data;
future adaptation is a separate experiment and cannot retain the untouched-test
claim.
