# DAU v2 (standalone simulation)

Numpy-only Active Inference triad (Designer, Smartphone, User) without the web dashboard.

## Dependencies

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m dau_v2.main --steps 200 --seed 0
```

Writes matplotlib figures to `results/` next to this package (see `plotting.py`), or see `README_DAU_V2.md` for full detail.

## Tests

```bash
python -m dau_v2.self_check
```
