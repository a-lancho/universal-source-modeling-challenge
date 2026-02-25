# Public Release Checklist (GO/NO-GO)

Use this checklist before pushing a public GitHub release of the Universal Source Modeling Challenge repo.

Scope:
- Public-safe documentation and code release only
- No live-day secret artifacts
- No accidental caches or local build/runtime junk

## GO / NO-GO Checklist

Mark **GO** only if all items pass:

- `GO` No `data/live_release/` artifacts are present
- `GO` No hidden/private test split is included (especially no `data/generator/test.npy`)
- `GO` Public practice data only: `data/public_practice/test.npy` is okay
- `GO` `.gitignore` covers caches and live artifacts (`__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/`, `data/live_release/`)
- `GO` No accidental local caches (`__pycache__`, `.pytest_cache`, `.ipynb_checkpoints`) are present
- `GO` Docs do not reveal a concrete live-day seed (use placeholders like `<SECRET_SEED>`)
- `GO` No extra student submissions / local scratch files / bundled zips are included
- `GO` `git status` is clean (or only intended doc changes are staged)

If any item fails: **NO-GO** until fixed.

## Exact Commands To Run Before Pushing

Run from the repo root:

```bash
pwd
git status --short
```

Inventory and data audit:

```bash
find . -maxdepth 2 -mindepth 1 | sort
find data -type f -printf '%p\t%s bytes\n' | sort
find . -type f -name '*.npy' -printf '%p\t%s bytes\n' | sort
```

Risk checks (live artifacts, caches, hidden test split):

```bash
test -f data/generator/test.npy; echo "data/generator/test.npy exists? exit=$?"
find . \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.ipynb_checkpoints' \) -print | sort
find data -type f | rg 'live_release'
```

Tracked-file sanity checks:

```bash
git ls-files | rg '^data/live_release/'
git ls-files | rg '^data/generator/test\.npy$'
git ls-files | rg '\.npy$'
```

Docs sanity checks (seed placeholders / secret wording):

```bash
rg -n "release_test .*--seed " competition/README.md competition/LIVE_DAY_CHECKLIST.md competition/PACKAGING.md
rg -n "SECRET_SEED|secret" competition/*.md OFFICIAL_CONFIG.md
```

Optional packaging sanity check (recommended if student bundle tooling changed):

```bash
python -m competition.live_day_quick_check --check-bundle
python -m competition.make_student_bundle --output-dir /tmp/student_bundle_final --overwrite
cd /tmp && zip -r student_bundle_final.zip student_bundle_final
```

## If All PASS, Push With:

Review final diff:

```bash
git status
git diff --stat
```

Commit and push:

```bash
git add -A
git commit -m "Prepare public release"
git push origin <branch-name>
```

If pushing a tagged release:

```bash
git tag -a vX.Y.Z -m "Public release vX.Y.Z"
git push origin <branch-name> --tags
```
