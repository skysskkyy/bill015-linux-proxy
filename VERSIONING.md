# Version Management

This project uses local Git version control.

## Version files

- `VERSION` stores the current project version.
- `CHANGELOG.md` records notable changes.
- `.gitignore` excludes local secrets and runtime artifacts.
- FastAPI reads `VERSION` at startup, so update `VERSION` before release commits.

Current version:

```text
0.2.2
```

## Required workflow for every modification

Every code/config/documentation modification must be recorded in local Git before the task is considered done:

```powershell
git status --short
python scripts/selftest.py
git add <changed safe files>
git diff --cached --name-only
git diff --cached
git commit -m "type: concise summary"
```

If a remote is configured, push after committing:

```powershell
git remote -v
git push
```

If no remote is configured, the local commit is the required upload point for this workspace.

## Do not commit

The following are intentionally ignored:

- `config.local.json` and other `*.local.json` files
- `.env*`
- `proxy_evidence/`
- `proxy.pid`
- `__pycache__/`
- `*.bak*`

## Common commands

```powershell
git status
git add <files>
git commit -m "message"
git tag v0.2.2
```

Before committing, verify no secrets are staged:

```powershell
git diff --cached --name-only
git diff --cached
```
