# Version Management

This project uses local Git version control.

## Version files

- `VERSION` stores the current project version.
- `CHANGELOG.md` records notable changes.
- `.gitignore` excludes local secrets and runtime artifacts.

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
git tag v0.1.0
```

Before committing, verify no secrets are staged:

```powershell
git diff --cached --name-only
git diff --cached
```
