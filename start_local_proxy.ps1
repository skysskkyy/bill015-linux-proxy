Set-Location -LiteralPath "S:\hack\packyapi.com\bill015_local_proxy"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8787
