# Часовой Dukascopy XAU/USD Bid, день до начала периода для первой полной сессии.
# Запускать из корня репозитория. Нужен Node.js и npx.
npx --yes dukascopy-node@1.51.1 -i xauusd -from 2025-10-27 -to 2026-10-28 -t h1 -p bid -v -f csv -utc 0 -dir data/raw
if ($LASTEXITCODE -ne 0) { throw 'Не удалось скачать котировки Dukascopy.' }
