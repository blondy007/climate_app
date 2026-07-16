#!/bin/sh
set -eu

mkdir -p /app/data

if [ ! -f /app/data/meteostat_master.csv ]; then
  cp /app/seed-data/meteostat_master.csv /app/data/meteostat_master.csv
fi

if [ ! -f /app/data/wind_thresholds.json ]; then
  cp /app/seed-data/wind_thresholds.json /app/data/wind_thresholds.json
fi

exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}" \
  --server.headless=true \
  --browser.gatherUsageStats=false
