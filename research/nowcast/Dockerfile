FROM python:3.12-slim AS wheels
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++
COPY research/nowcast/requirements.txt /requirements.txt
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r /requirements.txt

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1
COPY --from=wheels /wheels /wheels
COPY research/nowcast/requirements.txt /requirements.txt
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /requirements.txt
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /work
COPY research /work/research
USER 1000:1000
HEALTHCHECK --interval=5m --timeout=10s --start-period=10m --retries=2 CMD python -m research.nowcast.health --root /data
CMD ["python", "-m", "research.nowcast.live", "--root", "/data"]
