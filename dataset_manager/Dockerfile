FROM python:3.9-slim AS builder
WORKDIR /app
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y curl && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY dataset_manager/ /app/
# RUN apt-get install -y curl && apt-get clean
RUN apt-get update && apt-get install -y tree curl && apt-get clean

USER root

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

CMD python dataset_manager_script.py && python dataset_manager_api.py