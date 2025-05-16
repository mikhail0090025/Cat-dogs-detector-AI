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
COPY neural_net/neural_net/ /app/
COPY neural_net/manage.py /app/
RUN apt-get install -y curl && apt-get clean

USER root

ENV PORT=8001

EXPOSE $PORT
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/ || exit 1

CMD python manage.py runserver $PORT