FROM python:3.11.4-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y curl wget

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --no-install-project
ENV PATH="/app/.venv/bin/:$PATH"

COPY src src
COPY config config
COPY scripts scripts
RUN chmod +x scripts/entrypoint.sh

ENV FLASK_APP=/app/src/app.py
ENV PYTHONPATH=/app

CMD ["bash", "scripts/entrypoint.sh"]