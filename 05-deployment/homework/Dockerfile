FROM agrigorev/zoomcamp-model:2025
WORKDIR /code


RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml ./
COPY uv.lock ./

RUN uv sync --frozen

COPY predict.py .

EXPOSE 32000

ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--reload", "--host", "0.0.0.0", "--port", "32000"]