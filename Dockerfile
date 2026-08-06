FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
COPY eval/ eval/
RUN pip install --no-cache-dir -e .
COPY results/ results/
CMD ["make", "gate"]
