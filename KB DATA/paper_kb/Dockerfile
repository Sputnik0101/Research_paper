FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run ingestion (mount /data for PDFs)
# docker build -t paper_kb .
# docker run --rm -v $(pwd)/pdfs:/app/pdfs -v $(pwd)/storage:/app/storage paper_kb python ingest.py --pdf_dir ./pdfs
