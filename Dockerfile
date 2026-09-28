FROM python:3.11-slim
WORKDIR /workspace
COPY . .
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["python", "scripts/run_experiment.py"]
CMD ["--config", "configs/uci.yaml"]
