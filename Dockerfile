# 1. Start from a small Linux image that already has Python 3.12
FROM python:3.12-slim

# 2. Copy the uv program in from uv's official image (same version as on your machine)
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

# 3. All following commands run inside /app in the container
WORKDIR /app

# 4. Copy ONLY the dependency files first, then install packages
#    (cached: only re-runs when these files change)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen

# 5. Now copy your code
COPY . .

# 6. Put the .venv on PATH so "python" means the env's python
ENV PATH="/app/.venv/bin:$PATH"

# 7. Default command when the container starts
CMD ["python", "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"]
