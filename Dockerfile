# Multistage so dependency layers survive source edits.
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libasound2t64 libasound2-dev \
        build-essential gfortran pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src

# Dependency-only layers. Copying pyproject alone keeps these cached across source edits.
FROM base AS deps-full
COPY pyproject.toml README.md LICENSE ./
RUN python -m venv /venv && /venv/bin/pip install --no-cache-dir \
        black pylint pytest pytest-cov pytest-xdist import-linter \
    && /venv/bin/pip install --no-cache-dir \
        "numpy>=1.24" "scipy>=1.11" "opencv-python-headless>=4.9" \
        "pyyaml>=6.0" "pyarrow>=15.0" "librosa>=0.10" "soundfile>=0.12" "tqdm>=4.66" \
        "pyvmancer[serial]>=0.2"

# The seam check (design SS2.7): only the aesthetics extra, no device stack.
FROM base AS deps-aesthetics
COPY pyproject.toml README.md LICENSE ./
RUN python -m venv /venv && /venv/bin/pip install --no-cache-dir \
        pytest pytest-cov pytest-xdist \
    && /venv/bin/pip install --no-cache-dir \
        "numpy>=1.24" "scipy>=1.11" "opencv-python-headless>=4.9"

# The image the rig is driven from: ffmpeg for capture and mastering, ssh for the Pi.
FROM base AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg openssh-client libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE ./
RUN python -m venv /venv && /venv/bin/pip install --no-cache-dir \
        "numpy>=1.24" "scipy>=1.11" "opencv-python-headless>=4.9" \
        "pyyaml>=6.0" "pyarrow>=15.0" "librosa>=0.10" "soundfile>=0.12" "tqdm>=4.66" \
        "pyvmancer[all]>=0.2"
COPY . .
RUN /venv/bin/pip install --no-deps -e .
WORKDIR /work
ENTRYPOINT ["/venv/bin/syncsummoner"]

FROM deps-full AS lint
COPY . .
RUN /venv/bin/pip install --no-deps -e . \
    && /venv/bin/black --check . \
    && /venv/bin/pylint syncsummoner tests \
    && /venv/bin/lint-imports

FROM deps-full AS test
COPY . .
RUN /venv/bin/pip install --no-deps -e .
CMD ["/venv/bin/pytest"]

# Must pass with pyvmancer/pyyaml/pyarrow absent, or the seam has leaked.
FROM deps-aesthetics AS test-aesthetics
COPY . .
RUN /venv/bin/pip install --no-deps -e .
CMD ["/venv/bin/pytest", "tests/aesthetics", "-o", "addopts=", "-n", "auto", \
     "--cov=syncsummoner.aesthetics", "--cov-report=term-missing", "--cov-fail-under=85"]
