# Multistage so dependency layers survive source edits.
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
# librosa jits with cache=True; numba has nowhere to write it when the container
# runs as a user who owns neither /venv nor the mounted home, so name a path
# every uid can write.
ENV NUMBA_CACHE_DIR=/tmp/numba-cache
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libasound2t64 libasound2-dev \
        build-essential gfortran pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src

# The Pi's player, cross-built as one static aarch64 binary the host pushes on every start.
FROM debian:trixie-slim AS kmsplay-build
RUN dpkg --add-architecture arm64 && apt-get update && apt-get install -y --no-install-recommends \
        cmake make pkg-config g++-aarch64-linux-gnu libdrm-dev:arm64 nlohmann-json3-dev libcli11-dev \
    && rm -rf /var/lib/apt/lists/*
COPY native/kmsplay /src/kmsplay
RUN cmake -S /src/kmsplay -B /build -DCMAKE_TOOLCHAIN_FILE=/src/kmsplay/cmake/aarch64.cmake \
        -DCMAKE_BUILD_TYPE=Release -DKMSPLAY_STATIC=ON \
    && cmake --build /build -j "$(nproc)" \
    && aarch64-linux-gnu-strip -o /kmsplay /build/kmsplay

# The player's own tests, built natively with coverage; formatting is checked first.
FROM debian:trixie-slim AS kmsplay-test
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake make pkg-config g++ libdrm-dev nlohmann-json3-dev libcli11-dev libgtest-dev libgmock-dev \
        gcovr clang-format ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY native/kmsplay /src/kmsplay
RUN clang-format --dry-run -Werror /src/kmsplay/src/* /src/kmsplay/tests/* \
    && cmake -S /src/kmsplay -B /build -DKMSPLAY_TESTS=ON -DKMSPLAY_COVERAGE=ON \
    && cmake --build /build -j "$(nproc)"
WORKDIR /build
CMD ["sh", "-c", "ctest --output-on-failure -j \"$(nproc)\" && gcovr -r /src/kmsplay --filter /src/kmsplay/src/ --exclude /src/kmsplay/src/main.cpp --exclude-throw-branches --fail-under-line 85 /build"]

# Yosys, GHDL with its Yosys plugin, Project IceStorm and Icarus, as one pinned upstream build.
FROM debian:trixie-slim AS hdl-tools
ARG OSS_CAD_SUITE=2026-11-01
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL "https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${OSS_CAD_SUITE}/oss-cad-suite-linux-x64-$(echo ${OSS_CAD_SUITE} | tr -d -).tgz" \
        | tar -xz -C /opt \
    && rm -rf /var/lib/apt/lists/*

# The SDK core RTL and the community program sources the liveness analysis elaborates.
FROM debian:trixie-slim AS upstream
ARG VIDEOMANCER_SDK=606b6686caf0a211840f453019f4eb9911b545ca
ARG VIDEOMANCER_COMMUNITY=a51fa544b89e2ed09a6169e9f6abf5549e8e885b
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
RUN for spec in "videomancer-sdk ${VIDEOMANCER_SDK}" "videomancer-community-programs ${VIDEOMANCER_COMMUNITY}"; do \
        set -- $spec; git init -q /opt/upstream/$1 \
        && git -C /opt/upstream/$1 fetch -q --depth 1 https://github.com/lzxindustries/$1 $2 \
        && git -C /opt/upstream/$1 checkout -q FETCH_HEAD && rm -rf /opt/upstream/$1/.git; \
    done

# dd's Cython bindings to CUDD, built against the CUDD release dd pins and verifies.
FROM base AS bdd
ARG DD=0.6.0
RUN pip install --no-cache-dir cython setuptools wheel \
    && DD_FETCH=1 DD_CUDD=1 pip wheel --no-cache-dir --no-deps --no-binary dd --no-build-isolation \
        "dd==${DD}" -w /wheels

FROM base AS hdl
RUN apt-get update && apt-get install -y --no-install-recommends zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=bdd /wheels /wheels
COPY --from=hdl-tools /opt/oss-cad-suite /opt/oss-cad-suite
COPY --from=upstream /opt/upstream /opt/upstream
ENV SYNCSUMMONER_HDL_BIN=/opt/oss-cad-suite/bin \
    SYNCSUMMONER_SDK=/opt/upstream/videomancer-sdk \
    SYNCSUMMONER_COMMUNITY=/opt/upstream/videomancer-community-programs

# Dependency-only layers. Copying pyproject alone keeps these cached across source edits.
FROM hdl AS deps-full
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE ./
RUN python -m venv /venv && /venv/bin/pip install --no-cache-dir \
        black pylint pytest pytest-cov pytest-xdist pytest-timeout import-linter \
    && /venv/bin/pip install --no-cache-dir \
        "numpy>=1.24" "scipy>=1.11" "opencv-python-headless>=4.9" \
        "pyyaml>=6.0" "pyarrow>=15.0" "librosa>=0.10" "soundfile>=0.12" "tqdm>=4.66" "threadpoolctl>=3.1" "linuxpy>=0.25" \
        "pyvmancer[serial]>=0.6.5" "python-sat>=1.8.dev16" "scikit-learn>=1.3" /wheels/dd-*.whl

# The seam check (design SS2.7): only the aesthetics extra, no device stack.
FROM base AS deps-aesthetics
COPY pyproject.toml README.md LICENSE ./
RUN python -m venv /venv && /venv/bin/pip install --no-cache-dir \
        pytest pytest-cov pytest-xdist pytest-timeout \
    && /venv/bin/pip install --no-cache-dir \
        "numpy>=1.24" "scipy>=1.11" "opencv-python-headless>=4.9"

# The image the rig is driven from: ffmpeg for capture and mastering, ssh for the Pi.
FROM hdl AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg openssh-client libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=kmsplay-build /kmsplay /opt/syncsummoner/kmsplay
ENV SYNCSUMMONER_KMSPLAY=/opt/syncsummoner/kmsplay
COPY pyproject.toml README.md LICENSE ./
RUN python -m venv /venv && /venv/bin/pip install --no-cache-dir \
        "numpy>=1.24" "scipy>=1.11" "opencv-python-headless>=4.9" \
        "pyyaml>=6.0" "pyarrow>=15.0" "librosa>=0.10" "soundfile>=0.12" "tqdm>=4.66" "threadpoolctl>=3.1" "linuxpy>=0.25" \
        "pyvmancer[all]>=0.6.5" "python-sat>=1.8.dev16" "scikit-learn>=1.3" /wheels/dd-*.whl
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
