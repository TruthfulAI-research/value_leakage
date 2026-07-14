# Rater image: Claude Code CLI + claude-agent-sdk on Python 3.12.
#
# Built once by run.py (`docker build`). On each rated trial a fresh
# container is spawned with the trial CSV mounted at /work; the
# container runs rate.py to pick the best answer per question and
# writes /work/best_answers.csv back into the mount, then exits.
#
# Needs ANTHROPIC_API_KEY at runtime (passed via `docker run -e`).
#
# rate.py is INTENTIONALLY NOT baked into the image. It's bind-mounted
# read-only at /opt/rater/rate.py by run.py:_rate_one each invocation.
# This keeps the image rate-agnostic, so per-project differences in the
# rating prompt live entirely in each project's `rate.py` (which is
# also snapshotted into the run dir for provenance — see
# run.py:_snapshot_config).

FROM python:3.12-slim

# Node.js 20 is needed by the @anthropic-ai/claude-code npm package,
# which claude-agent-sdk spawns as a subprocess.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI (the binary claude-agent-sdk drives).
RUN npm install -g @anthropic-ai/claude-code

# Python SDK + minimal runtime deps.
RUN pip install --no-cache-dir \
    "claude-agent-sdk>=0.1.81,<0.2"

# Pre-create the bind-mount target so Docker doesn't have to guess
# whether /opt/rater/rate.py should be a file or a directory.
RUN mkdir -p /opt/rater && touch /opt/rater/rate.py

# When `docker run` mounts a CSV at /work, this is where rate.py looks.
WORKDIR /work
ENTRYPOINT ["python", "/opt/rater/rate.py"]
