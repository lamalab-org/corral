ARG BASE=corral-trace_audit_pilot:claude-langfuse
# Add the Claude Code CLI to a Corral image with the Python SDK dependencies.
FROM node:22-bookworm-slim AS claude-code
RUN npm install --global @anthropic-ai/claude-code@2.1.322

ARG BASE
FROM ${BASE}
USER root
COPY --from=claude-code /usr/local/bin/node /usr/local/bin/node
COPY --from=claude-code /usr/local/bin/claude /usr/local/bin/claude
COPY --from=claude-code /usr/local/lib/node_modules/@anthropic-ai/claude-code /usr/local/lib/node_modules/@anthropic-ai/claude-code
RUN claude --version
