# ============================================================
# CT Simulator - Frontend Dockerfile
# ============================================================
# Multi-stage build: node deps → build → nginx serve
# ============================================================

# ---- Stage 1: Dependencies ----
FROM node:20-slim AS deps

ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY="*"
ENV HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY} NO_PROXY=${NO_PROXY}

WORKDIR /app

# Copy package.json only (lockfile is Windows-generated, skip it so
# npm resolves platform-specific optional deps for Linux)
COPY frontend/package.json ./

# Install dependencies (using npmmirror registry for faster downloads)
RUN npm config set registry https://registry.npmmirror.com && \
    npm install --timeout=300000

# ---- Stage 2: Build ----
FROM node:20-slim AS builder

ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY="*"
ENV HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY} NO_PROXY=${NO_PROXY}

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ .

# Build the application
RUN npm run build

# ---- Stage 3: Serve ----
FROM nginx:alpine AS runner

# Copy nginx configuration
COPY docker/frontend/nginx.conf /etc/nginx/nginx.conf

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:80/health || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
