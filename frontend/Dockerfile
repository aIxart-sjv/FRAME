# ==============================================================================
# FRAME Frontend Dockerfile
# Multi-stage build: Vite compilation + Nginx Alpine static serving & proxy
# ==============================================================================

# Stage 1: Build static assets
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency specifications
COPY package.json package-lock.json ./

# Install dependencies cleanly
RUN npm ci

# Copy source code
COPY . .

# Build argument for backend API URL (defaults to http://localhost:8000 for browser client)
ARG VITE_FRAME_API_URL="http://localhost:8000"
ENV VITE_FRAME_API_URL=${VITE_FRAME_API_URL}

# Compile production bundle to /app/dist
RUN npm run build

# Stage 2: Production web server
FROM nginx:alpine

# Remove default nginx html
RUN rm -rf /usr/share/nginx/html/*

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
