# syntax=docker/dockerfile:1
#
# Recall API server image.
#
#   docker build -f deploy/Dockerfile -t recall-server .
#   docker run -p 8080:8080 \
#     -v recall-config:/etc/recall -v recall-data:/var/lib/recall \
#     recall-server
#
# The build is pure Go (CGO_ENABLED=0): no C toolchain is needed and the
# resulting binary is fully static, so it runs on any distroless/static
# base image.

FROM golang:1.26 AS build
WORKDIR /src

# Cache module downloads separately from source changes.
COPY go.mod go.sum ./
RUN go mod download

COPY . .
ARG TARGETOS
ARG TARGETARCH
RUN CGO_ENABLED=0 GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH:-amd64} \
    go build -trimpath -ldflags="-s -w" \
    -o /out/recall-server ./cmd/recall-server

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/recall-server /usr/local/bin/recall-server

# Mount /etc/recall for the config file and /var/lib/recall for the
# SQLite database (when using the sqlite backend).
EXPOSE 8080
USER nonroot:nonroot

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD ["/usr/local/bin/recall-server", "-health-probe", "-probe-url", "http://127.0.0.1:8080/healthz"]

ENTRYPOINT ["/usr/local/bin/recall-server"]
CMD ["-config", "/etc/recall/recall.yaml"]
