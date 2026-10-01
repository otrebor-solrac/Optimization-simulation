FROM christmas-tree-optimizer:latest

USER root

# Install necessary libraries for X11 GUI and audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxkbcommon-dev \
    libx11-xcb-dev \
    libxi-dev \
    libgl1-mesa-dev \
    libasound2-dev \
    pkg-config \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV RUST_BACKTRACE=1

ENTRYPOINT []
CMD ["/bin/bash", "-c", "cd /app/ui_rust && cargo run --release"]
