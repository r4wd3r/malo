FROM kalilinux/kali-rolling

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    gdb \
    radare2 \
    ltrace \
    strace \
    binwalk \
    checksec \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-pwntools \
    python3-capstone \
    python3-unicorn

# Install Poetry
RUN pip3 install poetry

# Create a non-root user with sudo permissions
ARG USERNAME=malo-user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Configure Poetry to not create virtual environments in Docker
RUN poetry config virtualenvs.create false

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies with Poetry
RUN poetry install --no-dev --no-interaction --no-ansi

# Switch to user
USER $USERNAME
WORKDIR /home/$USERNAME

# Install additional CTF tools
RUN pip install --break-system-packages \
    angr \
    ropper \
    keystone-engine \
    pwntools \
    ipython \
    gmpy2

WORKDIR /ctf
VOLUME ["/ctf/challenges"]

CMD ["/bin/bash"]