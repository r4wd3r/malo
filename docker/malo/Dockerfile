# MaLO CTF Environment - Kali Linux Base
# This Dockerfile creates a comprehensive CTF environment based on Kali Linux
# Kali Linux already includes many security tools, so we only install additional ones
FROM kalilinux/kali-rolling

ENV DEBIAN_FRONTEND=noninteractive

# Only add i386 architecture on x86_64 platforms
RUN if [ "$(dpkg --print-architecture)" = "amd64" ]; then dpkg --add-architecture i386; fi

# Update package lists and install additional tools not included in Kali base
# Note: Some packages have different names in Kali Linux (e.g., libtbb12 instead of libtbb2, openjdk-11 instead of openjdk-17)
RUN apt-get update && apt-get install -y \
    # Basic utilities (most are already in Kali)
    curl netcat-traditional wget jq vim \
    # Build essentials
    build-essential pkg-config \
    # Debugging tools
    gdb gdbserver \
    # Python development
    python3-dev python3-pip python3-venv \
    # Libraries
    libssl-dev libffi-dev \
    libtbb12 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
    # File analysis
    bsdmainutils file \
    # Network analysis tools
    tcpdump wireshark-common tshark\
    # Certificate tools
    p11-kit p11-kit-modules \
    # Rust for modern CTF challenges
    rustc cargo \
    # Forensics
    sleuthkit \
    # Java for Ghidra and Android tools
    openjdk-11-jdk openjdk-11-jre \
    # Zip tools
    unzip \
    # Sudo for privileged operations
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install i386 libraries only on x86_64 (for 32-bit binary challenges)
RUN if [ "$(dpkg --print-architecture)" = "amd64" ]; then \
    apt-get update && apt-get install -y \
    libc6-i386 libc6-dev:i386 gcc-multilib g++-multilib \
    libtbb12:i386 \
    && rm -rf /var/lib/apt/lists/*; \
    fi

# Install radare2 (architecture-aware) - Kali already has radare2, but we'll ensure latest version
WORKDIR /tmp
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
        curl -LO https://github.com/radareorg/radare2/releases/download/5.8.8/radare2_5.8.8_amd64.deb && \
        curl -LO https://github.com/radareorg/radare2/releases/download/5.8.8/radare2-dev_5.8.8_amd64.deb && \
        apt-get install -y ./radare2_5.8.8_amd64.deb ./radare2-dev_5.8.8_amd64.deb && \
        rm -f ./radare2-dev_5.8.8_amd64.deb ./radare2_5.8.8_amd64.deb; \
    else \
        apt-get update && apt-get install -y radare2 && rm -rf /var/lib/apt/lists/*; \
    fi

# Install Ghidra (architecture-aware)
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ] || [ "$ARCH" = "arm64" ]; then \
        wget https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.0.1_build/ghidra_11.0.1_PUBLIC_20240130.zip && \
        unzip -d /opt/ghidra ghidra_11.0.1_PUBLIC_20240130.zip && \
        rm ghidra_11.0.1_PUBLIC_20240130.zip && \
        mkdir -p /opt/ghidra/customScripts/; \
    fi

# Install jadx for Android reverse engineering
RUN curl -LO https://github.com/skylot/jadx/releases/download/v1.4.7/jadx-1.4.7.zip && \
    unzip -d /usr/local jadx-1.4.7.zip && \
    rm -f jadx-1.4.7.zip

# Install Docker CE for container communication (Kali Linux compatible method)
# Using docker-ce from Docker's official repository for latest features
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce docker-ce-cli containerd.io && \
    rm -rf /var/lib/apt/lists/*

# Enable Docker service (as recommended in Kali documentation)
RUN systemctl enable docker

# Install pipx for better Python package management (recommended by Kali)
RUN apt-get update && apt-get install -y pipx && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment for CTF Python packages
RUN python3 -m venv /opt/ctf-venv && \
    /opt/ctf-venv/bin/pip install --upgrade pip && \
    /opt/ctf-venv/bin/pip install pwntools gmpy2 angr chepy scapy volatility3

# Install dwarf2json for Volatility3 (requires Go)
RUN apt-get update && apt-get install -y golang-go git && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://github.com/volatilityfoundation/dwarf2json.git /tmp/dwarf2json && \
    cd /tmp/dwarf2json && \
    go build -o /usr/local/bin/dwarf2json && \
    chmod +x /usr/local/bin/dwarf2json && \
    rm -rf /tmp/dwarf2json

# Make the virtual environment easily accessible
RUN echo '#!/bin/bash\nsource /opt/ctf-venv/bin/activate\nexec "$@"' > /usr/local/bin/ctf-python && \
    chmod +x /usr/local/bin/ctf-python


# Create a symbolic link for python to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create a non-root user WITH sudo permissions for privileged operations
ARG USERNAME=ctfplayer
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Set up Python environment globally and for the user
ENV PATH="/opt/ctf-venv/bin:$PATH"
RUN echo 'export PATH="/opt/ctf-venv/bin:$PATH"' >> /home/$USERNAME/.bashrc && \
    echo 'alias python="/opt/ctf-venv/bin/python"' >> /home/$USERNAME/.bashrc && \
    echo 'alias pip="/opt/ctf-venv/bin/pip"' >> /home/$USERNAME/.bashrc && \
    echo 'source /opt/ctf-venv/bin/activate' >> /home/$USERNAME/.bashrc

# Create working directory
RUN mkdir -p /tmp/ctf && chown $USERNAME:$USERNAME /tmp/ctf

# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Keep container running
CMD ["/bin/bash", "-c", "while true; do sleep 3600; done"]
