FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

ARG USERNAME=trainer
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        jq \
        less \
        procps \
        rsync \
        sudo \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -f video \
    && groupadd -f render \
    && if ! getent group "${USER_GID}" >/dev/null; then groupadd --gid "${USER_GID}" "${USERNAME}"; fi \
    && if ! id -u "${USERNAME}" >/dev/null 2>&1; then useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}"; fi \
    && usermod -aG sudo,video,render "${USERNAME}" \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}" \
    && python3 -m pip install --no-cache-dir "numpy>=2.0.0" "pytest>=8.0.0"

WORKDIR /workspace

COPY docker/training/amd-rocm/entrypoint.sh /usr/local/bin/quake-ai-trainer-entrypoint
RUN chmod +x /usr/local/bin/quake-ai-trainer-entrypoint

USER ${USERNAME}

ENTRYPOINT ["quake-ai-trainer-entrypoint"]
CMD ["bash"]
