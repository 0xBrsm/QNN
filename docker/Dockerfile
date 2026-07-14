FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

ARG USERNAME=trainer
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        fteqcc \
        git \
        jq \
        less \
        patch \
        procps \
        rsync \
        sudo \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -f video \
    && groupadd -f render \
    && if ! getent group "${USER_GID}" >/dev/null; then groupadd --gid "${USER_GID}" "${USERNAME}"; fi \
    && existing_user="" \
    && if getent passwd "${USER_UID}" >/dev/null; then existing_user="$(getent passwd "${USER_UID}" | cut -d: -f1)"; fi \
    && if [ -n "${existing_user}" ]; then \
         if [ "${existing_user}" != "${USERNAME}" ]; then usermod --login "${USERNAME}" --home "/home/${USERNAME}" --move-home "${existing_user}"; fi; \
       elif ! id -u "${USERNAME}" >/dev/null 2>&1; then \
         useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}"; \
       fi \
    && usermod -aG sudo,video,render "${USERNAME}" \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}" \
    && python3 -m pip install --no-cache-dir \
        "pytest>=8.0.0" \
        "smbprotocol" \
        "optuna"

WORKDIR /workspace

COPY src/docker/entrypoint.sh /usr/local/bin/qnn-trainer-entrypoint
RUN chmod +x /usr/local/bin/qnn-trainer-entrypoint

USER ${USERNAME}

ENTRYPOINT ["qnn-trainer-entrypoint"]
CMD ["bash"]
