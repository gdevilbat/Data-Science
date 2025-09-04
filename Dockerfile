FROM python:3.10
# Define build arguments
ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
ARG PW=docker

# Define environment variables
ENV USER_ID=$USER_ID
ENV GROUP_ID=$GROUP_ID
ENV USER_NAME=$USER_NAME

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd -m ${USER_NAME} --uid=${USER_ID} && echo "${USER_NAME}:${PW}" | \
      chpasswd

WORKDIR /code
USER ${USER_ID}:${GROUP_ID}

# Pastikan HOME benar
ENV HOME=/home/${USER_NAME}
ENV PATH="${HOME}/.local/bin:${PATH}"

COPY requirements.txt /code/
RUN python -m pip install --user --upgrade "pip<25.3"
RUN pip install --user --default-timeout=1000 -r requirements.txt
COPY . /code/