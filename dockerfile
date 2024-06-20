FROM nvcr.io/nvidia/pytorch:21.09-py3

LABEL maintainer="miras_zakaryanov"

ENV TZ=Asia/Almaty \
    USER=miras_zakaryanov \
    UID=1010


RUN groupadd -g ${UID} ${USER} && useradd -l -r -m -s /bin/bash -u ${UID} ${USER} -g ${USER}

#Install your required software
RUN apt update -y && \
    apt -y install ca-certificates tzdata software-properties-common cmake

USER ${USER}

WORKDIR /home/user_name/
COPY . .

RUN python3 -m venv venv && source venv/bin/activate

RUN python3 -m pip install --upgrade pip
RUN pip3 install -U -r requirements.txt