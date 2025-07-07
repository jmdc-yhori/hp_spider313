# reference: https://qiita.com/moritalous/items/263a5d14faad621daea6

ARG FUNCTION_DIR="/function"
FROM python:3.13-slim
ARG FUNCTION_DIR
ENV PLAYWRIGHT_BROWSERS_PATH=/playwright

RUN mkdir -p ${FUNCTION_DIR}


COPY requirements.txt ${FUNCTION_DIR}
COPY .aws ${FUNCTION_DIR}

WORKDIR ${FUNCTION_DIR}

RUN apt-get update && apt-get install -y \
    curl \
    libnss3 \
    libgbm-dev \
    libgtk-3-0 \
    libasound2 \
    libx11-xcb1 \
    libxcomposite-dev \
    libxdamage-dev \
    libxrandr2 \
    libgbm-dev \
    libpango-1.0-0 \
    libcups2 \
    libdbus-glib-1-2 \
    libdrm2 \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p ${FUNCTION_DIR}/.aws-lambda-rie && curl -Lo ${FUNCTION_DIR}/.aws-lambda-rie/aws-lambda-rie \
https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie \
&& chmod +x ${FUNCTION_DIR}/.aws-lambda-rie/aws-lambda-rie  

RUN pip install -r requirements.txt
RUN playwright install --with-deps chromium

COPY main.py ${FUNCTION_DIR}

ENTRYPOINT ["/function/.aws-lambda-rie/aws-lambda-rie", "/usr/local/bin/python", "-m", "awslambdaric"]
CMD ["main.lambda_handler"]
