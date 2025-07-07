FROM public.ecr.aws/lambda/python:3.13

COPY main.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY .aws ${LAMBDA_TASK_ROOT}

RUN python3.13 -m pip install -r requirements.txt -t .
RUN python3.13 -m playwright install
RUN python3.13 -m playwright install-deps

CMD ["main.lambda_handler"]
