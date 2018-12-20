FROM tensorflow/tensorflow:latest-gpu

WORKDIR /usr/src/app

RUN apt update; apt install python-tk -y
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "./nn.py" ]