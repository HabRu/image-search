FROM python:3.8
WORKDIR /usr/src/app
COPY requirenments.txt .
RUN pip install -r requirenments.txt
RUN pip install pillow
COPY . .
EXPOSE 80
CMD ["python", "./src/app.py"]