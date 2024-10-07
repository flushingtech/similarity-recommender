FROM python 
COPY requirements.txt .
COPY similarity-recommender.py .
RUN pip3 install -r requirements.txt
CMD ["python","similarity-recommender.py"]