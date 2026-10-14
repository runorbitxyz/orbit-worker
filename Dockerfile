FROM python:3.13-alpine
COPY orbit_worker.py /orbit_worker.py
RUN mkdir -p /state && chown nobody /state
USER nobody
ENV ORBIT_HOME=/state
ENTRYPOINT ["python", "-u", "/orbit_worker.py"]
