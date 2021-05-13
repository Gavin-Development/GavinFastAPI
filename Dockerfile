# set base image (host OS)
FROM python:3.8

RUN apt-get update && apt-get upgrade && apt-get install bash git

RUN git clone https://github.com/Scot-Survivor/GavinFastAPI.git --recursive
RUN cd GavinFastAPI && pip install -r requirements.txt
RUN cd GavinFastAPI && chmod +x start.sh
ADD models ./GavinFastAPI/models/.
EXPOSE 8000
WORKDIR "./GavinFastAPI"
ENTRYPOINT ["./start.sh"]
