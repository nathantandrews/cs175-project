# cs175-project

# To Run the Project

Download Docker Desktop: 

`https://www.docker.com/products/docker-desktop/`

After downloading docker desktop create a Docker image using the command below:

`docker build -t cs175 .`

Run the docker image to create a container

`docker run  -d --name model -v .:/model cs175 sleep infinity`

Execute the container 

`docker exec -it model bash ` or `docker exec -it model python main.py`