FROM --platform=linux/amd64 rocker/shiny:4.3.2

# Instalar dependencias del sistema necesarias para bibliometrix
RUN apt-get update && apt-get install -y \
  libcurl4-openssl-dev \
  libssl-dev \
  libxml2-dev \
  libfontconfig1-dev \
  libharfbuzz-dev \
  libfribidi-dev \
  libfreetype6-dev \
  libpng-dev \
  libtiff5-dev \
  libjpeg-dev \
  libgit2-dev \
  pandoc \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Instalar paquetes de R
RUN R -e "install.packages(c( \
  'bibliometrix', \
  'shiny', \
  'shinydashboard', \
  'DT', \
  'ggplot2', \
  'plotly', \
  'dplyr' \
  ), repos='https://packagemanager.posit.co/cran/__linux__/jammy/latest', dependencies=TRUE)"

# Copiar la aplicación Shiny
COPY shiny-app/ /srv/shiny-server/bibliometrix/

# Permisos correctos
RUN chown -R shiny:shiny /srv/shiny-server/

EXPOSE 3838
