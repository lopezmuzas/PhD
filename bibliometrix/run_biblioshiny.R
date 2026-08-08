# Script de inicio rápido para Bibliometrix / Biblioshiny en R local
# No usa Docker (evita problemas de rendimiento en macOS ARM/Intel)

if (!requireNamespace("bibliometrix", quietly = TRUE)) {
  message("Instalando el paquete 'bibliometrix'...")
  install.packages("bibliometrix", dependencies = TRUE)
}

library(bibliometrix)

message("\n=======================================================")
message("  Iniciando Biblioshiny en tu navegador local...")
message("  Los datos .bib se encuentran en la carpeta 'bibliometrix/data/'")
message("=======================================================\n")

biblioshiny()
