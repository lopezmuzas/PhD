---
title: "🌊 Ocean Enterprise Node — Guía de Instalación Completa"
tags: [guia]
status: borrador
updated: 2026-08-08
---

# 🌊 Ocean Enterprise Node — Guía de Instalación Completa

> **Entorno:** Ubuntu 24.04 LTS · Docker · Sepolia Testnet  
> **Stack:** OE Node + Elasticsearch + Marketplace + Nginx + UFW  
> **Modo:** Sin SSI (modo simple)

---

## Índice

1. [Prerequisitos y preparación](#1-prerequisitos-y-preparación)
2. [Limpieza Docker previa](#2-limpieza-docker-previa)
3. [Proveedor RPC — Alchemy](#3-proveedor-rpc--alchemy)
4. [Wallet del nodo — MetaMask](#4-wallet-del-nodo--metamask)
5. [Instalación del OE Node](#5-instalación-del-oe-node)
6. [Instalación del Marketplace](#6-instalación-del-marketplace)
7. [Nginx — Reverse Proxy](#7-nginx--reverse-proxy)
8. [Firewall — UFW](#8-firewall--ufw)
9. [Verificación final](#9-verificación-final)
10. [Mantenimiento y comandos útiles](#10-mantenimiento-y-comandos-útiles)
11. [Futuro: Conectar con Pontus-X](#11-futuro-conectar-con-pontus-x)

---

## 1. Prerequisitos y preparación

### Hardware mínimo recomendado

| Recurso | Mínimo |
|---|---|
| CPU | 2 cores |
| RAM | 16 GB |
| Disco libre | 20 GB |

### Software requerido

```bash
# Verificar Docker instalado
docker --version
docker compose version

# Verificar sistema operativo
uname -a
```

### Cuentas externas necesarias

- **Alchemy** (RPC blockchain gratuito) → [alchemy.com](https://alchemy.com)
- **MetaMask** (wallet para el nodo) → [metamask.io](https://metamask.io)

---

## 2. Limpieza Docker previa

Si el servidor tenía contenedores anteriores, limpiar todo antes de empezar:

```bash
# Parar todos los contenedores
docker stop $(docker ps -q)

# Eliminar todo de una vez (contenedores, imágenes, volúmenes, redes, caché)
docker system prune -af --volumes
```

---

## 3. Proveedor RPC — Alchemy

1. Crear cuenta gratuita en [alchemy.com](https://alchemy.com)
2. Crear nueva app:
   - **Chain:** Ethereum
   - **Network:** Ethereum Sepolia
3. En el dashboard de la app, ir a **"Node RPC Setup"**
4. Cambiar el desplegable **Network** a **Ethereum Sepolia**
5. Copiar la **Endpoint URL**:

```
https://eth-sepolia.g.alchemy.com/v2/TU_API_KEY
```

> ⚠️ Guardar esta URL en un lugar seguro. No compartirla públicamente.

---

## 4. Wallet del nodo — MetaMask

Crear una wallet **nueva y dedicada exclusivamente** para el nodo (nunca usar una con fondos reales).

### Crear cuenta nueva en MetaMask

1. Abrir MetaMask en el navegador
2. Clic en el círculo de cuenta (arriba a la derecha)
3. **"Add account or hardware wallet"** → **"Add a new account"**
4. Nombre: `ocean-node`

### Exportar la clave privada

1. Con la cuenta `ocean-node` seleccionada
2. Clic en los tres puntos (⋮) → **"Account details"**
3. **"Show private key"** → introducir contraseña de MetaMask
4. Copiar la clave privada

> ⚠️ La clave privada se usa en el `.env.node` con el prefijo `0x`.  
> **Nunca compartir la clave privada.**

### Dirección pública

La dirección pública (para `ALLOWED_ADMINS`) es el texto `0x...` visible en la parte superior de MetaMask, mucho más corto que la clave privada.

---

## 5. Instalación del OE Node

### 5.1 Clonar el repositorio

```bash
cd /var/www
git clone https://github.com/OceanProtocolEnterprise/ocean-node.git
cd /var/www/ocean-node/ocean-node
```

### 5.2 Configurar Elasticsearch

```bash
cp .env.elasticsearch.example .env.elasticsearch
nano .env.elasticsearch
```

Contenido del `.env.elasticsearch`:

```env
ELASTIC_PASSWORD=TU_PASSWORD_SEGURO
```

### 5.3 Configurar el nodo

```bash
cp .env.node.example .env.node
nano .env.node
```

Contenido del `.env.node`:

```env
# REQUERIDO
PRIVATE_KEY=0xTU_CLAVE_PRIVADA

# BLOCKCHAIN
RPCS={"11155111":{"rpc":"https://eth-sepolia.g.alchemy.com/v2/TU_API_KEY","chainId":11155111,"network":"sepolia","chunkSize":50,"startBlock":10347788}}

# ELASTICSEARCH
DB_URL=http://elasticsearch:9200
DB_TYPE=elasticsearch
DB_USERNAME=elastic
DB_PASSWORD=TU_PASSWORD_SEGURO

# IPFS
IPFS_GATEWAY=https://ipfs.io/
ARWEAVE_GATEWAY=https://arweave.net/

# NODE FEE
FEE_TOKENS={"11155111":""}
FEE_AMOUNT={"amount":1,"unit":"MB"}

# HTTP
HTTP_API_PORT=8000

# POLICY SERVER (vacío = sin SSI)
POLICY_SERVER_URL=

# NODE OWNER INFO
NODE_OWNER_INFO={"imprint":{"legalName":"Test Node","email":"test@test.com","url":"http://localhost","address":"Test Address"},"termsAndConditions":{"url":"http://localhost"},"privacyPolicy":{"url":"http://localhost"}}

# ADMIN
ALLOWED_ADMINS=["0xTU_DIRECCION_PUBLICA_METAMASK"]

# OTROS
DASHBOARD=true
MAX_REQ_PER_MINUTE=10000
MAX_CONNECTIONS_PER_MINUTE=10000
```

### 5.4 Configurar docker-compose.yml con límites de disco y logs

```bash
nano docker-compose.yml
```

```yaml
services:
  oe-node:
    image: oceanenterprise/oe-node:latest
    container_name: oe-node
    restart: unless-stopped
    pull_policy: always
    ports:
      - '9000:9000'
      - '9001:9001'
      - '9002:9002'
      - '9003:9003'
      - '9005:9005'
      - '8000:8000'
    depends_on:
      elasticsearch:
        condition: service_healthy
        restart: true
    networks:
      - backend
    volumes:
      - ./oe-node-logs:/usr/src/app/logs
      - ./oe-node-c2d-storage:/usr/src/app/c2d_storage
      - ./oe-node-databases:/usr/src/app/databases
      - /var/run/docker.sock:/var/run/docker.sock
    env_file: .env.node
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.19.3
    container_name: elasticsearch
    restart: unless-stopped
    env_file: .env.elasticsearch
    environment:
      - "ES_JAVA_OPTS=-Xms512m -Xmx1g"
    ulimits:
      memlock:
        soft: -1
        hard: -1
    ports:
      - '9200:9200'
    healthcheck:
      test:
        [
          'CMD-SHELL',
          'curl -s http://localhost:9200/_cluster/health | grep -vq ''"status":"red"'''
        ]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    volumes:
      - esdata:/usr/share/elasticsearch/data
    networks:
      - backend
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"

networks:
  backend:
    driver: bridge

volumes:
  esdata:
    driver_opts:
      type: none
      o: bind
      device: /var/www/ocean-node/ocean-node/esdata
```

### 5.5 Crear directorio de datos y arrancar

```bash
mkdir -p /var/www/ocean-node/ocean-node/esdata
docker compose up -d
```

### 5.6 Verificar que el nodo responde

```bash
curl http://localhost:8000
```

Debe devolver un JSON con `nodeId`, `chainIds`, `providerAddress` y `serviceEndpoints`.

---

## 6. Instalación del Marketplace

### 6.1 Clonar el fork del marketplace

```bash
cd /var/www
git clone https://github.com/Ibatec-es/market.git
cd market
cp .env.example .env
nano .env
```

### 6.2 Configurar el .env del marketplace

```env
# OE NODES
NEXT_PUBLIC_PROVIDER_URL=http://localhost:8000
NEXT_PUBLIC_METADATACACHE_URI=http://localhost:8000/
NEXT_PUBLIC_NODE_URI_INDEXED=["http://localhost:8000/"]

# BLOCKCHAIN RPC
NEXT_PUBLIC_NODE_URI_MAP={"11155111":"https://eth-sepolia.g.alchemy.com/v2/TU_API_KEY"}

# IPFS (básico para pruebas)
NEXT_PUBLIC_IPFS_GATEWAY=https://ipfs.io/
NEXT_PUBLIC_IPFS_UPLOAD_URL=
NEXT_PUBLIC_IPFS_DELETE_URL=
IPFS_JWT=
NEXT_PUBLIC_IPFS_UNPIN_FILES=false

# CURRENCIES (EURC en Sepolia)
NEXT_PUBLIC_ALLOWED_ERC20_ADDRESSES={"11155111":["0x08210F9170F89Ab7658F0B5E3fF39b0E03C594D4"]}

# SSI - desactivado
NEXT_PUBLIC_SSI_ENABLED=false
NEXT_PUBLIC_SSI_DEFAULT_POLICIES_URL=

# OTROS
NEXT_PUBLIC_ENCRYPT_ASSET=true
NEXT_PUBLIC_HIDE_ONBOARDING_MODULE_BY_DEFAULT=false
```

### 6.3 Arrancar el marketplace

```bash
docker compose down --remove-orphans
docker compose up -d
```

### 6.4 Verificar que el marketplace responde

```bash
curl http://localhost:8008
```

Debe devolver HTML de la aplicación Next.js.

---

## 7. Nginx — Reverse Proxy

Permite acceder al nodo y marketplace por dominio sin especificar puerto.
Este setup asume que **ya tienes SSL en tu dominio** (certificado propio o wildcard).

### 7.1 Instalar Nginx

```bash
apt update
apt install -y nginx
```

### 7.2 Configurar virtual host para el OE Node

```bash
nano /etc/nginx/sites-available/ocean-node
```

```nginx
server {
    listen 80;
    server_name node.tudominio.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name node.tudominio.com;

    ssl_certificate     /ruta/a/tu/certificado.crt;
    ssl_certificate_key /ruta/a/tu/clave.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

### 7.3 Configurar virtual host para el Marketplace

```bash
nano /etc/nginx/sites-available/ocean-market
```

```nginx
server {
    listen 80;
    server_name market.tudominio.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name market.tudominio.com;

    ssl_certificate     /ruta/a/tu/certificado.crt;
    ssl_certificate_key /ruta/a/tu/clave.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://localhost:8008;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

### 7.4 Activar los virtual hosts

```bash
ln -s /etc/nginx/sites-available/ocean-node /etc/nginx/sites-enabled/
ln -s /etc/nginx/sites-available/ocean-market /etc/nginx/sites-enabled/

# Verificar configuración
nginx -t

# Recargar Nginx
systemctl reload nginx
systemctl enable nginx
```

### 7.5 Actualizar el marketplace para usar la URL pública del nodo

Una vez Nginx esté configurado, actualizar el `.env` del marketplace:

```bash
nano /var/www/market/.env
```

Cambiar estas líneas:
```env
NEXT_PUBLIC_PROVIDER_URL=https://node.tudominio.com
NEXT_PUBLIC_METADATACACHE_URI=https://node.tudominio.com/
NEXT_PUBLIC_NODE_URI_INDEXED=["https://node.tudominio.com/"]
```

Reiniciar el marketplace:
```bash
cd /var/www/market
docker compose down && docker compose up -d
```

---

## 8. Firewall — UFW

### 8.1 Instalar y configurar UFW

```bash
apt install -y ufw

# Política por defecto: denegar entrada, permitir salida
ufw default deny incoming
ufw default allow outgoing

# SSH — ¡imprescindible, no bloquearte!
ufw allow ssh
# Si usas un puerto SSH personalizado:
# ufw allow TU_PUERTO_SSH/tcp

# HTTP y HTTPS (para Nginx)
ufw allow 80/tcp
ufw allow 443/tcp

# P2P del OE Node (necesario para comunicación entre nodos de la red)
ufw allow 9000/tcp
ufw allow 9001/tcp
ufw allow 9002/tcp
ufw allow 9003/tcp
ufw allow 9005/tcp

# Bloquear puertos internos explícitamente
ufw deny 9200/tcp
ufw deny 8000/tcp
ufw deny 8008/tcp

# Activar firewall
ufw enable

# Verificar estado
ufw status verbose
```

### 8.2 Resumen de puertos

| Puerto | Servicio | Estado |
|---|---|---|
| 22 | SSH | ✅ Abierto |
| 80 | HTTP (redirect) | ✅ Abierto |
| 443 | HTTPS (Nginx) | ✅ Abierto |
| 9000-9005 | P2P OE Node | ✅ Abierto |
| 8000 | OE Node API | ❌ Cerrado (via Nginx) |
| 8008 | Marketplace | ❌ Cerrado (via Nginx) |
| 9200 | Elasticsearch | ❌ Cerrado (interno) |

---

## 9. Verificación final

```bash
# Estado de los contenedores
docker ps

# Estado de Nginx
systemctl status nginx

# Estado del firewall
ufw status

# Test local del nodo
curl http://localhost:8000

# Test local del marketplace
curl -s http://localhost:8008 | head -5

# Test via dominio público
curl https://node.tudominio.com
curl https://market.tudominio.com
```

---

## 10. Mantenimiento y comandos útiles

### Ver logs en tiempo real

```bash
# Nodo
cd /var/www/ocean-node/ocean-node && docker compose logs -f oe-node

# Marketplace
cd /var/www/market && docker compose logs -f

# Nginx
tail -f /var/log/nginx/error.log
```

### Reiniciar servicios

```bash
cd /var/www/ocean-node/ocean-node && docker compose restart oe-node
cd /var/www/market && docker compose restart
systemctl restart nginx
```

### Actualizar versiones

```bash
# Nodo
cd /var/www/ocean-node/ocean-node
docker compose pull && docker compose up -d

# Marketplace (desde tu fork)
cd /var/www/market
git pull origin main
docker compose pull && docker compose up -d
```

### Controlar espacio en disco

```bash
df -h                          # Uso general
docker system df               # Uso Docker
docker image prune -f          # Limpiar imágenes huérfanas
docker builder prune -f        # Limpiar caché de builds
du -sh /var/www/ocean-node/ocean-node/esdata/   # Datos Elasticsearch
```

---

## 11. Futuro: Conectar con Pontus-X

### ¿Qué es Pontus-X?

Pontus-X es el primer y mayor ecosistema X europeo públicamente disponible. Usa Gaia-X, Ocean Enterprise smart contracts y OASIS para potenciar la colaboración de datos entre industrias, con cumplimiento de regulaciones europeas (Data Act, Data Governance Act, AI Act, GDPR).

Con más de 180 instituciones incorporadas — incluyendo Airbus, OHB SE y T-Systems — el ecosistema soporta más de 570 servicios en sectores como aeroespacial, fabricación, movilidad, IA y agricultura. Opera en producción desde febrero de 2025.

**Tu nodo ya habla el mismo idioma que Pontus-X** — ambos usan Ocean Enterprise smart contracts. La diferencia está en la red blockchain (ahora estás en Sepolia testnet) y en los requisitos de identidad (SSI + Gaia-X Trust Framework).

### ¿Es técnicamente difícil?

La parte técnica no es lo más complejo — el mayor esfuerzo es el proceso de identidad y cumplimiento legal:

| Aspecto | Dificultad | Descripción |
|---|---|---|
| Identidad SSI + Gaia-X | ⭐⭐⭐ Media | Generar credenciales verificables Gaia-X |
| Activar SSI en el nodo | ⭐⭐ Baja-Media | Configurar Policy Server y SSI Stack |
| Cambiar de red (Sepolia → Pontus-X) | ⭐⭐ Baja-Media | Cambiar RPC y chainId en el `.env` |
| Onboarding legal | ⭐⭐⭐ Media | Requiere ser persona jurídica verificada |

### Requisitos para unirse a Pontus-X

El ecosistema Pontus-X requiere que todos los participantes estén identificados y usa Self-Sovereign Identities (SSI) como base. Solo pueden participar personas jurídicas cuyas identidades son validadas contra registros oficiales.

Necesitarás:

1. **Nombre legal** que coincida con registros oficiales
2. **Número de registro** — alguno de: EUID, EORI, VAT ID o LEI Code
3. **Dirección física** de la sede (formato ISO 3166-2, ej: `ES-MD`)
4. **Participant Credential Gaia-X** — generada con el Gaia-X Wizard
5. **Dirección pública de wallet** para el registro en el ecosistema

### Pasos para unirse (cuando estés listo)

#### Paso 1 — Generar tu identidad Gaia-X

Usar el Gaia-X Wizard para generar tus Verifiable Credentials firmadas con tu clave privada:
```
https://wizard.lab.gaia-x.eu/
```

#### Paso 2 — Activar SSI en tu nodo OE

Actualmente tienes `POLICY_SERVER_URL=` vacío. Para Pontus-X necesitarás el stack SSI completo con estos componentes adicionales:

- SSI Wallet (walt.id)
- Policy Server
- Policy Server Proxy
- OPA Server

Documentación: [SSI Stack installation](https://docs.oceanenterprise.io/infrastructure/ssi-stack-installation-and-configuration)

#### Paso 3 — Cambiar la red blockchain

Pontus-X usa su propia red basada en Oasis Network. Actualizar el `RPCS` en `.env.node` con el RPC de la red de Pontus-X en lugar de Sepolia. Consultar: [docs.pontus-x.eu](https://docs.pontus-x.eu)

#### Paso 4 — Solicitar el onboarding

```
Email: contact@delta-dao.com
Asunto: Pontus-X onboarding
```

El proceso de validación suele completarse en 24 horas.

#### Paso 5 — Obtener tokens de red

```
https://portal.pontus-x.eu/faucet
```

### Tu ventaja actual

El hecho de tener ya un OE Node funcionando con marketplace propio es exactamente el punto de partida correcto para Pontus-X:

- Nodo OE operativo ✅
- Marketplace propio y personalizable (fork Ibatec-es) ✅
- Stack Docker controlado y documentado ✅
- Falta: identidad Gaia-X + SSI Stack + cambio de red

### Recursos para el futuro

| Recurso | URL |
|---|---|
| Documentación Pontus-X | [docs.pontus-x.eu](https://docs.pontus-x.eu) |
| Portal Pontus-X | [portal.pontus-x.eu](https://portal.pontus-x.eu) |
| Gaia-X Wizard (credenciales) | [wizard.lab.gaia-x.eu](https://wizard.lab.gaia-x.eu) |
| Nautilus SDK (toolkit TypeScript) | [nautilus.delta-dao.com](https://nautilus.delta-dao.com) |
| SSI Stack OE | [docs.oceanenterprise.io/infrastructure/ssi-stack](https://docs.oceanenterprise.io/infrastructure/ssi-stack-installation-and-configuration) |
| Gaia-X Trust Framework | [docs.gaia-x.eu/policy-rules-committee/trust-framework](https://docs.gaia-x.eu/policy-rules-committee/trust-framework/22.10/) |

---

## Notas adicionales

- **IPFS para producción:** Configurar [Pinata](https://pinata.cloud) y añadir `NEXT_PUBLIC_IPFS_UPLOAD_URL`, `NEXT_PUBLIC_IPFS_DELETE_URL` e `IPFS_JWT` en el `.env` del marketplace.
- **Fork del marketplace:** [github.com/Ibatec-es/market](https://github.com/Ibatec-es/market)
- **Red actual:** Testnet Sepolia (chainId `11155111`) con Alchemy como proveedor RPC.
- **SSI:** Desactivado actualmente. Necesario activarlo para unirse a Pontus-X en producción.

---

*Generado durante la instalación — Ocean Enterprise Node v2.1.1 — Abril 2026*
