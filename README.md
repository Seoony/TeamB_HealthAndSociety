# Proyecto UNSA & Oklahoma: Salud y Sociedad (Equipo B)

Este repositorio contiene el desarrollo del **Equipo B** para el proyecto de investigación conjunta entre la **Universidad Nacional de San Agustín (UNSA)** y la **University of Oklahoma**, enfocado en salud pública y sociedad.

## 👥 Integrantes del Equipo

* **Calienes Rodríguez, Ricardo Fabrizio** (Docente encargado / Líder Team B)
* **Rivas Chire, Anthony Juancarlo** (Líder de Equipo)
* **Haytara Tonconi, Alex Antonio**
* **Quispe Ttito, Juan Carlos**
* **Quispe Huacho, Rodolfo Robert**
* **Zapana Romero, Pedro Luis Christian**

---

## 🛠️ Configuración del Proyecto

Sigue estos pasos para preparar el entorno de desarrollo. Este proyecto requiere **Python 3**.

### 1. Estructura de Configuración

```text
PROYECTO-UNSA-OKLAHOMA/
├── data/               # Datos del proyecto (No sincronizados en Git)
│   ├── raw/            # Datos originales extraídos de PostgreSQL
│   ├── processed/      # Datos limpios y transformados para el modelo
│   └── external/       # Diccionarios de datos o fuentes externas
├── notebooks/          # Jupyter Notebooks para experimentación y EDA
├── src/                # Código fuente modular
│   ├── config.py       # Configuración centralizada y variables de entorno
│   ├── data_loader.py  # Gestión de conexión a DB y caché local
│   ├── preprocessing.py# Limpieza de datos y feature engineering
│   └── models/         # Scripts de entrenamiento y predicción
├── models/             # Archivos de modelos entrenados (.pkl, .h5) (No sincronizados en Git)
├── reports/            # Informes y resultados
│   └── figures/        # Gráficas generadas para la investigación
├── .env                # Variables sensibles (Local)
├── .gitignore          # Archivos excluidos del repositorio
├── requirements.txt    # Dependencias de Python
└── README.md           # Documentación general
```

* `src/`: Directorio que contiene el código fuente.
* `.env` / `.env.example`: Archivos para la gestión de variables de entorno (datos sensibles).
* `variables.ps1`: Script para cargar el `.env` en sesiones de PowerShell.
* `requirements.txt`: Dependencias del proyecto.
* `.gitignore`: Configuración para evitar subir archivos innecesarios o sensibles.

#### Preparación de Directorios
Dado que las carpetas de datos y modelos están ignoradas en el repositorio por seguridad y peso, debes crearlas localmente ejecutando el siguiente comando:

```bash
mkdir data/raw, data/processed, data/external, notebooks, models -Force
```

### 2. Creación del Entorno Virtual

Se recomienda el uso de un entorno virtual para aislar las dependencias.

#### Opción A: virtualenv (Recomendado)
```bash
# Instalar virtualenv si no lo tienes
pip install virtualenv

# Crear el entorno
virtualenv .

# Activar (Windows)
.\Scripts\activate
```

#### Opción B: venv (Nativo de Python)
```bash
# Crear el entorno
python -m venv venv

# Activar (Windows)
.\venv\Scripts\activate
```
### 3. Instalación de Dependencias
Una vez activado el entorno, instala las librerías necesarias:

```bash
pip install -r requirements.txt
```
### 4. Variables de Envorno
Para configurar el entorno correctamente:

1. Crea una copia de .env.example y nómbrala .env.

2. Completa los valores requeridos dentro de .env.

3. Si usas PowerShell, carga las variables ejecutando:

```powerShell
./variables.ps1
```
## 🚀 Ejecución
Para iniciar el proyecto, navega a la carpeta de código:

```bash
cd src
python main.py
```
### 🔬 Sobre la Investigación
Este trabajo es parte de una colaboración académica internacional para el análisis y visualización de datos de salud en la región de Arequipa.