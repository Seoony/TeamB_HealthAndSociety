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

### 5. Descarga de Datos

Antes de comenzar con el modelado predictivo, es necesario sincronizar la base de datos local con el servidor centralizado de OSCER.

#### 1. Variables de Entorno
Asegúrate de tener un archivo `.env` en la raíz del proyecto con las credenciales de la base de datos. Puedes guiarte del archivo `.env.example`.

#### 2. Instalación de Dependencias
Instala las librerías necesarias, incluyendo el driver de PostgreSQL y herramientas de visualización de progreso:
```bash
pip install -r requirements.txt
```

#### 3. Ejecución del script data_loader
Para descargar las tablas prioritarias (Neumonía y Población) y convertirlas al formato de alto rendimiento Parquet, ejecuta el siguiente comando desde la raíz:

```bash
#Ejecución estandar
py -m src.data_loader
```

## 🚀 Ejecución
Para iniciar el proyecto, navega a la carpeta de código:

```bash
cd src
python main.py
```
### 🔬 Sobre la Investigación
Este trabajo forma parte de una colaboración académica internacional entre la Universidad Nacional de San Agustín (UNSA) y la University of Oklahoma (OU) para el análisis y visualización de datos de salud pública.

#### Objetivo del Equipo B
El Equipo B se enfoca en el Análisis y Modelado Predictivo de la Incidencia de Hospitalizaciones y Mortalidad por Infecciones Respiratorias Agudas (IRAs) y Neumonía en el Perú, con énfasis en la región de Arequipa.

#### Alcance Técnico y Científico
* **Análisis Multiescala:** El estudio abarca proyecciones a nivel nacional, regional y provincial.

* **Estratificación Demográfica:** Priorización de grupos vulnerables, específicamente menores de 5 años y adultos mayores de 60 años.

* **Modelado Avanzado:** Implementación de modelos estadísticos (ARIMA/SARIMA), Machine Learning (XGBoost/Prophet) y Deep Learning (LSTM/GRU) para predecir la carga hospitalaria.

#### Impacto Esperado
Los resultados de esta investigación se integrarán en la plataforma PanViz para apoyar la toma de decisiones informadas en salud pública y se publicarán en manuscritos científicos de alto impacto.

