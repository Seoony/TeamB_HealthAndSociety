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
* `src/`: Directorio que contiene el código fuente.
* `.env` / `.env.example`: Archivos para la gestión de variables de entorno (datos sensibles).
* `variables.ps1`: Script para cargar el `.env` en sesiones de PowerShell.
* `requirements.txt`: Dependencias del proyecto.
* `.gitignore`: Configuración para evitar subir archivos innecesarios o sensibles.

### 2. Creación del Entorno Virtual

Se recomienda el uso de un entorno virtual para aislar las dependencias.

#### Opción A: virtualenv (Recomendado)
```bash
# Instalar virtualenv si no lo tienes
pip install virtualenv

# Crear el entorno
virtualenv venv

# Activar (Windows)
.\venv\Scripts\activate
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