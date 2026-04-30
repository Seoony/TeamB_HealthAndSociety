import os
from pathlib import Path
from dotenv import load_dotenv

# Rutas del Proyecto
PROJ_ROOT = Path(__file__).resolve().parent.parent
#DATA_DIR = BASE_DIR / "data"
#MODELS_DIR = BASE_DIR / "models"
#REPORTS_DIR = BASE_DIR / "reports"

# Cargar variables de entorno del archivo .env
load_dotenv(dotenv_path=PROJ_ROOT / ".env")

class Config:
  """Configuración centralizada para el Equipo B (Hospitalizaciones y Decesos)"""

  BASE_DIR = PROJ_ROOT

  # Infraestructura de Datos (OSCER/PostgreSQL) 
  DB_USER = os.getenv("DB_USER", "postgres")
  DB_PASS = os.getenv("DB_PASS", "")
  DB_HOST = os.getenv("DB_HOST", "localhost")
  DB_PORT = os.getenv("DB_PORT", "5432")
  DB_NAME = os.getenv("DB_NAME", "isphysa_db")
  
  
  @classmethod
  def get_db_uri(cls):
    """Genera la URI de conexión para SQLAlchemy"""
    return f"postgresql://{cls.DB_USER}:{cls.DB_PASS}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"

# Instancia global para importar en el resto del código
config = Config()