import pandas as pd
from sqlalchemy import create_engine, inspect, text
from src.config import config
from tqdm import tqdm
import time
import os

def download_all_tables(force_download=False):
    engine = create_engine(config.get_db_uri())
    raw_dir = config.BASE_DIR / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print(f"📂 Se detectaron {len(tables)} tablas en el servidor.")
    
    for table_name in tables:
        local_path = raw_dir / f"{table_name}.parquet"
        
        if local_path.exists() and not force_download:
            print(f"⏩ Saltando '{table_name}' (ya existe).")
            continue
            
        print(f"\n📥 Iniciando descarga de: '{table_name}'")
        start_time = time.time()
        
        try:
            # Usamos chunks para dar feedback constante
            chunk_size = 50000  # 50k filas por vez
            chunks = []
            
            # Consultamos el total de filas para la barra de progreso
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                total_rows = result.scalar()
            
            with tqdm(total=total_rows, desc=f"📊 Progreso {table_name}", unit="filas") as pbar:
                # Descarga por partes (chunks)
                for chunk in pd.read_sql_query(f"SELECT * FROM {table_name}", engine, chunksize=chunk_size):
                    chunks.append(chunk)
                    pbar.update(len(chunk))
            
            # Consolidar y guardar
            df = pd.concat(chunks, ignore_index=True)
            df.to_parquet(local_path, index=False)
            
            duration = time.time() - start_time
            print(f"✅ Completado: {len(df)} filas en {duration:.2f}s ({len(df)/duration:.2f} filas/s)")
            
        except Exception as e:
            print(f"❌ Error en '{table_name}': {e}")

if __name__ == "__main__":
    download_all_tables()