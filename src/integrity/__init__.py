"""Modulo de validacion y limpieza de integridad de los datos.

Submodulos:
    validation : Detecta nulos, negativos, inconsistencias logicas,
                 duplicados y silencios sospechosos en las tablas raw.
    cleaning   : Aplica las correcciones derivadas del reporte de
                 validacion (deduplicacion, flags, clamp de cases) y
                 deja las tablas en data/processed/.
"""
