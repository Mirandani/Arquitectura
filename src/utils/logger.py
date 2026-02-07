"""Módulo para configuración centralizada de logging."""

import os
import logging
import inspect
from datetime import datetime
from pathlib import Path


def configurar_logger(nombre_modulo=None):
    """Configura el logger para guardar en archivo y mostrar en consola.

    Args:
        nombre_modulo (str, optional): Nombre del módulo para identificar el
            archivo de log. Si no se proporciona o es "__main__", se detecta
            automáticamente desde el archivo que llama la función.

    Returns:
        logging.Logger: Logger configurado con handlers de archivo y consola.
    """
    # Si no se proporciona nombre o es __main__, detectar el nombre del archivo
    if nombre_modulo is None or nombre_modulo == "__main__":
        frame = inspect.currentframe().f_back
        caller_file = frame.f_code.co_filename
        nombre_modulo = Path(caller_file).stem

    log_dir = "artifacts/logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/{nombre_modulo}_{timestamp}.log"

    # Crear handlers
    file_handler = logging.FileHandler(log_filename)
    console_handler = logging.StreamHandler()
    
    # Formato para ambos handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configurar el logger específico
    logger = logging.getLogger(nombre_modulo)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
