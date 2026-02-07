"""Módulo para configuración centralizada de logging."""

import os
import logging
from datetime import datetime


def configurar_logger(nombre_modulo):
    """Configura el logger para guardar en archivo y mostrar en consola.

    Args:
        nombre_modulo (str): Nombre del módulo para identificar el
            archivo de log.

    Returns:
        logging.Logger: Logger configurado con handlers de archivo y consola.
    """
    log_dir = "artifacts/logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/{nombre_modulo}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )
    return logging.getLogger(nombre_modulo)
