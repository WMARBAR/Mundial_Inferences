import pandas as pd
import os
class Loaders:
    
    def cargar_excel(self, path_archivo, nombre_hoja):
        """
        Carga una hoja específica desde un archivo Excel.

        Args:
            path_archivo (str): Ruta completa del archivo Excel.
            nombre_hoja (str): Nombre de la hoja a cargar.

        Returns:
            pd.DataFrame: Contenido de la hoja en un DataFrame de pandas.
        """
        try:
            df = pd.read_excel(path_archivo, sheet_name=nombre_hoja)
            return df
        except Exception as e:
            print(f"Error al leer el archivo: {e}")
            return None


    def guardar_excel(self,df, nombre_archivo):
        """
        Guarda un DataFrame como archivo Excel en la raíz del proyecto.

        Args:
            df (pd.DataFrame): La base de datos a guardar.
            nombre_archivo (str): Nombre del archivo (sin extensión .xlsx).
        """
        ruta = os.path.join(os.getcwd(), f"{nombre_archivo}.xlsx")
        try:
            df.to_excel(ruta, index=False)
            print(f"✅ Archivo guardado exitosamente en: {ruta}")
        except Exception as e:
            print(f"❌ Error al guardar el archivo: {e}")
