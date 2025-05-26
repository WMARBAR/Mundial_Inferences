
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
class Experimentos:
    
    def probabilidad_binaria_empirica(self,df, columna_objetivo):
        """
        Calcula la probabilidad empírica de éxito (valor 1) en una columna binaria.

        Args:
            df (pd.DataFrame): Base de datos.
            columna_objetivo (str): Nombre de la columna con valores 0 y 1.

        Returns:
            float: Probabilidad empírica de éxito (valor 1).
        """
        if columna_objetivo not in df.columns:
            raise ValueError("La columna especificada no existe en el DataFrame.")
        
        total = len(df)
        exitos = df[columna_objetivo].sum()

        if total == 0:
            return 0  # Evita división por cero

        probabilidad = exitos / total
        return probabilidad


    def tabla_probabilidad_conjunta(self, df, columna_1, columna_2):
        """
        Calcula una tabla de contingencia con probabilidades conjuntas.

        Args:
            df (pd.DataFrame): DataFrame con los datos.
            columna_1 (str or list): Una o varias columnas categóricas para combinar como índice (filas).
            columna_2 (str): Nombre de la columna del evento binario (columnas en la tabla).

        Returns:
            pd.DataFrame: Tabla de contingencia normalizada con probabilidades conjuntas.
        """
        # Validación de columnas
        if isinstance(columna_1, list):
            for col in columna_1:
                if col not in df.columns:
                    raise ValueError(f"La columna '{col}' no existe en el DataFrame.")
            if columna_2 not in df.columns:
                raise ValueError(f"La columna '{columna_2}' no existe en el DataFrame.")
            
            # Crear una nueva columna combinada
            nombre_combinado = '_'.join(columna_1)
            df[nombre_combinado] = df[columna_1].astype(str).agg('_'.join, axis=1)
            index_col = nombre_combinado

        elif isinstance(columna_1, str):
            if columna_1 not in df.columns or columna_2 not in df.columns:
                raise ValueError("Una o ambas columnas no existen en el DataFrame.")
            index_col = columna_1

        else:
            raise TypeError("El parámetro 'columna_1' debe ser str o list de str.")

        # Calcular tabla de contingencia con probabilidad conjunta
        tabla = pd.crosstab(df[index_col], df[columna_2], normalize='all')
        return tabla


    def grap_contingencia(self,tabla_contingencia, top=10):
        """
        Genera un heatmap de la probabilidad de ser campeón (columna 1) por país.

        Args:
            tabla_contingencia (pd.DataFrame): DataFrame con índice 'Seleccion' y columnas [0, 1]
            top (int): Número de países a mostrar (ordenados por la probabilidad de ser campeón)

        Returns:
            None. Muestra el gráfico.
        """
        if 1 not in tabla_contingencia.columns:
            raise ValueError("La tabla no contiene la columna '1' (probabilidad de ser campeón).")

        # Seleccionar y ordenar por la columna de interés
        df_top = tabla_contingencia[[1]].sort_values(by=1, ascending=False).head(top)

        # Renombrar columna para estética
        df_top.columns = ['Probabilidad_Campeon']

        # Generar heatmap
        plt.figure(figsize=(8, top * 0.5 + 1))
        sns.heatmap(df_top, annot=True, cmap="YlOrRd", fmt=".5f", linewidths=0.5)
        plt.title(f"Probabilidad conjunta (Top {top})")
        plt.xlabel("Probabilidad")
        plt.ylabel("País")
        plt.tight_layout()
        plt.show()
