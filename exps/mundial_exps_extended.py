import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, hypergeom
import numpy as np

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

    # ==================== DISTRIBUCIONES NUEVAS ====================
    
    def distribucion_binomial(self, df, n_trials=None, success_prob=None):
        """
        Calcula 3 probabilidades usando distribución binomial para análisis de mundiales.
        
        Args:
            df (pd.DataFrame): Base de datos de mundiales
            n_trials (int): Número de ensayos (por defecto: número de mundiales únicos)
            success_prob (float): Probabilidad de éxito (por defecto: probabilidad empírica de ser campeón)
        
        Returns:
            dict: Diccionario con 3 probabilidades binomiales
        """
        # Configurar parámetros por defecto
        if n_trials is None:
            n_trials = df['MYEAR'].nunique()  # Número de mundiales diferentes
        
        if success_prob is None:
            success_prob = self.probabilidad_binaria_empirica(df, 'dummy_campeon')
        
        # Calcular 3 probabilidades diferentes
        resultados = {
            'prob_exactamente_1_campeonato': binom.pmf(1, n_trials, success_prob),
            'prob_al_menos_2_campeonatos': 1 - binom.cdf(1, n_trials, success_prob),
            'prob_maximo_3_campeonatos': binom.cdf(3, n_trials, success_prob)
        }
        
        print(f"Distribución Binomial (n={n_trials}, p={success_prob:.4f}):")
        for key, value in resultados.items():
            print(f"  {key}: {value:.6f}")
        
        return resultados
    
    def distribucion_poisson(self, df, lambda_param=None):
        """
        Calcula 3 probabilidades usando distribución de Poisson para análisis de goles.
        
        Args:
            df (pd.DataFrame): Base de datos de mundiales
            lambda_param (float): Parámetro lambda (por defecto: promedio de goles marcados)
        
        Returns:
            dict: Diccionario con 3 probabilidades de Poisson
        """
        # Configurar parámetro por defecto
        if lambda_param is None:
            # Calcular promedio de goles marcados por jugador en mundiales
            lambda_param = df['Goles Marcados(mundial)'].mean()
        
        # Calcular 3 probabilidades diferentes
        resultados = {
            'prob_0_goles': poisson.pmf(0, lambda_param),
            'prob_exactamente_2_goles': poisson.pmf(2, lambda_param),
            'prob_mas_de_3_goles': 1 - poisson.cdf(3, lambda_param)
        }
        
        print(f"Distribución Poisson (λ={lambda_param:.4f}):")
        for key, value in resultados.items():
            print(f"  {key}: {value:.6f}")
        
        return resultados
    
    def distribucion_hipergeometrica(self, df, N=None, K=None, n=None):
        """
        Calcula 3 probabilidades usando distribución hipergeométrica para selección de jugadores.
        
        Args:
            df (pd.DataFrame): Base de datos de mundiales
            N (int): Población total (por defecto: total de jugadores únicos)
            K (int): Número de éxitos en población (por defecto: jugadores goleadores)
            n (int): Número de extracciones (por defecto: jugadores por selección promedio)
        
        Returns:
            dict: Diccionario con 3 probabilidades hipergeométricas
        """
        # Configurar parámetros por defecto
        if N is None:
            N = df['PLAYER_NAME'].nunique()  # Total de jugadores únicos
        
        if K is None:
            K = df['dummy_goleador'].sum()  # Total de goleadores
        
        if n is None:
            # Promedio de jugadores por selección en cada mundial
            jugadores_por_seleccion = df.groupby(['MYEAR', 'SIG_SLECCCION']).size()
            n = int(jugadores_por_seleccion.mean())
        
        # Validar parámetros
        if K > N or n > N:
            print("Error: Parámetros inválidos para distribución hipergeométrica")
            return {}
        
        # Calcular 3 probabilidades diferentes
        resultados = {
            'prob_0_goleadores_seleccion': hypergeom.pmf(0, N, K, n),
            'prob_exactamente_1_goleador': hypergeom.pmf(1, N, K, n),
            'prob_al_menos_2_goleadores': 1 - hypergeom.cdf(1, N, K, n)
        }
        
        print(f"Distribución Hipergeométrica (N={N}, K={K}, n={n}):")
        for key, value in resultados.items():
            print(f"  {key}: {value:.6f}")
        
        return resultados
    
    def analisis_completo_distribuciones(self, df):
        """
        Ejecuta análisis completo con las 3 distribuciones.
        
        Args:
            df (pd.DataFrame): Base de datos de mundiales
        
        Returns:
            dict: Resultados de todas las distribuciones
        """
        print("=== ANÁLISIS COMPLETO DE DISTRIBUCIONES ===\n")
        
        resultados_completos = {
            'binomial': self.distribucion_binomial(df),
            'poisson': self.distribucion_poisson(df),
            'hipergeometrica': self.distribucion_hipergeometrica(df)
        }
        
        return resultados_completos
    
    def comparacion_distribuciones(self, df):
        """
        Genera gráficos comparativos de las distribuciones.
        
        Args:
            df (pd.DataFrame): Base de datos de mundiales
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Binomial
        n_trials = df['MYEAR'].nunique()
        p = self.probabilidad_binaria_empirica(df, 'dummy_campeon')
        x_binom = np.arange(0, min(n_trials + 1, 20))
        y_binom = [binom.pmf(k, n_trials, p) for k in x_binom]
        
        axes[0,0].bar(x_binom, y_binom, alpha=0.7, color='blue')
        axes[0,0].set_title('Distribución Binomial\n(Campeonatos por país)')
        axes[0,0].set_xlabel('Número de campeonatos')
        axes[0,0].set_ylabel('Probabilidad')
        
        # Poisson
        lambda_param = df['Goles Marcados(mundial)'].mean()
        x_poisson = np.arange(0, 15)
        y_poisson = [poisson.pmf(k, lambda_param) for k in x_poisson]
        
        axes[0,1].bar(x_poisson, y_poisson, alpha=0.7, color='green')
        axes[0,1].set_title('Distribución Poisson\n(Goles por jugador)')
        axes[0,1].set_xlabel('Número de goles')
        axes[0,1].set_ylabel('Probabilidad')
        
        # Hipergeométrica
        N = df['PLAYER_NAME'].nunique()
        K = df['dummy_goleador'].sum()
        jugadores_por_seleccion = df.groupby(['MYEAR', 'SIG_SLECCCION']).size()
        n = int(jugadores_por_seleccion.mean())
        
        if K <= N and n <= N:
            x_hyper = np.arange(0, min(K + 1, n + 1, 10))
            y_hyper = [hypergeom.pmf(k, N, K, n) for k in x_hyper]
            
            axes[1,0].bar(x_hyper, y_hyper, alpha=0.7, color='red')
            axes[1,0].set_title('Distribución Hipergeométrica\n(Goleadores por selección)')
            axes[1,0].set_xlabel('Número de goleadores')
            axes[1,0].set_ylabel('Probabilidad')
        
        # Resumen estadístico
        axes[1,1].axis('off')
        resumen_text = f"""RESUMEN ESTADÍSTICO:
        
Binomial:
- Ensayos (n): {n_trials}
- Prob. éxito (p): {p:.4f}
- Media: {n_trials * p:.2f}

Poisson:
- Lambda (λ): {lambda_param:.4f}
- Media: {lambda_param:.2f}

Hipergeométrica:
- Población (N): {N}
- Éxitos (K): {K}
- Muestra (n): {n}
- Media: {n * K / N:.2f}"""
        
        axes[1,1].text(0.1, 0.9, resumen_text, transform=axes[1,1].transAxes, 
                      fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()