import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm
class AnalisisEstadistico:
    def __init__(self, df):
        self.df = df

    def analisis_binomial(self, columna_dummy, n, p=None):
        if p is None:
            p = self.df[columna_dummy].mean()

        x = np.arange(0, n + 1)
        pmf = stats.binom.pmf(x, n, p)

        plt.bar(x, pmf)
        plt.title(f'Distribución Binomial (n={n}, p={p:.2f})')
        plt.xlabel('Número de éxitos')
        plt.ylabel('Probabilidad')
        plt.grid(True)
        plt.show()

        return pd.DataFrame({'x': x, 'P(X=x)': pmf})

    def analisis_poisson(self, columna_eventos):
        lamb = self.df[columna_eventos].mean()
        x = np.arange(0, self.df[columna_eventos].max() + 5)
        probs = stats.poisson.pmf(x, lamb)

        plt.bar(x, probs)
        plt.title(f'Distribución de Poisson (λ={lamb:.2f})')
        plt.xlabel('Eventos')
        plt.ylabel('Probabilidad')
        plt.grid(True)
        plt.show()

        return pd.DataFrame({'x': x, 'P(X=x)': probs})

    def analisis_normal(self, columna):
        mu = self.df[columna].mean()
        sigma = self.df[columna].std()
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
        y = stats.norm.pdf(x, mu, sigma)

        plt.plot(x, y)
        plt.title(f'Distribución Normal (μ={mu:.2f}, σ={sigma:.2f})')
        plt.xlabel(columna)
        plt.ylabel('Densidad')
        plt.grid(True)
        plt.show()

        return mu, sigma

    def prueba_normalidad(self, columna):
        stat, p = stats.shapiro(self.df[columna].dropna())
        print(f"Shapiro-Wilk: W={stat:.4f}, p={p:.4f}")
        if p > 0.05:
            print("No se rechaza H0: la variable parece seguir una distribución normal.")
        else:
            print("Se rechaza H0: la variable no sigue una distribución normal.")

    def prueba_t_media(self, columna, grupo, grupo1, grupo2):
        x1 = self.df[self.df[grupo] == grupo1][columna].dropna()
        x2 = self.df[self.df[grupo] == grupo2][columna].dropna()
        stat, p = stats.ttest_ind(x1, x2, equal_var=False)
        print(f"Prueba t: t={stat:.4f}, p={p:.4f}")
        if p < 0.05:
            print("Se rechaza H0: hay diferencia significativa entre los grupos.")
        else:
            print("No se rechaza H0: no hay diferencia significativa.")

    def regresion_lineal_simple(self, x_col, y_col):
        df_clean = self.df[[x_col, y_col]].dropna()
        X = sm.add_constant(df_clean[x_col])
        Y = df_clean[y_col]
        model = sm.OLS(Y, X).fit()
        print(model.summary())

        sns.regplot(x=x_col, y=y_col, data=df_clean, line_kws={"color": "red"})
        plt.title('Regresión Lineal Simple')
        plt.grid(True)
        plt.show()

        return model


class Muestreos:
    def muestreo_aleatorio_simple(self, df, n):
        """
        Devuelve una muestra aleatoria simple de tamaño n.
        
        Args:
            df (pd.DataFrame): Base de datos completa.
            n (int): Tamaño de la muestra.
            
        Returns:
            pd.DataFrame: Subconjunto muestreado.
        """
        return df.sample(n=n, random_state=42).reset_index(drop=True)


    def muestreo_estratificado(self,df, columna_estrato, n_por_estrato):
        """
        Muestreo estratificado con tamaño fijo por estrato.
        
        Args:
            df (pd.DataFrame): Base de datos.
            columna_estrato (str): Nombre de la columna con los estratos.
            n_por_estrato (int): Tamaño por cada grupo.
        
        Returns:
            pd.DataFrame: Subconjunto muestreado.
        """
        return df.groupby(columna_estrato, group_keys=False).apply(lambda x: x.sample(n=min(n_por_estrato, len(x)), random_state=42)).reset_index(drop=True)
    

    def muestreo_estratificado_proporcional(self,df, columna_estrato, n_total):
        """
        Muestreo estratificado proporcional al tamaño de cada grupo.

        Args:
            df (pd.DataFrame): Base de datos.
            columna_estrato (str): Columna categórica para estratificar (ej. 'confederacion').
            n_total (int): Tamaño total de la muestra.

        Returns:
            pd.DataFrame: Muestra estratificada proporcionalmente.
        """
        freqs = df[columna_estrato].value_counts(normalize=True)
        tamaños = (freqs * n_total).round().astype(int)

        muestra = df.groupby(columna_estrato, group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), tamaños[g.name]), random_state=42)
        ).reset_index(drop=True)

        return muestra


    def muestreo_sistematico(self,df, n):
        """
        Muestreo sistemático sobre la base ordenada.
        
        Args:
            df (pd.DataFrame): Base de datos.
            n (int): Tamaño deseado de la muestra.
        
        Returns:
            pd.DataFrame: Subconjunto muestreado.
        """
        N = len(df)
        k = N // n
        start = np.random.randint(0, k)
        indices = np.arange(start, N, k)[:n]
        return df.iloc[indices].reset_index(drop=True)
    
    def resumen_estimaciones(self, df_pob, df_muestra, nombre_muestra, columna_media, columna_proporcion):
        """
        Calcula estimaciones puntuales y las compara con la población.
        """
        resumen = {
            'muestra': nombre_muestra,
            'media_muestral': df_muestra[columna_media].mean(),
            'media_poblacional': df_pob[columna_media].mean(),
            'proporcion_muestral': df_muestra[columna_proporcion].mean(),
            'proporcion_poblacional': df_pob[columna_proporcion].mean(),
            'varianza_muestral': df_muestra[columna_media].var(ddof=1),
            'varianza_poblacional': df_pob[columna_media].var(ddof=0)
        }
        return resumen


    def calcular_intervalos(self, df, nombre_muestra, col_media, col_proporcion, confianza=0.95):
        n = len(df)
        z = norm.ppf(1 - (1 - confianza) / 2)

        media = df[col_media].mean()
        s = df[col_media].std(ddof=1)
        error_media = z * s / np.sqrt(n)

        p_hat = df[col_proporcion].mean()
        error_prop = z * np.sqrt(p_hat * (1 - p_hat) / n)

        return {
            'muestra': nombre_muestra,
            'media_IC': (media - error_media, media + error_media),
            'proporcion_IC': (p_hat - error_prop, p_hat + error_prop)
        }


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (
    chisquare, kstest, anderson, shapiro, jarque_bera,
    weibull_min, norm, gamma, expon, uniform
)


class GoodnessOfFitAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.numeric_columns = self._get_numeric_columns()
        self.categorical_columns = self._get_categorical_columns()
        self.results = {}

    def _get_numeric_columns(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['KEY', 'MYEAR', 'birth_year']
        return [col for col in numeric_cols if col not in exclude_cols]

    def _get_categorical_columns(self):
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        important_cats = ['Posicion', 'Seleccion', 'confederacion', 'FaseAlcanzada', 'Era']
        return [col for col in categorical_cols if col in important_cats]

    def chi_square_test(self, column, expected_freq=None):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")
        observed_freq = self.df[column].value_counts().values
        if expected_freq is None:
            expected_freq = np.full(len(observed_freq), np.mean(observed_freq))
        chi2_stat, p_val = chisquare(observed_freq, expected_freq)
        result = {
            'test': 'Chi-square',
            'column': column,
            'statistic': chi2_stat,
            'p_value': p_val,
            'observed_frequencies': observed_freq,
            'expected_frequencies': expected_freq,
            'interpretation': 'Reject H0: Not uniform distribution' if p_val < 0.05 else 'Fail to reject H0: Uniform distribution'
        }
        self.results[f'chi_square_{column}'] = result
        return result

    def kolmogorov_smirnov_test(self, column, distribution='norm'):
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric or not found")
        data = self.df[column].dropna()
        if distribution == 'norm':
            data_std = (data - data.mean()) / data.std()
            ks_stat, p_val = kstest(data_std, 'norm')
        else:
            ks_stat, p_val = kstest(data, distribution)
        result = {
            'test': 'Kolmogorov-Smirnov',
            'column': column,
            'distribution_tested': distribution,
            'statistic': ks_stat,
            'p_value': p_val,
            'sample_size': len(data),
            'interpretation': f'Reject H0: Not {distribution} distribution' if p_val < 0.05 else f'Fail to reject H0: {distribution} distribution'
        }
        self.results[f'ks_{column}_{distribution}'] = result
        return result

    def anderson_darling_test(self, column, distribution='norm'):
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric or not found")
        data = self.df[column].dropna()
        ad_result = anderson(data, dist=distribution)
        critical_value = ad_result.critical_values[2]
        significance_level = ad_result.significance_level[2]
        result = {
            'test': 'Anderson-Darling',
            'column': column,
            'distribution_tested': distribution,
            'statistic': ad_result.statistic,
            'critical_value': critical_value,
            'significance_level': significance_level,
            'interpretation': f'Reject H0: Not {distribution} distribution' if ad_result.statistic > critical_value else f'Fail to reject H0: {distribution} distribution'
        }
        self.results[f'ad_{column}_{distribution}'] = result
        return result

    def shapiro_wilk_test(self, column):
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric or not found")
        data = self.df[column].dropna()
        if len(data) > 5000:
            data = data.sample(n=5000, random_state=42)
        stat, p_val = shapiro(data)
        result = {
            'test': 'Shapiro-Wilk',
            'column': column,
            'statistic': stat,
            'p_value': p_val,
            'sample_size': len(data),
            'interpretation': 'Reject H0: Not normal distribution' if p_val < 0.05 else 'Fail to reject H0: Normal distribution'
        }
        self.results[f'shapiro_{column}'] = result
        return result

    def jarque_bera_test(self, column):
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric or not found")
        data = self.df[column].dropna()
        stat, p_val = jarque_bera(data)
        result = {
            'test': 'Jarque-Bera',
            'column': column,
            'statistic': stat,
            'p_value': p_val,
            'sample_size': len(data),
            'interpretation': 'Reject H0: Not normal distribution' if p_val < 0.05 else 'Fail to reject H0: Normal distribution'
        }
        self.results[f'jb_{column}'] = result
        return result

    def fit_distribution(self, column, distributions=['norm', 'gamma', 'expon', 'weibull_min']):
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric or not found")
        data = self.df[column].dropna()
        results = {}
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                ks_stat, p_val = kstest(data, dist.cdf, args=params)
                log_likelihood = np.sum(dist.logpdf(data, *params))
                aic = 2 * len(params) - 2 * log_likelihood
                bic = len(params) * np.log(len(data)) - 2 * log_likelihood
                results[dist_name] = {
                    'parameters': params,
                    'ks_statistic': ks_stat,
                    'ks_p_value': p_val,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic
                }
            except Exception as e:
                results[dist_name] = {'error': str(e)}
        self.results[f'fit_{column}'] = results
        return results

    def comprehensive_analysis(self, column):
        print(f"\n=== Comprehensive Analysis for {column} ===")
        results = {}
        if column in self.numeric_columns:
            data = self.df[column].dropna()
            print(f"Sample size: {len(data)}")
            print(f"Mean: {data.mean():.4f}, Std: {data.std():.4f}")
            print(f"Skewness: {stats.skew(data):.4f}, Kurtosis: {stats.kurtosis(data):.4f}")
            results['shapiro'] = self.shapiro_wilk_test(column)
            results['anderson_darling'] = self.anderson_darling_test(column)
            results['jarque_bera'] = self.jarque_bera_test(column)
            results['ks_normal'] = self.kolmogorov_smirnov_test(column, 'norm')
            results['distribution_fit'] = self.fit_distribution(column)
        elif column in self.categorical_columns:
            results['chi_square'] = self.chi_square_test(column)
        else:
            print(f"Column {column} not suitable for analysis")
            return None
        return results

    def plot_distribution_comparison(self, column, distributions=['norm', 'gamma', 'expon']):
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric")
        data = self.df[column].dropna()
        plt.figure(figsize=(12, 8))
        plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Data')
        colors = ['red', 'green', 'orange', 'purple']
        for i, dist_name in enumerate(distributions):
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                x = np.linspace(data.min(), data.max(), 100)
                y = dist.pdf(x, *params)
                plt.plot(x, y, color=colors[i % len(colors)], linewidth=2, label=f'{dist_name} fit')
            except Exception as e:
                print(f"Could not fit {dist_name}: {str(e)}")
        plt.title(f'Distribution Comparison for {column}')
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


