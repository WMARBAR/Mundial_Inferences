import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

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
