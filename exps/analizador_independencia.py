import pandas as pd

class AnalizadorIndependencia:

    """
    Clase para analizar independencia entre eventos usando probabilidades
    marginales, conjuntas y condicionales
    """

    def crear_tabla_completa_probabilidades(self, df, var1, var2):
        """
        Crea tabla de probabilidades conjuntas con marginales
        """
        # Tabla de contingencia con probabilidades conjuntas
        tabla_conjunta = pd.crosstab(df[var1], df[var2], normalize='all')
        
        # Agregar marginales
        tabla_completa = self.agregar_marginales(tabla_conjunta)
        
        return tabla_completa
    
    def agregar_marginales(self, tabla_conjunta):
        """
        Agrega probabilidades marginales a tabla conjunta
        """
        if set(tabla_conjunta.columns) != {0, 1}:
            raise ValueError("La tabla debe tener exactamente las columnas 0 y 1")
        
        tabla = tabla_conjunta.copy()
        
        # Marginal por fila
        tabla['Marginal_fila'] = tabla.sum(axis=1)
        
        # Marginal por columna
        marginal_col = tabla.sum(axis=0)
        marginal_col['Marginal_fila'] = marginal_col.sum()
        
        # Agregar fila de marginales de columna
        tabla.loc['Marginal_col'] = marginal_col
        
        return tabla
    
    def analizar_independencia_completo(self, df, var1, var2, mostrar_detalles=True):
        """
        Análisis completo de independencia entre dos variables binarias
        """
        print(f"\n{'='*60}")
        print(f"ANÁLISIS DE INDEPENDENCIA: {var1} vs {var2}")
        print(f"{'='*60}")
        
        # 1. Crear tabla de probabilidades
        tabla_completa = self.crear_tabla_completa_probabilidades(df, var1, var2)
        
        if mostrar_detalles:
            print("\n1. TABLA DE PROBABILIDADES CONJUNTAS Y MARGINALES:")
            print(tabla_completa.round(4))
        
        # 2. Extraer probabilidades clave
        # Probabilidades marginales
        P_A1 = tabla_completa.loc['Marginal_col', 1]  # P(evento = 1)
        P_A0 = tabla_completa.loc['Marginal_col', 0]  # P(evento = 0)
        
        # Probabilidades conjuntas
        probabilidades_conjuntas = {}
        probabilidades_condicionales = {}
        
        for categoria in tabla_completa.index[:-1]:  # Excluir 'Marginal_col'
            P_B = tabla_completa.loc[categoria, 'Marginal_fila']  # P(categoria)
            P_AB1 = tabla_completa.loc[categoria, 1]  # P(categoria ∩ evento=1)
            P_AB0 = tabla_completa.loc[categoria, 0]  # P(categoria ∩ evento=0)
            
            probabilidades_conjuntas[categoria] = {
                'P_B': P_B,
                'P_AB1': P_AB1,
                'P_AB0': P_AB0
            }
            
            # Probabilidades condicionales
            P_A1_dado_B = P_AB1 / P_B if P_B > 0 else 0
            P_A0_dado_B = P_AB0 / P_B if P_B > 0 else 0
            
            probabilidades_condicionales[categoria] = {
                'P_A1_dado_B': P_A1_dado_B,
                'P_A0_dado_B': P_A0_dado_B
            }
        
        # 3. Análisis de independencia
        print(f"\n2. PROBABILIDADES MARGINALES:")
        print(f"P(evento = 1) = {P_A1:.4f}")
        print(f"P(evento = 0) = {P_A0:.4f}")
        
        print(f"\n3. ANÁLISIS POR CATEGORÍA:")
        resultados_independencia = {}
        
        for categoria in probabilidades_conjuntas.keys():
            print(f"\n--- {categoria} ---")
            
            P_B = probabilidades_conjuntas[categoria]['P_B']
            P_AB1 = probabilidades_conjuntas[categoria]['P_AB1']
            P_A1_dado_B = probabilidades_condicionales[categoria]['P_A1_dado_B']
            
            # Probabilidad esperada bajo independencia
            P_AB1_esperada = P_A1 * P_B
            
            # Test de independencia
            es_independiente = abs(P_AB1 - P_AB1_esperada) < 0.001
            diferencia = P_AB1 - P_AB1_esperada
            
            print(f"P({categoria}) = {P_B:.4f}")
            print(f"P(evento=1 ∩ {categoria}) = {P_AB1:.4f}")
            print(f"P(evento=1 | {categoria}) = {P_A1_dado_B:.4f}")
            print(f"P(evento=1) × P({categoria}) = {P_AB1_esperada:.4f}")
            print(f"Diferencia: {diferencia:.4f}")
            print(f"¿Independiente?: {'SÍ' if es_independiente else 'NO'}")
            
            if not es_independiente:
                if diferencia > 0:
                    print(f"→ {categoria} AUMENTA la probabilidad del evento")
                else:
                    print(f"→ {categoria} DISMINUYE la probabilidad del evento")
            
            resultados_independencia[categoria] = {
                'es_independiente': es_independiente,
                'diferencia': diferencia,
                'P_condicional': P_A1_dado_B,
                'P_marginal': P_A1
            }
        
        # 4. Resumen general
        print(f"\n4. RESUMEN DE INDEPENDENCIA:")
        independientes = sum(1 for r in resultados_independencia.values() if r['es_independiente'])
        total = len(resultados_independencia)
        
        print(f"Categorías independientes: {independientes}/{total}")
        
        if independientes == total:
            print("✅ TODAS las categorías son independientes del evento")
        else:
            print("❌ Existen dependencias entre categorías y el evento")
            
            # Mostrar las más dependientes
            dependencias = [(cat, abs(res['diferencia'])) 
                          for cat, res in resultados_independencia.items() 
                          if not res['es_independiente']]
            dependencias.sort(key=lambda x: x[1], reverse=True)
            
            print("\nCategorías con mayor dependencia:")
            for cat, diff in dependencias[:3]:
                direccion = "positiva" if resultados_independencia[cat]['diferencia'] > 0 else "negativa"
                print(f"- {cat}: {diff:.4f} (dependencia {direccion})")
        
    


    
    