import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from scipy.signal import find_peaks 
from scipy.optimize import curve_fit
import scipy.stats as st
from scipy.optimize import minimize
import scipy.special as sps
from scipy.stats import poisson
from scipy.stats import geom # Esta es la Bose-Einstein (creo)



#%%

def cargar_datos_2(n, i, ruta, a):
    
    #n número de datos tomados +1
    # i número de medición -1 (empezando a contar desde 1)
    # sufijo para cada medición distinta

    columnas = [f'{j}' for j in range(1, n)]

    # Rutas a archivos
    señales_Ch1 = os.path.join(ruta, 'Mediciones_CH1.csv')
    tiempos_Ch1 = os.path.join(ruta, 'Tiempos_CH1.csv')
    unidades_escalas = os.path.join(ruta, 'Unidades y escala.csv')

    # Lectura de archivos
    data_señales_Ch1 = pd.read_csv(señales_Ch1, skiprows=3, delimiter=',', header=None, names=columnas, usecols=range(1, n))
    data_tiempos_Ch1 = pd.read_csv(tiempos_Ch1, skiprows=3, delimiter=',', header=None, names=columnas, usecols=range(1, n))
    data_unidades_escalas = pd.read_csv(unidades_escalas, delimiter=',', header=None, names=columnas)

    # Selección de columna
    col_name = str(i + 1)

    # Variables con sufijo personalizado
    globals()[f'tiempos1_{a}'] = data_tiempos_Ch1[col_name].values
    globals()[f'voltajes1_{a}'] = data_señales_Ch1[col_name].values

    escala_Ch1 = float(data_unidades_escalas[col_name].iloc[4])

    globals()[f'error_Ch1_{a}'] = escala_Ch1 * 10 / 256


def cargar_datos(ruta, n):
    """Carga los archivos CSV una sola vez y devuelve DataFrames."""
    columnas = [f'{j}' for j in range(1, n)]

    data_señales = pd.read_csv(os.path.join(ruta, 'Mediciones_CH1.csv'), skiprows=3, header=None, names=columnas, usecols=range(1, n))
    data_tiempos = pd.read_csv(os.path.join(ruta, 'Tiempos_CH1.csv'), skiprows=3, header=None, names=columnas, usecols=range(1, n))
    data_unidades = pd.read_csv(os.path.join(ruta, 'Unidades y escala.csv'), header=None, names=columnas, usecols=range(1, n))

    return data_señales, data_tiempos, data_unidades

def mle_lambda_from_data(data):
    """
    MLE Poisson para datos individuales.
    Devuelve lambda_hat y su incertidumbre por teoría asintótica.
    """
    data = np.asarray(data)
    n = len(data)
    lam_hat = data.mean()
    sigma = np.sqrt(lam_hat / n)   # sqrt(Var(lam_hat))
    return lam_hat, sigma


def var_poisson_with_error(x):
    n = len(x)
    lam_hat = np.mean(x)        # estimador MLE de lambda
    var = np.var(x, ddof=1)      # varianza muestral

    # Error estándar del estimador de varianza s^2
    var_err = np.sqrt(lam_hat/n + 2*(lam_hat**2)/(n - 1))

    return var, var_err




def analisis_rapido(n, ruta, h, dis, pro, i_graficado,m_max, graficar = True, Poisson = False, png = False):
    """Analiza todos los canales sin recargar archivos en cada iteración."""
    
    data_señales, data_tiempos, data_unidades = cargar_datos(ruta, n)

    escala = data_unidades.iloc[4].astype(float)  # escala por columna

    intensidades = []
    numero_de_picos = []

    for i in range(0, n-1):
        col_name = str(i + 1)

        # Señal y parámetros
        señal = (-1) * data_señales[col_name].values
        escala_i = escala[col_name]
        error = escala_i * 10 / 256

        # Detección de picos
        peaks, _ = find_peaks(señal, height=h, distance=dis, prominence=pro)
        intensidades.extend(señal[peaks])
        numero_de_picos.append(len(peaks))

        # Liberar memoria de arrays grandes explícitamente
        del señal, peaks




    if graficar:
        
        for i in range(int(i_graficado), int(i_graficado) + 10):
        # Crear figura con dos subplots (1 fila, 2 columnas)
            fig, axs = plt.subplots(1, 1, figsize=(6, 4))  # ancho 10, alto 4
            
            if png:
                fig.patch.set_alpha(0)   # fondo de la figura transparente
                axs.patch.set_alpha(0)    # fondo del área de los ejes transparente
                
    
            cargar_datos_2(n = n, i = i, ruta = ruta, a = "g")
            señal = voltajes1_g * (-1)
            peaks, _ = find_peaks(señal, height=h, distance=dis, prominence=pro)
    
            axs.plot(tiempos1_g * 1000000 + tiempos1_g[-1] * 1000000, señal * 1000, label='Señal', color = 'indianred')
            axs.plot(tiempos1_g[peaks] * 1000000 + tiempos1_g[-1] * 1000000, señal[peaks] * 1000, 'ro', label='Picos')
            #axs.legend()
            
            #axs.set_ylim(-1, 10)
            #axs.axhline(y=h * 1000, color="cadetblue", linestyle="--", linewidth=2, label="Label vline")
            #axs.set_title('Ejemplo medición')
            axs.grid(alpha = 0.4)
            axs.set_xlabel('Tiempo [μs]')
            axs.set_ylabel('Amplitud [mV]')
    
            #axs.set_yticks([])        # oculta los ticks
            axs.set_yticklabels([])   # oculta los textos (redundante pero seguro)
            axs.tick_params(axis='x', labelrotation=180)
            axs.xaxis.label.set_rotation(180)
    
            # axs[0].plot(tiempos1_g, señal, label='Señal', color = 'indianred')
            # axs[0].plot(tiempos1_g[peaks], señal[peaks], 'ro', label='Picos detectados')
            # axs[0].legend()
            
            # #axs[0].set_xlim(x_min, x_max)
            # axs[0].axhline(y=h, color="cadetblue", linestyle="--", linewidth=1, label="Label vline")
            # axs[0].set_title('Ejemplo medición')
            # axs[0].set_xlabel('Tiempo [s]')
            # axs[0].set_ylabel('Amplitud [V]')
    
            # --- Histograma 1: intensidades ---
            # axs[1].hist(intensidades, density=True, bins=30, histtype='barstacked', alpha=0.7, color='teal', edgecolor='gray')
            # axs[1].set_xlabel('Intensidad del pico')
            # axs[1].set_ylabel('Frecuencia')
            # axs[1].set_yscale('log')
            # axs[1].set_title('Distribución de intensidades')
    
    
            # Ajustar espacios entre gráficos
            plt.tight_layout()
            plt.show()


    if Poisson:
    
        fig, axs = plt.subplots(1, 1, figsize=(6, 3))
        
        # --- (1,0) Histograma de cantidad de picos ---
        numero_de_picos = np.array(numero_de_picos)
        
        numero_de_picos=numero_de_picos[numero_de_picos<int(m_max)]
        
        bins = np.arange(numero_de_picos.min(), numero_de_picos.max() + 2)
        
        axs.hist(numero_de_picos, density=True, bins=bins,
                color='darkcyan', edgecolor='slateblue', align='left')
        #axs.set_title('Cantidad de picos por medición')
        axs.set_xlabel('Cantidad de picos')
        #axs.set_xlim(0,15)
        axs.set_ylabel('Frecuencia')
    
        mean=np.mean(numero_de_picos)
        std=np.std(numero_de_picos)
        
        print("Media:", f"{mean:.4f}")
        print("Varianza:", f"{std**2:.4f}")
        print("Var teórica B-E:", f"{mean**2 + mean:.4f}")
        print("Factor Fano:",  f"{(std**2)/mean:.2f}")
        
        # Ajustar distribución de subplots
        plt.tight_layout()
        plt.show()
        
    return numero_de_picos

#--------Test Kolmogorov-Smirnov------------

#El estadístico es M, la máxima diferencia entre la F teórica y la "F" medida
def ecdf(data):
    # Devuelve la función de probabilidad acumulada de mis datos, F_emp
    # y también los xs, que son los datos donde se evaúa la F_emp
    data = np.sort(data)
    xs = np.unique(data)
    F_emp = np.array([np.mean(data <= x) for x in xs])
    return xs, F_emp


def M(data, lam):
    xs, F_medida = ecdf(data)
    F_teorica = poisson.cdf(xs, lam)
    return np.max(np.abs(F_medida - F_teorica))



# función para generar la distribución de M
# Dado el lambda obtenido por el MLE, lam_hat

def generate_M_distribution(lam_hat, n, B=5000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    M_sim = np.zeros(B)
    for b in range(B):
        sim = np.random.poisson(lam_hat, size=n)
        M_sim[b] = M(sim, lam_hat)
    
    return M_sim


#calcula el M crítico
def critical_value(M_sim, alpha=0.05):
    # M_alpha tal que P(M >= M_alpha) = alpha
    return np.quantile(M_sim, 1 - alpha)



def kolmogorov_poisson_montecarlo(data, B=5000, seed=None, graficar = False):

    if seed is not None:
        np.random.seed(seed)
    
    n = len(data)
    lam_hat, lam_err = mle_lambda_from_data(data)
    
    # --- Estadístico real ---
    M_real = M(data, lam_hat)
    
    # --- Monte Carlo ---
    M_sim = np.zeros(B)
    for b in range(B):
        sim = np.random.poisson(lam_hat, size=n)
        M_sim[b] = M(sim, lam_hat)
    
    
    # Valor-p
    p_value = np.mean(M_sim >= M_real)
    
    
    M_crit = critical_value(M_sim, alpha=0.05)

    #print("λ̂ =", lam_hat)
    print(f"λ ={lam_hat:.2f} ± {lam_err:.2f}")
    #print("M_real =", M_real)
    #print("M_crit (α=0.05) =", M_crit)

    p_value = np.mean(M_sim >= M_real) #(esto es equivalente a hacer la integral desde M_real hasta infinito)

    print("p-valor", p_value)
    
    
    if p_value > 0.05:
        print("✔ No rechazo H0")
    else:

        print("❌ Rechazo H0")
    
    

    if graficar:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.hist(M_sim, density=True, bins= 'auto', histtype='barstacked', alpha=0.7, color='teal', edgecolor='gray')
        ax.axvline(M_crit, linestyle="--", color = "darkviolet" , label = "M_crítico")
        ax.axvline(M_real, linestyle="--", color = "firebrick", label = "M_medido")
        ax.set_title('Distribución de M - H0 Verdadera')
        ax.set_xlabel('Valor v.a.')
        ax.set_ylabel('Densidad de probabilidad')
        plt.legend()
        plt.show()
    
    
    
    
    return M_real, p_value, lam_hat



#------------- voy a tener que hacer las 2 funcions para Bose-Einstein también---------


def M_bose(data, eta):
    xs, F_medida = ecdf(data)
    
    p = 1/(1+eta)
    F_teorica = geom.cdf(xs, p, loc=-1)
    return np.max(np.abs(F_medida - F_teorica))
    
    
def bose_generate_M_distribution(eta_hat, n, B=5000, seed=None):
    
    if seed is not None:
        np.random.seed(seed)

    p_hat = 1/(1+eta_hat)

    M_sim = np.zeros(B)
    for b in range(B):
        sim = geom.rvs(p_hat, loc=-1, size=n)
        M_sim[b] = M_bose(sim, eta_hat)
    
    return M_sim


def kolmogorov_bose_einstein_montecarlo(data, B=5000, seed=None, graficar = False):

    if seed is not None:
        np.random.seed(seed)
    
    n = len(data)
    vals = data  
    
    #Esto caclula eta con el MLE
    k_bar = np.mean(vals)
    eta_hat = k_bar
    
    #var_eta_hat = (1 + eta_hat) / (eta_hat**3 * n)
    var_eta_hat = eta_hat*(1 + eta_hat) / n
    err_eta  = np.sqrt(var_eta_hat)
    
    
    
    # --- Estadístico real ---
    M_real = M_bose(data, eta_hat)
    
    # --- Monte Carlo ---
    p_hat = 1/(1+eta_hat)

    M_sim = np.zeros(B)
    for b in range(B):
        sim = geom.rvs(p_hat, loc=-1, size=n)
        M_sim[b] = M_bose(sim, eta_hat)
    
    

    
    # Valor-p
    p_value = np.mean(M_sim >= M_real)
    
    
    M_crit = critical_value(M_sim, alpha=0.05)

    #print("λ̂ =", lam_hat)
    print(f"η ={eta_hat:.2f} ± {err_eta:.2f}")
    #print("M_real =", M_real)
    #print("M_crit (α=0.05) =", M_crit)

    p_value = np.mean(M_sim >= M_real) #(esto es equivalente a hacer la integral desde M_real hasta infinito)

    print("p-valor", p_value)
    
    
    if p_value > 0.05:
        print("✔ No rechazo H0")
    else:

        print("❌ Rechazo H0")
    
    

    if graficar:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.hist(M_sim, density=True, bins= 'auto', histtype='barstacked', alpha=0.7, color='teal', edgecolor='gray')
        ax.axvline(M_crit, linestyle="--", color = "darkviolet" , label = "M_crítico")
        ax.axvline(M_real, linestyle="--", color = "firebrick", label = "M_medido")
        ax.set_title('Distribución de M - H0 Verdadera')
        ax.set_xlabel('Valor v.a.')
        ax.set_ylabel('Densidad de probabilidad')
        plt.legend()
        plt.show()
    
    
    
    return M_real, p_value, eta_hat



#-----------






def Ajustar_poisson(numero_de_picos, info = False, test = False, png = False, returnn = False):
        
    fig, axs = plt.subplots(1, 1, figsize=(6, 3))
    
    if png:
        fig.patch.set_alpha(0)   # fondo de la figura transparente
        axs.patch.set_alpha(0)    # fondo del área de los ejes transparente

    numero_de_picos = np.array(numero_de_picos)
    #numero_de_picos=numero_de_picos[numero_de_picos<20]
    bins = np.arange(numero_de_picos.min(), numero_de_picos.max() + 2)
    
    #calculamos los errores de los bins
    counts, bin_edges = np.histogram(numero_de_picos, bins=bins)
    
    #Centros del bin y errores
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) -0.5
    bin_err = np.sqrt(counts)
    
    #como graficamos la densidad de probabilidad (va, es discreta pero se entiende), los normalizamos
    density_counts = counts / (np.sum(counts) * np.diff(bin_edges))
    errors_density = bin_err / (np.sum(counts) * np.diff(bin_edges))
    
    
    #Graficar conteos en cada bin con barras de error
    axs.errorbar(bin_centers, density_counts, yerr=errors_density,
                 fmt='o', color='black', capsize=4, markersize=4)
    
    
    axs.hist(numero_de_picos, density=True, bins=bins,
            color='darkcyan', edgecolor='slateblue', align='left')
    axs.set_xlabel('Cuentas por ventana K')
    axs.set_ylabel('Frecuencia')
    axs.grid(alpha=0.3)
    
    
    mean=np.mean(numero_de_picos)
    std=np.std(numero_de_picos)
    
    vals = numero_de_picos
    
    
    #Esto caclula lambda con el MLE
    lam, lam_err = mle_lambda_from_data(vals)
    #print(f"λ ={lam:.2f} ± {lam_err:.2f}")
    
    #Calculamos la varianza muestral y su error
    var, var_err = var_poisson_with_error(vals)



    # --- Curva de Poisson "continua" ---
    #k_fine = np.linspace(vals.min(), vals.max(), 500)
    #pmf_fine = poisson.pmf(np.round(k_fine), lam)
    
    #axs.plot(k_fine, pmf_fine, 'r-', linewidth=2, label=f"λ ={lam:.2f} ± {lam_err:.2f}")
    
    
    # --- Curva de Poisson discreta ---
    k_vals = np.arange(vals.min(), vals.max()+1)
    pmf_vals = poisson.pmf(k_vals, lam)
    plt.plot(k_vals, pmf_vals, 'o--', color='mediumblue', label="Poisson λ MLE")
    

    #plt.scatter(k_vals, pmf_vals, color='red', zorder=10, label=f"λ ={lam:.2f} ± {lam_err:.2f}")
    #plt.plot(k_vals, pmf_vals, color='red', linestyle='--', alpha=0.5)
        
    
    if test:
        data = numero_de_picos
        M_real, p_value, lam_hat = kolmogorov_poisson_montecarlo(data, B=5000, seed=None, graficar = False)
        
        
    if info:
        
        if test==False:
            print(f"λ ={lam:.2f} ± {lam_err:.2f}")
            
        print("Media:", f"{mean:.2f}")
        print("Varianza:", f"{var:.2f} ± {var_err:.2f}")
        print("Var teórica B-E:", f"{mean**2 + mean:.2f}" )
        print("Factor Fano:",  f"{(std**2)/mean:.2f}")
    
    
    axs.legend()
    plt.tight_layout()
    plt.show()
    
    
    if returnn:
        return lam, lam_err, bin_centers, density_counts, errors_density, k_vals, pmf_vals
    


def Ajustar_Bose_Einstein(numero_de_picos, info = False, test = False, returnn = False, png = False):
        
    fig, axs = plt.subplots(1, 1, figsize=(6, 3))
    
    if png:
        fig.patch.set_alpha(0)   # fondo de la figura transparente
        axs.patch.set_alpha(0)    # fondo del área de los ejes transparente


    #numero_de_picos=numero_de_picos[numero_de_picos<20]
    bins = np.arange(numero_de_picos.min(), numero_de_picos.max() + 2)
    
    
    #calculamos los errores de los bins
    counts, bin_edges = np.histogram(numero_de_picos, bins=bins)
    
    #Centros del bin y errores
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) -0.5
    bin_err = np.sqrt(counts)
    
    #como graficamos la densidad de probabilidad (va, es discreta pero se entiende), los normalizamos
    density_counts = counts / (np.sum(counts) * np.diff(bin_edges))
    errors_density = bin_err / (np.sum(counts) * np.diff(bin_edges))
    
    
    #Graficar conteos en cada bin con barras de error
    axs.errorbar(bin_centers, density_counts, yerr=errors_density,
                 fmt='o', color='black', capsize=4, markersize=4)
    
    
    axs.hist(numero_de_picos, density=True, bins=bins,
            color='darkcyan', edgecolor='slateblue', align='left')
    axs.set_xlabel('Cuentas por ventana K')
    axs.grid(alpha = 0.4)
    axs.set_ylabel('Frecuencia')
    
    
    
    
    mean=np.mean(numero_de_picos)
    std=np.std(numero_de_picos)
    
    
    vals = numero_de_picos
    n = len(numero_de_picos)
    
    
    #Esto caclula eta con el MLE
    k_bar = np.mean(vals)
    eta_hat = k_bar
    
    #var_eta_hat = (1 + eta_hat) / (eta_hat**3 * n)
    var_eta_hat = eta_hat*(1 + eta_hat) / n
    err_eta  = np.sqrt(var_eta_hat)



    #Calculamos la varianza muestral y su error
    var, var_err = var_poisson_with_error(vals)



    # --- Curva de Poisson "continua" ---
    #k_fine = np.linspace(vals.min(), vals.max(), 500)
    #pmf_fine = poisson.pmf(np.round(k_fine), lam)
    
    #axs.plot(k_fine, pmf_fine, 'r-', linewidth=2, label=f"λ ={lam:.2f} ± {lam_err:.2f}")
    
    
    # --- Curva de Bose-Einstein discreta ---
    k_vals = np.arange(vals.min(), vals.max()+1)
    # parámetro "p" para scipy
    p_hat = 1 / (1 + eta_hat)
    # pmf según scipy, ajustada para empezar en 0
    pmf_vals = geom.pmf(k_vals, p_hat, loc=-1)
    
    plt.plot(k_vals, pmf_vals, 'o--', color='mediumblue', label="Bose-Einstein $\eta$ MLE")
    


    #plt.scatter(k_vals, pmf_vals, color='red', zorder=10, label=f"λ ={lam:.2f} ± {lam_err:.2f}")
    #plt.plot(k_vals, pmf_vals, color='red', linestyle='--', alpha=0.5)
        
    
    if test:
        data = numero_de_picos
        M_real, p_value, eta_hat = kolmogorov_bose_einstein_montecarlo(data, B=5000, seed=None, graficar = False)
        
        
    if info:
        
        if test==False:
            print(f"η ={eta_hat:.2f} ± {err_eta:.2f}")
            
        print("Media:", mean)
        print("Varianza:", f"{var:.2f} ± {var_err:.2f}")
        print("Var teórica B-E:", mean**2 + mean)
        print("Factor Fano:",  f"{(std**2)/mean:.2f}")
    
    
    axs.legend()
    plt.tight_layout()
    plt.show()
    
    if returnn:
        
        return k_vals, pmf_vals, bin_centers, density_counts, errors_density
    
    
    
    
    
    
    
#%%------ Cosas eficiciencia


p_180 = analisis_rapido(n = 1001, ruta ='/Users/Mauri/Desktop/Labo 5/Conteo de fotones/7-11/r = 50Ohm (Poisson)/segundas mil', 
         h = 0.005, dis = 2, pro = 0, i_graficado = 17, m_max = 140,
         graficar = True, Poisson = True, png = True)

#Ajustar_poisson(p_180, test = True, info = False, png = True)




#%%Gráfico Estadísitica


est = analisis_rapido(n = 1001, ruta ='/Users/Mauri/Desktop/Labo 5/Conteo de fotones/14-11/Variamos intensidad a ventana 25microsegundos/theta = 160/', 
         h = 0.002, dis = 2, pro = 0, i_graficado = 17, m_max = 140,
         graficar = False, Poisson = True, png = False)

Ajustar_poisson(est, test = True, info = False, png = False)


#%%
p_180 = analisis_rapido(n = 1001, ruta ='/Users/Mauri/Desktop/Labo 5/Conteo de fotones/14-11/Variamos intensidad a ventana 25microsegundos/theta = 180', 
         h = 0.002, dis = 2, pro = 0, i_graficado = 7, m_max = 100,
         graficar = False, Poisson = True)

Ajustar_poisson(p_180, test = True, info = True)




    
    
    
    
#%%----- Láser Poisson ---------
    

p_180 = analisis_rapido(n = 1001, ruta ='/Users/Mauri/Desktop/Labo 5/Conteo de fotones/14-11/Variamos intensidad a ventana 25microsegundos/theta = 180', 
         h = 0.002, dis = 2, pro = 0, i_graficado = 7, m_max = 100,
         graficar = False, Poisson = True, png = False)


Ajustar_poisson(p_180, test = True, info = True, png = True)









#%%------ Láser Bose-Einstein -------
    

bose_3 = analisis_rapido(n = 701, ruta ='/Users/Mauri/Desktop/Labo 5/Conteo de fotones/7-11/Bose-Einstein (r= 50Ohm)/Ventana osc variable (5V)/2,5 us', 
         h = 0.005, dis = 7, pro = 0, i_graficado = 27, m_max = 10,
         graficar = False, Poisson = True)

kolmogorov_bose_einstein_montecarlo(bose_3, B=5000, seed=None, graficar = True)


Ajustar_Bose_Einstein(bose_3, info = True, test = True, png = True)


#%%
eta = 0.75
eta_err = 0.04

var_teórica_bose = eta**2 + eta
err_var_teórica_bose = np.abs(2*eta + 1) * eta_err


print("Varianza_teórica_bose:", f"{var_teórica_bose:.2f} ± {err_var_teórica_bose:.2f}")

#checkeo de que no sea poissoniana
kolmogorov_poisson_montecarlo(bose_3, B=5000, seed=None, graficar = True)

Ajustar_poisson(bose_3, test = True, info = True)






#%% Led Poisson solo -------

led_p2 = analisis_rapido(n = 701, ruta ='/Users/Mauri/Desktop/Labo 5/Conteo de fotones/28-11/Led/Poisson/2.6V', 
         h = -0.0005, dis = 7, pro = 0, i_graficado = 7, m_max = 100,
         graficar = False, Poisson = True)

lam_2, lam_err_2, bins_2, cuentas_2, err_cuentas_2, bin_ajuste_2, cuentas_ajuste_2 = Ajustar_poisson(led_p2, test = True, info = False, returnn = True, png = True)






#%%----- Led Poisson variación intensidad ---------


#%%

led_p1 = analisis_rapido(n = 701, ruta ='/Users/Mauri/Desktop/Labo 5/Conteo de fotones/28-11/Led/Poisson/2.5V', 
         h = -0.0005, dis = 7, pro = 0, i_graficado = 7, m_max = 200,
         graficar = False, Poisson = True)

#%%

led_p2 = analisis_rapido(n = 701, ruta ='/Users/Mauri/Desktop/Labo 5/Conteo de fotones/28-11/Led/Poisson/2.6V', 
         h = -0.0005, dis = 7, pro = 0, i_graficado = 7, m_max = 200,
         graficar = False, Poisson = True)



#%%

led_p3 = analisis_rapido(n = 701, ruta ='/Users/Mauri/Desktop/Labo 5/Conteo de fotones/28-11/Led/Poisson/2.7V', 
         h = -0.0005, dis = 7, pro = 0, i_graficado = 7, m_max = 200,
         graficar = False, Poisson = True)


#%%

led_p4 = analisis_rapido(n = 701, ruta ='/Users/Mauri/Desktop/Labo 5/Conteo de fotones/28-11/Led/Poisson/2.8V', 
         h = -0.0005, dis = 7, pro = 0, i_graficado = 7, m_max = 200,
         graficar = False, Poisson = True)

#%%

led_p5 = analisis_rapido(n = 701, ruta ='/Users/Mauri/Desktop/Labo 5/Conteo de fotones/28-11/Led/Poisson/2.9V', 
         h = -0.0005, dis = 7, pro = 0, i_graficado = 7, m_max = 200,
         graficar = False, Poisson = True)




#%%


lam_1, lam_err_1, bins_1, cuentas_1, err_cuentas_1, bin_ajuste_1, cuentas_ajuste_1 = Ajustar_poisson(led_p1, test = False, info = False, returnn = True)

lam_2, lam_err_2, bins_2, cuentas_2, err_cuentas_2, bin_ajuste_2, cuentas_ajuste_2 = Ajustar_poisson(led_p2, test = False, info = False, returnn = True)

lam_3, lam_err_3, bins_3, cuentas_3, err_cuentas_3, bin_ajuste_3, cuentas_ajuste_3 = Ajustar_poisson(led_p3, test = False, info = False, returnn = True)

lam_4, lam_err_4, bins_4, cuentas_4, err_cuentas_4, bin_ajuste_4, cuentas_ajuste_4 = Ajustar_poisson(led_p4, test = False, info = False, returnn = True)



#%%

fig, axs = plt.subplots(1, 1, figsize=(6, 3))

fig.patch.set_alpha(0)   # fondo de la figura transparente
axs.patch.set_alpha(0)    # fondo del área de los ejes transparente


colormap = plt.cm.viridis  # plasma, magma, viridis, inferno, etc.
colors = [colormap(i / (4 - 1)- 0.3) for i in range(4)]


axs.errorbar(bins_1, cuentas_1, yerr=err_cuentas_1, fmt='o', color=colors[0], capsize=2, markersize=4)
axs.errorbar(bins_2, cuentas_2, yerr=err_cuentas_2, fmt='o', color=colors[1], capsize=2, markersize=4)
axs.errorbar(bins_3, cuentas_3, yerr=err_cuentas_3, fmt='o', color=colors[2], capsize=2, markersize=4)
axs.errorbar(bins_4, cuentas_4, yerr=err_cuentas_4, fmt='o', color=colors[3], capsize=2, markersize=4)


#plt.plot(bin_ajuste_1, cuentas_ajuste_1, 'o--', color=colors[0], label=f"λ = {lam_1:.2f} ± {lam_err_1:.2f}", markersize=2)
#plt.plot(bin_ajuste_2, cuentas_ajuste_2, 'o--', color=colors[1], label=f"λ = {lam_2:.2f} ± {lam_err_2:.2f}", markersize=2)
#plt.plot(bin_ajuste_3, cuentas_ajuste_3, 'o--', color=colors[2], label=f"λ = {lam_3:.2f} ± {lam_err_3:.2f}", markersize=2)
#plt.plot(bin_ajuste_4, cuentas_ajuste_4, 'o--', color=colors[3], label=f"λ = {lam_4:.2f} ± {lam_err_4:.2f}", markersize=2)

plt.plot(bin_ajuste_1, cuentas_ajuste_1, 'o--', color=colors[0], label="25 mA", markersize=2)
plt.plot(bin_ajuste_2, cuentas_ajuste_2, 'o--', color=colors[1], label="26 mA", markersize=2)
plt.plot(bin_ajuste_3, cuentas_ajuste_3, 'o--', color=colors[2], label="27 mA", markersize=2)
plt.plot(bin_ajuste_4, cuentas_ajuste_4, 'o--', color=colors[3], label="28 mA", markersize=2)



#axs.set_title('LED al variar la intensidad')
axs.set_xlabel('Cuentas por ventana K')
axs.set_ylabel('Frecuencia')

plt.grid(alpha = 0.2)

plt.legend(framealpha=0)
plt.show()




#%%

fig, axs = plt.subplots(1, 1, figsize=(6, 3))
    

bins_1 = np.arange(led_p1.min(), led_p1.max() + 2)
bins_2 = np.arange(led_p2.min(), led_p2.max() + 2)
bins_3 = np.arange(led_p3.min(), led_p3.max() + 2)
bins_4 = np.arange(led_p4.min(), led_p4.max() + 2)
bins_5 = np.arange(led_p5.min(), led_p5.max() + 2)


axs.hist(led_p1, density=True, bins=bins_1,color='darkcyan', edgecolor='slateblue', align='left')
axs.hist(led_p2, density=True, bins=bins_2,color='firebrick', edgecolor='slateblue', align='left')
axs.hist(led_p3, density=True, bins=bins_3,color='darkviolet', edgecolor='slateblue', align='left')
axs.hist(led_p4, density=True, bins=bins_4,color='pink', edgecolor='slateblue', align='left')
#axs.hist(led_p5, density=True, bins=bins_5,color='pink', edgecolor='slateblue', align='left')



#axs.set_title('Cantidad de picos por medición')
axs.set_xlabel('Cantidad de picos')
axs.set_ylabel('Frecuencia')



# Ajustar distribución de subplots
plt.tight_layout()
plt.show()



#%%

corrientes = [25, 26, 27, 28, 29] #mA

medias = [1.5, 4.14, 6.38, 8.95, 8.81]
err_medias = [0.05, 0.08, 0.1, 0.11, 0.11]

varianzas = [1.76, 4.27, 6.59, 8.71, 21.88]
err_varianzas = [0.09, 0.23, 0.35, 0.49, 0.48]


fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    
fig.patch.set_alpha(0)   # fondo de la figura transparente
ax.patch.set_alpha(0)    # fondo del área de los ejes transparente

#ax.errorbar(corrientes, medias,  yerr=err_medias, fmt="x",  label="Medias")
#ax.errorbar(corrientes, varianzas,  yerr=err_varianzas, fmt="s", markerfacecolor="none", markeredgecolor="k",  label="Varianzas")

ax.scatter(corrientes, medias, marker="x", color = "black",  label="Medias")
ax.scatter(corrientes, varianzas,marker="s", facecolors="none",edgecolors="k",label="Varianzas")


ax.set_xlabel('Corriente [mA]')

plt.grid(alpha = 0.2)
plt.legend(framealpha=0)
plt.show()


#%% Ajuste con las primeras 4


def lineal(x, m, b):
  y =  m*x + b
  return y

corrientes_ = np.array([25, 26, 27, 28]) #mA

medias_ = np.array([1.5, 4.14, 6.38, 8.95])
err_medias_ = np.array([0.05, 0.08, 0.1, 0.11])



param_iniciales = [0.5, 1]

popt_n, pcov_n = curve_fit(lineal, corrientes_, medias_, p0=param_iniciales, sigma = err_medias_, absolute_sigma=True)
incertidumbre_n = np.sqrt(np.diag(pcov_n))


tabla = pd.DataFrame({
        'Los parámetros óptimos son': popt_n,
        'Los errores de los parámetros son': incertidumbre_n,
        })

print(tabla)

fig, axs = plt.subplots(1, 1, figsize=(6, 3))

fig.patch.set_alpha(0)   # fondo de la figura transparente
axs.patch.set_alpha(0)    # fondo del área de los ejes transparente

x_fit = np.linspace(corrientes_[0], corrientes_[-1], 1000)

axs.errorbar(corrientes_, medias_,  yerr=err_medias_, fmt=".k",  label="λ Poisson")
axs.plot(x_fit, lineal(x_fit, *popt_n), "-b")

axs.set_xlabel('Corrientes [mA]')
axs.set_ylabel('λ')

plt.legend()
plt.show()




#Test de hipótesis
puntos = len(corrientes_)
params = len(popt_n)
y = medias_
yerr = err_medias_
y_modelo = lineal(corrientes_,popt_n[0],popt_n[1])
gl = puntos - params #grados de libertad (df)

#estimador S ("chi cuadrado"...)
S_medido = np.sum(((y-y_modelo)/yerr)**2)
p_valor = st.chi2.sf(S_medido, puntos - params)


print('p-valor:', p_valor)


#%%

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    
fig.patch.set_alpha(0)   # fondo de la figura transparente
ax.patch.set_alpha(0)    # fondo del área de los ejes transparente

#ax.errorbar(corrientes, medias,  yerr=err_medias, fmt="x",  label="Medias")
#ax.errorbar(corrientes, varianzas,  yerr=err_varianzas, fmt="s", markerfacecolor="none", markeredgecolor="k",  label="Varianzas")

ax.scatter(corrientes[:-1], medias[:-1], marker="x", color = "black",  label="Medias")
ax.scatter(corrientes[:-1], varianzas[:-1],marker="s", facecolors="none",edgecolors="k",label="Varianzas")

#ax.errorbar(corrientes_, medias_,  yerr=err_medias_, fmt=".k",  label="Ajuste")
ax.plot(x_fit, lineal(x_fit, *popt_n), "-b",  label="Ajuste")

ax.set_xlabel('Corriente [mA]')

plt.grid(alpha = 0.2)
plt.legend(framealpha=0)
plt.show()










#%%-------- Led - Bose-Einstein ------------



led_b3 = analisis_rapido(n = 701, ruta ='/Users/Mauri/Desktop/Labo 5/Conteo de fotones/28-11/Led/Bose/250 museg', 
         h = -0.0005, dis = 7, pro = 0, i_graficado = 7, m_max = 200,
         graficar = False, Poisson = True)


led_b3_god = analisis_rapido(n = 701, ruta ='/Users/Mauri/Desktop/Labo 5/Conteo de fotones/28-11/Led/Bose/250 museg', 
         h = -0.0005, dis = 7, pro = 0, i_graficado = 7, m_max = 10,
         graficar = False, Poisson = True)



#%%
kolmogorov_bose_einstein_montecarlo(led_b3, B=5000, seed=None, graficar = True)


#%%

Ajustar_Bose_Einstein(led_b3_god, info = True, test = True)


#%%

k_vals, pmf_vals, bin_centers, density_counts, errors_density = Ajustar_Bose_Einstein(led_b3_god, info = True, test = True, returnn = True)


#%%


m_max = 50

N_sub = len(led_b3_god)
N_tot = len(led_b3)

numero_de_picos = led_b3

fig, axs = plt.subplots(1, 1, figsize=(6, 3))

fig.patch.set_alpha(0)   # fondo de la figura transparente
axs.patch.set_alpha(0)    # fondo del área de los ejes transparente


# --- (1,0) Histograma de cantidad de picos ---
numero_de_picos = np.array(numero_de_picos)

numero_de_picos=numero_de_picos[numero_de_picos<int(m_max)]

bins = np.arange(numero_de_picos.min(), numero_de_picos.max() + 2)

ajuste_rescalado = pmf_vals * (N_sub / N_tot)
density_counts_reescalado = density_counts * (N_sub / N_tot)
errors_density_reescalado = errors_density * (N_sub / N_tot)

axs.hist(numero_de_picos, density=True, bins=bins,
        color='darkcyan', edgecolor='slateblue', align='left')

axs.plot(k_vals, ajuste_rescalado, 'o--', color='mediumblue', label="Bose-Einstein $\eta$ MLE")

#Graficar conteos en cada bin con barras de error
axs.errorbar(bin_centers, density_counts_reescalado, yerr=errors_density_reescalado,
             fmt='o', color='black', capsize=4, markersize=4)



#axs.set_title('Cantidad de picos por medición')
axs.set_xlabel('Cuentas por ventana K')
axs.set_xlim(-3,47)
axs.set_ylim(0,0.72)

axs.set_ylabel('Frecuencia')


axs.axvspan(15, 50, color='palevioletred', alpha=0.3, label='Mediciones descartadas')



axs.grid(True, alpha=0.5)

axs.legend(framealpha=0)
# Ajustar distribución de subplots
plt.tight_layout()
plt.show()















