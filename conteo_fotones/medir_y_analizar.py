import pyvisa as visa
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from scipy.signal import savgol_filter
from scipy.signal import find_peaks     

#%%

def guardar_mediciones_ch1():
    # Mediciones_CH1.csv
    df_1 = pd.DataFrame([freq_gen_fun, Unidades_señales_CH1])
    df_1nuevo = pd.DataFrame(señales_CH1).T
    df_mediciones_CH1 = pd.concat([df_1, df_1nuevo], ignore_index=True)
    titulos_filas = ["frecuencias generador funciones (Hz)", "Unidades señales CH1", "Señales CH1"]
    while len(titulos_filas) < len(df_mediciones_CH1):
        titulos_filas.append("")
    df_mediciones_CH1.insert(0, "Títulos de Fila", titulos_filas)
    df_mediciones_CH1.to_csv(r'C:.\Mediciones_CH1.csv', index=False, encoding = 'utf-8')
    time.sleep(0.5)

    #Unidades y escala.csv
    df_unidades_y_escala = pd.DataFrame([freq_gen_fun, Unidades_señales_CH1, Unidades_tiempos_CH1, Escala_señales_CH1, Escala_tiempos_CH1])
    titulos_filas_3 = ["frecuencias generador funciones (Hz)", "Unidades señales Ch1", "Unidades tiempos Ch1", "Escala señales Ch1 (V)", "Escala Tiempos Ch1 (s)"]
    df_unidades_y_escala.insert(0, "Títulos de Fila", titulos_filas_3)
    df_unidades_y_escala.to_csv(r'C:.\Unidades y escala.csv', index=False, encoding = 'utf-8')
    time.sleep(0.5)

    #Tiempos_CH1.csv
    df_t1 = pd.DataFrame([freq_gen_fun, Unidades_tiempos_CH1])
    df_t1nuevo = pd.DataFrame(tiempos_CH1).T
    df_tiempos_CH1 = pd.concat([df_t1, df_t1nuevo], ignore_index=True)
    titulos_filas_4 = ["frecuencias generador funciones (Hz)", "Unidades tiempos CH1", "Tiempos CH1"]
    while len(titulos_filas_4) < len(df_tiempos_CH1):
        titulos_filas_4.append("")
    df_tiempos_CH1.insert(0, "Títulos de Fila", titulos_filas_4)
    df_tiempos_CH1.to_csv(r'C:.\Tiempos_CH1.csv', index=False, encoding = 'utf-8')
    time.sleep(0.5)

    print("Archivos guardados correctamente")
        
    
#%% conectar osci
    
rm = visa.ResourceManager()
rm.list_resources()

#%%
resource_name_osciloscopio='USB0::0x0699::0x0363::C065092::INSTR'

osci = rm.open_resource(resource_name_osciloscopio)
print(osci.query('*IDN?'))


#%%

señales_CH1 = []
tiempos_CH1 = []
freq_gen_fun = []
Unidades_señales_CH1 = []
Unidades_tiempos_CH1 = []
Escala_señales_CH1 = []
Escala_tiempos_CH1 = []


#%%Medir
osci.write('DATa:WIDth 1')
osci.write('DATa:SOUrce CH1')
osci.write('DATa:ENCdg RPBinary')
osci.write('DATa:SOUrce CH1')
osci.write('DATa:ENCdg RPBinary')
xze1, xin1, yze1, ymu1, yoff1 = osci.query_ascii_values('WFMPRE:XZE?;XIN?;YZE?;YMU?;YOFf?;', separator=';')


mediciones = 100
for freq in range(mediciones):
        
    #Medimos canal 1
    Unidades_señales_CH1.append(osci.query('WFMPRE:YUNit?'))
    Unidades_tiempos_CH1.append(osci.query('WFMPRE:XUNit?'))
    Escala_señales_CH1.append(osci.query('CH1:SCAle?'))
    Escala_tiempos_CH1.append(osci.query('HORizontal:MAIn:SCAle?'))
    
    data1 = osci.query_binary_values('CURV?', datatype='B', container=np.array)
    voltaje1 =(data1-yoff1)*ymu1+yze1;
    tiempo1 = xze1 + np.arange(len(data1)) * xin1
    
    señales_CH1.append(voltaje1)
    tiempos_CH1.append(tiempo1)
    
    time.sleep(0.01)
    print("Señal adquirida", freq)

print('FIN ( ˘▽˘)っ♨')




#%% guardar mediciones

guardar_mediciones_ch1()



#%% Análisis

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
    data_señales_Ch1 = pd.read_csv(señales_Ch1, skiprows=3, delimiter=',', header=None, names=columnas)
    data_tiempos_Ch1 = pd.read_csv(tiempos_Ch1, skiprows=3, delimiter=',', header=None, names=columnas)
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

    data_señales = pd.read_csv(os.path.join(ruta, 'Mediciones_CH1.csv'), skiprows=3, header=None, names=columnas)
    data_tiempos = pd.read_csv(os.path.join(ruta, 'Tiempos_CH1.csv'), skiprows=3, header=None, names=columnas)
    data_unidades = pd.read_csv(os.path.join(ruta, 'Unidades y escala.csv'), header=None, names=columnas)

    return data_señales, data_tiempos, data_unidades


def analisis_rapido(n, ruta, h, dis, pro, i_graficado, x_min, x_max, graficar = True, Poisson = False):
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
        # Crear figura con dos subplots (1 fila, 2 columnas)
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # ancho 10, alto 4

        cargar_datos_2(n = n, i = i_graficado, ruta = ruta, a = "g")
        señal = voltajes1_g * (-1)
        peaks, _ = find_peaks(señal, height=h, distance=dis, prominence=pro)

        axs[0].plot(tiempos1_g, señal, label='Señal', color = 'indianred')
        axs[0].plot(tiempos1_g[peaks], señal[peaks], 'ro', label='Picos detectados')
        axs[0].legend()
        
        #axs[0].set_xlim(x_min, x_max)
        axs[0].axhline(y=h, color="cadetblue", linestyle="--", linewidth=1, label="Label vline")
        axs[0].set_title('Ejemplo medición')
        axs[0].set_xlabel('Tiempo [s]')
        axs[0].set_ylabel('Amplitud [V]')

        # --- Histograma 1: intensidades ---
        axs[1].hist(intensidades, density=True, bins=30, histtype='barstacked', alpha=0.7, color='teal', edgecolor='gray')
        axs[1].set_xlabel('Intensidad del pico')
        axs[1].set_ylabel('Frecuencia')
        axs[1].set_title('Distribución de intensidades')


        # Ajustar espacios entre gráficos
        plt.tight_layout()
        plt.show()


        if Poisson:

            fig, axs = plt.subplots(1, 1, figsize=(6, 3))
            
            # --- (1,0) Histograma de cantidad de picos ---
            axs.hist(numero_de_picos, density=True, bins=range(min(numero_de_picos), max(numero_de_picos)+2),
                    color='mediumblue', edgecolor='slateblue', align='left')
            axs.set_title('Cantidad de picos por medición')
            axs.set_xlabel('Cantidad de picos')
            axs.set_ylabel('Frecuencia')

            # Ajustar distribución de subplots
            plt.tight_layout()
            plt.show()


#%%
            
analisis_rapido(n = 1001, ruta = '/Users/Mauri/Desktop/Labo 5/Conteo de fotones/31-10/Poisson r=3kOhm', 
         h = 0.05, dis = 5, pro = 0.05, i_graficado = 5, x_min = -4e-5, x_max = 4e-5, graficar = True, Poisson = False)



