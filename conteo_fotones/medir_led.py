
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
    df_mediciones_CH1.to_csv(r'.\Mediciones_CH1.csv', index=False, encoding = 'utf-8')
    time.sleep(0.5)

    #Unidades y escala.csv
    df_unidades_y_escala = pd.DataFrame([freq_gen_fun, Unidades_señales_CH1, Unidades_tiempos_CH1, Escala_señales_CH1, Escala_tiempos_CH1])
    titulos_filas_3 = ["frecuencias generador funciones (Hz)", "Unidades señales Ch1", "Unidades tiempos Ch1", "Escala señales Ch1 (V)", "Escala Tiempos Ch1 (s)"]
    df_unidades_y_escala.insert(0, "Títulos de Fila", titulos_filas_3)
    df_unidades_y_escala.to_csv(r'.\Unidades y escala.csv', index=False, encoding = 'utf-8')
    time.sleep(0.5)

    #Tiempos_CH1.csv
    df_t1 = pd.DataFrame([freq_gen_fun, Unidades_tiempos_CH1])
    df_t1nuevo = pd.DataFrame(tiempos_CH1).T
    df_tiempos_CH1 = pd.concat([df_t1, df_t1nuevo], ignore_index=True)
    titulos_filas_4 = ["frecuencias generador funciones (Hz)", "Unidades tiempos CH1", "Tiempos CH1"]
    while len(titulos_filas_4) < len(df_tiempos_CH1):
        titulos_filas_4.append("")
    df_tiempos_CH1.insert(0, "Títulos de Fila", titulos_filas_4)
    df_tiempos_CH1.to_csv(r'.\Tiempos_CH1.csv', index=False, encoding = 'utf-8')
    time.sleep(0.5)

    print("Archivos guardados correctamente")
        
    
#%% conectar osci
    
rm = visa.ResourceManager()
rm.list_resources()

#%%
resource_name_osciloscopio='USB0::0x0699::0x0363::C065092::INSTR'
resource_name_fuente='USB::0x0699::0x0368::C033542::INSTR'

osci = rm.open_resource(resource_name_osciloscopio)
fuente = rm.open_resource(resource_name_fuente)

print(osci.query('*IDN?'))
print(fuente.query('*IDN?'))

#%%------ Probar LED --------



# Mandarle una corriente deseada
# Mandarle corrientes deseadas en un for


#for freq in np.linspace(1534, 1536, 3):
#   gen.write('SOURce1:FREQuency {:f}Hz'.format(freq) )


#for freq in range(mediciones):
    
    #Cambiamos i
    #i_actual = i_arr[freq] 
    
    #fuente.write('SOURce1:FREQuency {}Hz'.format(i_actual) # si freq es entero esto debería funcionar

    #fuente.write('SOURce1:FREQuency {:f}Hz'.format(freq)
    
    #fuente.write(f"SOURce1:FREQuency {freq}Hz")
    



#%%



#%%

#array con realizaciones de una V.A. con distribución exponencial,
# que no se pasa de i maz
 
lam = 1.0  # λ en unidades 1/A (ajustar)
i_max = 10
n = 1000
U = np.random.rand(n)
i_arr = -np.log(1 - U*(1 - np.exp(-lam * i_max))) / lam

#fig, ax = plt.subplots(1, 1, figsize=(6, 3))
#ax.hist(i_arr, density=True, histtype='barstacked', alpha=0.7, color='teal', edgecolor='gray')

#plt.show()




#%%

señales_CH1 = []
tiempos_CH1 = []
freq_gen_fun = []
Unidades_señales_CH1 = []
Unidades_tiempos_CH1 = []
Escala_señales_CH1 = []
Escala_tiempos_CH1 = []


#%%Medir

lam = 1.0  # λ en unidades 1/A (ajustar)
i_max = 10
n = 1000
U = np.random.rand(n)
i_arr = -np.log(1 - U*(1 - np.exp(-lam * i_max))) / lam


osci.write('DATa:WIDth 1')
osci.write('DATa:SOUrce CH1')
osci.write('DATa:ENCdg RPBinary')
osci.write('DATa:SOUrce CH1')
osci.write('DATa:ENCdg RPBinary')
xze1, xin1, yze1, ymu1, yoff1 = osci.query_ascii_values('WFMPRE:XZE?;XIN?;YZE?;YMU?;YOFf?;', separator=';')


mediciones = 1000
for freq in range(mediciones):
    
    #Cambiamos i
    i_actual = i_arr[freq] 
    
    #fuente.write('SOURce1:FREQuency {}Hz'.format(i_actual) # si freq es entero esto debería funcionar

    #fuente.write('SOURce1:FREQuency {:f}Hz'.format(freq)
    
    #fuente.write(f"SOURce1:FREQuency {freq}Hz")
    
    
    #--------
    
        
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










