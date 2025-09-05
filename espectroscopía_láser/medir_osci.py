#El código del barrido de frecuencias pero sin hacer un barrido
import matplotlib.pyplot as plt
import numpy as np
import nidaqmx
import math
import time
import pyvisa
import pandas as pd


def guardar_mediciones():
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

    # Mediciones_CH2.csv
    df_2 = pd.DataFrame([freq_gen_fun, Unidades_señales_CH2])
    df_2nuevo = pd.DataFrame(señales_CH2).T
    df_mediciones_CH2 = pd.concat([df_2, df_2nuevo], ignore_index=True)
    titulos_filas_2 = ["frecuencias generador funciones (Hz)", "Unidades señales CH2", "Señales CH2"]
    while len(titulos_filas_2) < len(df_mediciones_CH2):
        titulos_filas_2.append("")
    df_mediciones_CH2.insert(0, "Títulos de Fila", titulos_filas_2)
    df_mediciones_CH2.to_csv(r'C:.\Mediciones_CH2.csv', index=False, encoding = 'utf-8')
    time.sleep(0.5)

    #Unidades y escala.csv
    df_unidades_y_escala = pd.DataFrame([freq_gen_fun, Unidades_señales_CH1, Unidades_tiempos_CH1, Escala_señales_CH1, Escala_tiempos_CH1, Unidades_señales_CH2, Unidades_tiempos_CH2, Escala_señales_CH2, Escala_tiempos_CH2 ])
    titulos_filas_3 = ["frecuencias generador funciones (Hz)", "Unidades señales Ch1", "Unidades tiempos Ch1", "Escala señales Ch1 (V)", "Escala Tiempos Ch1 (s)", "Unidades señales Ch2", "Unidades tiempos Ch2", "Escala señales Ch2 (V)", "Escala Tiempos Ch2 (s)"]
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

    #Tiempos_CH2.csv
    df_t2 = pd.DataFrame([freq_gen_fun, Unidades_tiempos_CH2])
    df_t2nuevo = pd.DataFrame(tiempos_CH2).T
    df_tiempos_CH2 = pd.concat([df_t2, df_t2nuevo], ignore_index=True)
    titulos_filas_5 = ["frecuencias generador funciones (Hz)", "Unidades tiempos CH2", "Tiempos CH2"]
    while len(titulos_filas_5) < len(df_tiempos_CH2):
        titulos_filas_5.append("")
    df_tiempos_CH2.insert(0, "Títulos de Fila", titulos_filas_5)
    df_tiempos_CH2.to_csv(r'C:.\Tiempos_CH2.csv', index=False, encoding = 'utf-8')
    time.sleep(0.5)

    print("Archivos guardados correctamente")






#%% Itensidad en función de frecuencia

# para variar la frecuencia, variamos un voltaje a la entrada del ITC4001

rm = pyvisa.ResourceManager()
rm.list_resources()

#%%
resource_name_osciloscopio='USB0::0x0699::0x0363::C108013::INSTR'

osci = rm.open_resource(resource_name_osciloscopio)
print(osci.query('*IDN?'))



#%%
#Creamos las distintas listas donde vamos a guardar las mediciones

señales_CH1 = []
señales_CH2 = []
tiempos_CH1 = []
tiempos_CH2 = []
freq_gen_fun = []
Unidades_señales_CH1 = []
Unidades_señales_CH2 = []
Unidades_tiempos_CH1 = []
Unidades_tiempos_CH2 = []
Escala_señales_CH1 = []
Escala_señales_CH2 = []
Escala_tiempos_CH1 = []
Escala_tiempos_CH2 = []
osci.write('DATa:WIDth 1')
     

#%%

#mediciones = 10
#for freq in range(mediciones):


#Medimos canal 2
osci.write('DATa:SOUrce CH2')
#osci.write('AUTOSet EXECute')
time.sleep(1)

osci.write('DATa:ENCdg RPBinary')
xze2, xin2, yze2, ymu2, yoff2 = osci.query_ascii_values('WFMPRE:XZE?;XIN?;YZE?;YMU?;YOFf?;', separator=';')
Unidades_señales_CH2.append(osci.query('WFMPRE:YUNit?'))
Unidades_tiempos_CH2.append(osci.query('WFMPRE:XUNit?'))
Escala_señales_CH2.append(osci.query('CH2:SCAle?'))
Escala_tiempos_CH2.append(osci.query('HORizontal:MAIn:SCAle?'))

data2 = osci.query_binary_values('CURV?', datatype='B', container=np.array)
voltaje2 =(data2-yoff2)*ymu2+yze2;
tiempo2 = xze2 + np.arange(len(data2)) * xin2
señales_CH2.append(voltaje2)
tiempos_CH2.append(tiempo2)

#Medimos canal 1
osci.write('DATa:SOUrce CH1')
osci.write('DATa:ENCdg RPBinary')
xze1, xin1, yze1, ymu1, yoff1 = osci.query_ascii_values('WFMPRE:XZE?;XIN?;YZE?;YMU?;YOFf?;', separator=';')

Unidades_señales_CH1.append(osci.query('WFMPRE:YUNit?'))
Unidades_tiempos_CH1.append(osci.query('WFMPRE:XUNit?'))
Escala_señales_CH1.append(osci.query('CH1:SCAle?'))
Escala_tiempos_CH1.append(osci.query('HORizontal:MAIn:SCAle?'))

data1 = osci.query_binary_values('CURV?', datatype='B', container=np.array)
voltaje1 =(data1-yoff1)*ymu1+yze1;
tiempo1 = xze1 + np.arange(len(data1)) * xin1

señales_CH1.append(voltaje1)
tiempos_CH1.append(tiempo1)

time.sleep(1)
print("Señal adquirida")

print('FIN ( ˘▽˘)っ♨')




#%%Adquisición

#(crear de antemano los 5 .csv con encoding utf-8):
#- Mediciones_CH1.csv
#- Mediciones_CH2.csv
#- Unidades y escala.csv
#- Tiempos_CH1.csv
#- Tiempos_CH2.csv
#poner direcciónes en cada código

guardar_mediciones()


