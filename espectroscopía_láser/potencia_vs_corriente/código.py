#código para medir y analizar la respuesta de diodo láser utilizado
# variamos manualmente la corriente suministrada, y medimos la potencia emitida
#utilizando un power meter PM100D de Thorlabs


import matplotlib.pyplot as plt
import numpy as np
import time
import pyvisa
import pandas as pd
from scipy.optimize import curve_fit

#%%

# Crear recurso VISA
rm = pyvisa.ResourceManager()
# Listar dispositivos conectados
print("Dispositivos disponibles:", rm.list_resources())


#%%
# Abrir conexión
pm = rm.open_resource('USB0::0x1313::0x807B::16020109::INSTR')

# Identificación del dispositivo
print(pm.query("*IDN?"))

#%%

# Configurar en modo potencia en Watts
pm.write("SENSE:FUNC 'POWer'")



df = pd.DataFrame()

#%%

# Ejemplo: adquirir 10 mediciones
for i in range(10):
    mediciones = []
    power = pm.query("MEAS:POW?")
    valor = float(power.strip())  # convertir a número
    mediciones.append(valor)
    
    print(f"Medición {i+1}: {power.strip()} W")
    time.sleep(0.5)  # medio segundo entre lecturas

# Convertir barrido en Serie y añadir como columna
df["P[W], I=20e-4 A"] = pd.Series(mediciones)

#Guardar progreso en CSV (por si falla en el medio)
df.to_csv(r"C:\Users\publico\Desktop\rnes_noche\Potencia_vs_corriente\medicion_potencia.csv", index=False)

print("✅ Barrido guardado en DataFrame y CSV")






#%% Calculo la pendiente de la recta

# uso el promedio de las mediciones para cada corriente fija

corrientes = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]) * (10**(-4)) #ampere

potencias = []
err_potencias = []

df = pd.read_csv("./medicion_potencia.csv")

for i in range(21):
    columna_n = df.iloc[:, i].to_numpy()
    potencia = np.mean(columna_n)
    error = np.sqrt

    #asumo una distrib gaussiana
    media = potencia
    desv_std_muestral = np.std(columna_n, ddof=1)  # ddof=1 para usar n-1
    n = len(columna_n)
    error_potencia = desv_std_muestral / np.sqrt(n)

    potencias.append(potencia)
    err_potencias.append(error_potencia)

#%%
#Grafico y veo la parte lineal

def lineal(x, m, b):
  y =  m * x + b
  return y

param_iniciales = [0.01, 0]

#%%


x_fit_n =  np.linspace(np.min(corrientes), np.max(corrientes), 1000, endpoint = True)

A = 10
B = 20

plt.plot(corrientes[A:B], potencias[A:B], ".", color = 'darkslategray')
plt.show()


#%%
popt_0, pcov_0 = curve_fit(lineal, corrientes[A:B], potencias[A:B], p0=param_iniciales, sigma = err_potencias, absolute_sigma=True)
incertidumbre_0 = np.sqrt(np.diag(pcov_0))

tabla = pd.DataFrame({
        'Los parametros optimos son': popt_0,
        'Los errores de los parametros son': incertidumbre_0,
        })

print(tabla)









