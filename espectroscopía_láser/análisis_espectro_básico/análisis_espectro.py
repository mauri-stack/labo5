#%% Análisis de las primeras mediciones del espectro del rubidio,
#Sin descomponer en polarización circ izquierda ni nada.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

#%% Importo los datos

n = 2 # número de datos tomados +1
i = 0 # número de medición -1 (empezando a contar desde 1)

# Creo un array del 0 al 50 ya que las columnas de archivo tienen esos nombres. Referencia la cantidad de frecuencias usadas. A cada numero le corresponde una frecuencia

columnas = [f'{i}' for i in range(1, n)] # donde dice 51 hay que poner la cantidad de datos que tomamos mas uno. Ej: tomamos 20 datos , ponemos 21.

# Nombro el archivo cvs que quiero referenciando la ruta del mismo.

señales_Ch1 = './Mediciones_CH1.csv'           # Referenciar la ruta de cada csv
señales_Ch2 = './Mediciones_CH2.csv'
tiempos_Ch1 = './Tiempos_CH1.csv'
tiempos_Ch2 = './Tiempos_CH2.csv'
unidades_escalas = './Unidades y escala.csv'

# Uso pandas para leer los datos y nombrar las columnas del archivo a mi conveniencia (Se puede abrir el archivo al costado a la derecha para poder guiarse)

data_señales_Ch1 = pd.read_csv(señales_Ch1, skiprows=3,  delimiter = ',', header = None, names = columnas) # Con el skiprows le decimos que comienze a leer desde la cuarta fila en este caso.
data_señales_Ch2 = pd.read_csv(señales_Ch2, skiprows=3, delimiter = ',', header = None, names = columnas)
data_tiempos_Ch1 = pd.read_csv(tiempos_Ch1, skiprows=3, delimiter = ',', header = None, names = columnas)
data_tiempos_Ch2 = pd.read_csv(tiempos_Ch2, skiprows=3, delimiter = ',', header = None, names = columnas)
data_unidades_escalas = pd.read_csv(unidades_escalas, delimiter = ',', header = None, names = columnas)


col_name = str(i+1)

freq_gen = data_unidades_escalas[col_name][1] # Pedimos la frecuencia generadora

tiempos1 = data_tiempos_Ch1[col_name].values  # Con el numero dentro del corchete referenciamos la columna que queremos tomar, en este caso la columna 1 es la de la primer frecuencia (50Hz)
voltajes1 = data_señales_Ch1[col_name].values

tiempos2 = data_tiempos_Ch2[col_name].values
voltajes2 = data_señales_Ch2[col_name].values

escala_Ch1 = float(data_unidades_escalas[col_name][4]) # Pedimos la escala de cada canal. SIEMPRE ES LA VERTICAL. porque la escala horizontal no la necesitamos.
escala_Ch2 = float(data_unidades_escalas[col_name][8])


  # Errores en los voltajes

error_Ch1 = escala_Ch1 * 10 / 256 # Una vez teniendo la escala calculamos el error que va a ser el de resolucion. Son 8 bits por lo tanto dividimos por 256
error_Ch2 = escala_Ch2 * 10 / 256

  # Selecciona solo 1 de cada N puntos para reducir la cantidad de datos
N = 1

tiempo_muestreado1 = tiempos1[::N]        # De la lista de tiempos y voltajes que tenemos , ajustar 2000 datos es muchisimo y no le da el cuero para hacerlo al curve fit.
voltaje_muestreado1 = voltajes1[::N]      # Entonces de los 2000 tomamos un solo dato cada 10 (ahi si se la banca el curve fit). Lo hacemos para ambos canales
tiempo_muestreado2 = tiempos2[::N]
voltaje_muestreado2 = voltajes2[::N]


# Graficamos ambos canales.

#plt.plot(x_fit1, sen_1(x_fit1, *popt_1), '-r',  label='Ajuste canal 1') # Curva 1 ajustada
#plt.plot(x_fit2, sen_2(x_fit2, *popt_2), '-g', label='Ajuste canal 2') # Curva 2 ajustada
plt.plot(tiempo_muestreado1, voltaje_muestreado1,'c.', label= 'Canal 1') # Esto son los datos discretos que lo activamos solo para chequear que el ajuste esta bien. Lo sacamos porque es engorroso el grafico
plt.plot(tiempo_muestreado2, voltaje_muestreado2, 'm.', label= 'Canal 2') # Esto son los datos discretos que lo activamos solo para chequear que el ajuste esta bien. Lo sacamos porque es engorroso el grafico
plt.title(f'Frecuencia: {freq_gen} Hz')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.legend(loc="upper left", bbox_to_anchor=(1, 1)) # Ponemos la legenda a un costado para que no estorbe arriba del grafico
#plt.xlim(-0.0015,-00.0016)
plt.grid()
plt.show()

#%%

#Ajusto la recta de voltajes:

def lineal(x, m, b):
  y =  m * x + b
  return y

param_iniciales = [0.01, 0]

popt_n, pcov_n = curve_fit(lineal, tiempos2, voltajes2, p0=param_iniciales, sigma = error_Ch2, absolute_sigma=True)
incertidumbre_n = np.sqrt(np.diag(pcov_n))


tabla = pd.DataFrame({
        'Los parametros optimos son': popt_n,
        'Los errores de los parametros son': incertidumbre_n,
        })

print(tabla)


#%%


x_fit_n = np.linspace(np.min(tiempos2), np.max(tiempos2), 1000, endpoint = True)

plt.errorbar(tiempos2, voltajes2, yerr= error_Ch2, fmt = ".m", ecolor = 'k', alpha = 0.2, label = 'Datos experimentales')
plt.plot(x_fit_n, lineal(x_fit_n, *popt_n), ".", color = 'darkslategray', label='Ajuste')

plt.show()

#%%

#Ahora hay que restarle los puntos de una recta con pendiente igual a la obtenida en la curva pitencia vs corriente
#(asumiendo que más voltaje = más potencia)

# Como las mediciones de potencia vs corriente salieron mal, ajusto por la recta en la región de antes
# de las líneas espectrales para restarle esa lineal a las mediciones
x_fit_n =  np.linspace(np.min(tiempos2), np.max(tiempos2), len(tiempos2), endpoint = True)

A = 400
B = 1400

plt.plot(tiempos2[A:B], voltajes1[A:B], ".", color = 'darkslategray')
plt.show()


#%%
popt_0, pcov_0 = curve_fit(lineal, tiempos2[A:B], voltajes1[A:B], p0=param_iniciales, sigma = error_Ch1, absolute_sigma=True)
incertidumbre_0 = np.sqrt(np.diag(pcov_0))

tabla = pd.DataFrame({
        'Los parametros optimos son': popt_0,
        'Los errores de los parametros son': incertidumbre_0,
        })

print(tabla)




#%%
#Grafico el espectro sin la componente que hace crecer la intensidad del láser por
#el solo hecho de que tiene más corriente

x_fit_n =  np.linspace(np.min(tiempos2), np.max(tiempos2), len(tiempos2), endpoint = True)

plt.plot(tiempos2, voltajes1 - lineal(x_fit_n, *popt_0), ".", color = 'darkslategray')
plt.xlim(0.1870)
plt.show()











#%% Calculo la pendiente de la recta (potencia vs corriente)

# uso el promedio de las mediciones para cada corriente fija

corrientes = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]) * (10**(-4)) #ampere

potencias = []

df = pd.read_csv("../medicion_potencia.csv")

for i in range(21):
    columna_n = df.iloc[:, i].to_numpy()
    potencia = np.mean(columna_n)
    potencias.append(potencia)

    
    
