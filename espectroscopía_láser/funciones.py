
#Importar datos de los 5 .csv

#medir con el osciloscopio

# ver cómo hacer para crear las variables globales, o ver de usar diccionarios



# Para probar:

def medir_una_vez_osci():
    
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



