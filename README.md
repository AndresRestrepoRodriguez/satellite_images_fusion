# satellite_images_fusion
## Descripción
La fusión de imágenes satelitales es un proceso digital que permite reunir en una imagen procesada la riqueza espectral de una imagen multiespectral y la resolución espacial de una imagen pancromática. Para llevar esto a cabo se cuenta con algunos algoritmos definidos previamente como técnica de valor medio, filtro paso alto, gran Schmidt y modulación de altas frecuencias, entre otras. A continuación, se da una breve descripción de estas técnicas.
### Gram-Schmidt
La imagen pancromática se combina con el resto de las bandas de menor resolución espacial mediante una transformación matemática de los datos originales. La transformación Gram-Schmidt es una técnica común en álgebra lineal y análisis multivariante; en este caso se aplica con objeto de ortogonalizar las bandas de una imagen digital. La ortogonalización de los datos elimina la información redundante que contienen, algo bastante acusado en las imágenes de satélite, especialmente en las bandas del espectro visible. En el caso de que exista una correlación perfecta entre las bandas de entrada, la ortogonalización Gram-Schmidt produciría una banda en la que todos los píxeles tendrían valor 0. En el caso más realista de que las correlaciones entre las bandas sean muy altas, el resultado será una imagen con valores muy bajos.
### Modulación de altas frecuencias
Se trata de una variación de los denominados métodos de fusión en el dominio espacial, cuya idea es la de transferir las altas frecuencias de la imagen de alta resolución a la imagen de baja resolución. Como ya es sabido, las altas frecuencias contienen la información relativa a los detalles de una imagen y pueden extraerse mediante operaciones de filtrado o convolución. Básicamente, estos métodos consisten en la suma de las altas frecuencias de la imagen pancromática a cada banda de la imagen, la eficiencia de este filtro se basa en la existencia de una correlación radiométrica elevada entre los componentes de altas frecuencias de ambas imágenes. Un caso particular de los métodos espaciales es el llamado modulación de las altas frecuencias (HFM, High Frequency Modulation). Consiste en la multiplicación de cada banda de baja resolución (MS)i, por la imagen de alta resolución (Pan), se obtienen mejores resultados si el filtro paso-bajo se diseña de forma que se ajuste a la función de dispersión puntual (PSF, Point Spread Function) relativa entre ambas imágenes. La PSF es una función de ponderación sobre la señal electrónica que se produce a la salida de los detectores y que depende de factores ópticos, del movimiento de la imagen en su adquisición, del propio detector y de los componentes electrónicos que entran en juego durante el proceso.
### Transformación High Pass Filter - HPF
Este método consiste en añadir la información espacial de la banda pancromática a la información multiespectral de menor resolución espacial aplicando un filtro de paso alto en combinación con una operación de álgebra de mapas. El funcionamiento del algoritmo, descrito en Leica-Geosystems (2006), consta de cinco pasos:

1. Calcular el parámetro R a partir del tamaño del píxel de la capa pancromática (tpan) y de la multiespectral.

2. Aplicar un filtro de paso alto a la imagen pancromática; el tamaño de la ventana de filtrado es proporcional a R. Todos los elementos de la ventana de filtrado toman el valor -1, con la excepción del valor central. Existen tres posibilidades para este valor; el menor de los tres es el que se utiliza por defecto y es el que se ha empleado en este trabajo.

3. Remuestrear la imagen multiespectral al tamaño del píxel de la resolución espacial de la imagen filtrada.

4. Sumar la imagen filtrada a las capas multiespectrales. Pero antes, la imagen filtrada se pondera en función de la desviación típica de la imagen multiespectral y el valor de R, a este factor de ponderación se le denomina W, donde es la desviación típica de cada una de las bandas; es la desviación típica de la imagen filtrada y M es un factor que determina la intensidad en la aplicación del filtro. Donde Pout es el píxel de salida de cada una de las bandas multiespectrales ya fusionadas; Pin es el píxel de entrada de cada una de las bandas multiespectrales originales y PHPF es el píxel de la imagen filtrada.

5. Expandir linealmente los niveles digitales (ND) de la imagen multiespectral fusionada; esta operación reescala la imagen resultante, de forma que la media y la desviación típica coincida con las de la imagen original.

### Valor medio simple
El método de transformación de valor medio aplica el promedio de valor medio simple, a cada una de las combinaciones de banda de salida. Por ejemplo, la nueva banda fusionada será la suma de la banda original con la banda pancromática dividido en dos. Así sucesivamente según el número de bandas que conforme la imagen multiespectral.

### Utilidades
Por otra parte, una vez se ha aplicado alguno de los algoritmos de fusión, es necesario realizar una evaluación de la riqueza espectral y espacial de la imagen generada por los métodos de fusión expuestos anteriormente, es posible se utilizar algunos índices como mean square error (mse), root mean square error (rmse), Bias y el índice de correlación. En este orden de ideas, al implementar los algoritmos de fusión de imágenes satelitales en forma serial, es decir, realizando su ejecución exclusivamente en CPU, se presentan tiempos elevados al utilizar imágenes de dimensiones superiores, es por esto que este proyecto busca realizar la implementación de las transformadas mencionadas anteriormente mediante procesamiento heterogéneo CPU/GPU con el fin de optimizar los tiempos de ejecución para este algoritmo. Así mismo, se tiene como objetivo proporcionar herramientas para la comparación en términos de tiempos de ejecución y evaluación de la calidad de la imagen obtenida.

## Manual de Usuario

El manual de usuario realizado para este paquete, tiene como finalidad presentar la interacción paso a paso entre el usuario y sus distintas funcionalidades. El manual de usuario se encuentra disponible en la siguiente dirección : [Documento](https://github.com/AndresRestrepoRodriguez/satellite_images_fusion/blob/main/handbook/ManualUsuario_satellite-images-fusion.pdf)

## Manual Técnico

El manual técnico realizado para esta aplicación, tiene como finalidad presentar el proceso de instalación para la librería y sus distintas dependencias. El manual técnico se encuentra disponible en la siguiente dirección : [Documento](https://github.com/AndresRestrepoRodriguez/satellite_images_fusion/blob/main/handbook/ManualTecnico_satellite-images-fusion.pdf)

## Especificación de Requisitos de Software
Este documento es una Especificación de Requisitos Software (ERS) para la librería llamada “Sallfus”. Esta especificación se ha estructurado basándose en las directrices dadas por el estándar IEEE Práctica Recomendada para Especificaciones de Requisitos Software ANSI/IEEE 830, 1998. Este documento se encuentra disponible en la siguiente dirección : [Documento](https://github.com/AndresRestrepoRodriguez/satellite_images_fusion/blob/main/handbook/IEEE-830-satellite_images_fusion.pdf)

## Notebook en Colab
Con el propósito de ilustrar la forma de instalar y usar el paquete, se ha creado un Notebook en Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tuWc60ub4scOSNl3wTQ8EoI5b9E-tyNS) 

## Video Tutorial
Con el fin de presentar detalladamente el funcionamiento de la librería, se realizó un video tutorial donde se presenta la interacción entre el usuario y cada una de las funciones que pertencen a esta. El video tutorial se encuentra disponible en la siguiente dirección: [Video]()
