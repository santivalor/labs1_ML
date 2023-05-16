# Lab 1 Henry

En el GitHub https://github.com/santivalor van a encontrar dos repositorios publicos https://github.com/santivalor/labs1.git y https://github.com/santivalor/labs1_ML.git. 

En el primero esta el codigo de las transformaciones solicitadas sobre el dataset_movies y los archivos para el deploy de FastAPI en Render. En este caso los endpoints de la FAST API son consultas que se realizan sobre el dataset luego de realizar un ETL inicial.

En labs1_ML van a encontrar el codigo completo ETL + Modelo de Machine Learning (ML) y los archivos para el deploy de FastAPI en Render. En este ultimo caso el resultado del endpoint es la recomendacion de 5 peliculas ordenadas por score de similitud una vez que el usuario ingresa una pelicula (lista de 6 peliculas = pelicula ingresada + 5 recomendaciones).
El deploy en Render del modelo de ML no se pudo realizar ya que se esta utilizando la version gratitua de Render y la implemntacion de la API requiere mas de 512 MB. Igualmente, se dejaron los archivos necesarios para el deploy en el repositorio y si se ejeuta el main.py en la maquina la API funciona de forma local.

El codigo en Python fue ejecutado en Google Colab (12 GB de RAM) por falta de capacidad en la maquina local. Se esta utilizando la version gratuita de Colab por lo que para realizar el modelo de machine learning se tuvo que filtrar el dataset movies por peliculas realizadas en Estados Unidos asi Colab no colapsaba al realizar el procesamiento de lenguaje natural.

![shutterstock_1098841148-e1536703790784](https://github.com/melisatirabassi/prueba/assets/124107756/59ab43f6-a664-46e9-b9ae-76013f9b6cf2)


# Tranformaciones

Se cargo el dataset_movies.csv a GoogleColab. Se procedio a realizar un analisis preliminar y se realizaron las siguientes transformaciones solicitadas:
Se desanidaron los campos como belongs_to_collection, production_companies y otros.
Los valores nulos de los campos revenue, budget deben son rellenados por el número 0.
Los valores nulos del campo release date se eliminaron
La columna release_date se cambio a formato fecha AAAA-mm-dd y se creo la columna release_year con el año de la fecha de estreno.
Secreo la columna return dividiendo revenue / budget, cuando no hay datos disponibles para calcularlo se toma el valor 0.
Se eliminaron las columnas que no serán utilizadas, video,imdb_id,adult,original_title,vote_count,poster_path y homepage
Una vez realizadas estas tranformaciones se guardo el nuevo dataset en Dropbox como .csv

<img width="1438" alt="Captura de pantalla 2023-05-15 a la(s) 23 31 31" src="https://github.com/melisatirabassi/prueba/assets/124107756/3135f1e7-715e-4047-8195-870a63aad216">


  # Fastapi

Con el dataset tranformado en Dropbox se creo una API con Fastapi. Con el archivo main.py ubicado en el repositorio labs1 se puede ejecutar la API cuyo endpoints son los siguientes:
def peliculas_mes(mes): '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes (nombre del mes, en str, ejemplo 'enero') historicamente''' return {'mes':mes, 'cantidad':respuesta}
def peliculas_dia(dia): '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrenaron ese dia (de la semana, en str, ejemplo 'lunes') historicamente''' return {'dia':dia, 'cantidad':respuesta}
def franquicia(franquicia): '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio''' return {'franquicia':franquicia, 'cantidad':respuesta, 'ganancia_total':respuesta, 'ganancia_promedio':respuesta}
def peliculas_pais(pais): '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo''' return {'pais':pais, 'cantidad':respuesta}
def productoras(productora): '''Ingresas la productora, retornando la ganancia total y la cantidad de peliculas que produjeron''' return {'productora':productora, 'ganancia_total':respuesta, 'cantidad':respuesta}
def retorno(pelicula): '''Ingresas la pelicula, retornando la inversion, la ganancia, el retorno y el año en el que se lanzo''' return {'pelicula':pelicula, 'inversion':respuesta, 'ganacia':respuesta,'retorno':respuesta, 'anio':respuesta}


  # Render 

En la ubicacion del archivo main.py en la maquina local se creo un archivo Dockerfile (sin extension), un entorno virtual (myenv) donde se instalaron las librerias que utiliza la API (pandas, numpy, requests, uvicorn, Fastapi, joblib, etc) y luego se genero el archivo requirements.txt haciendo freeze de myenv.
Los archivos Dockerfile, main.py y requirements.txt se cargaron en el repositorio publico labs1. Se creo un usuario en render.com y se pocedio a crear una Web Service desde el repositorio publico de GitHub (Docker).
Web Service: https://movie-etl.onrender.com/docs

<img width="1290" alt="Captura de pantalla 2023-05-15 a la(s) 23 28 30" src="https://github.com/melisatirabassi/prueba/assets/124107756/561654b4-8a2f-4978-8ead-1700ed4b13b2">


# Modelo de Machine Learning

Una vez realizadas las transformaciones en Google Colab se continuo con el codigo. Estas son algunos de los pasos implementados hasta obtener el modelo de ML:
Se eliminaron las columnas que no se utilizaron 
Se eliminaron algunas filas nulas
Se cambio el fomato a algunas de las columnas
Se filtro el dataset transformado por las peliculas realizas en EEUU. Por falta de capacidad en la nube por utilizar la version gratuita no se pudo entrenar el modelo con el el dataset completo
A las columnas numericas se les realizo una normalizacion con MinMax()
A las columnas categoricas se le realizo OneHot Encoding (dummies)
A las columnas de texto 'title' y 'overview' se las analizo con el paquete nltk. Por falta de capacidad en la nube por utilizar la version gratuita se procedio solo a analizar la columna 'title', 'overview' se elimino (se podria incluir en un futuro)
Se concateno la matriz obtenida luego del procesamiento de la columna 'title' con nltk a la matriz obtenida de las transformaciones de las columnas numericas y categoricas
Se genero un modelo de similitud con cosine_similarity con la matriz concatenada
Se guardo la matriz de similitud en Dropbox

<img width="1430" alt="Captura de pantalla 2023-05-15 a la(s) 23 25 42" src="https://github.com/melisatirabassi/prueba/assets/124107756/ce6faaee-3910-4262-aaee-696afc694482">


  # Fastapi

Con el dataset tranformado y el modelo de ML en Dropbox se creo una API con Fastapi. Con el archivo main.py ubicado en el repositorio labs1_ML se puede ejecutar la API cuyo endpoint es el siguiente:  
def recomendacion('titulo'): '''Ingresas un nombre de pelicula y te recomienda las similares en una lista de 5 valores''' return {'lista recomendada': respuesta}
La lista que esta retornando la API es de 6 valores, ya que incluye la pelicula ingresada.

<img width="1292" alt="Captura de pantalla 2023-05-15 a la(s) 23 22 48" src="https://github.com/melisatirabassi/prueba/assets/124107756/19a3a137-afaf-4959-bdf4-455b7b3f6898">


  # Render
  
En la ubicacion del archivo main.py en la maquina local se creo un archivo Dockerfile (sin extension), un entorno virtual (myenv) donde se instalaron las librerias que utiliza la API (pandas, numpy, requests, uvicorn, Fastapi, joblib, etc) y luego se genero el archivo requirements.txt haciendo freeze de myenv.
Los archivos Dockerfile, main.py y requirements.txt se cargan en el repositorio publico labs1_ML. Se creo un usuario en render.com y se pocedio a crear una Web Service desde el repositorio publico de GitHub (Docker). El deploy fallo por falta de memoria ya que se esta utilizando la version gratuita de 512 MB.

<img width="1411" alt="Captura de pantalla 2023-05-15 a la(s) 22 54 04" src="https://github.com/melisatirabassi/prueba/assets/124107756/e5c8b514-e69f-44f9-a2df-5f53af6c625c">

