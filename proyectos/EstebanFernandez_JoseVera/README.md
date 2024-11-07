# Proyecto final para la materia de Matemática Aplicada basado en el artículo 
> Vashishtha, S., & Susan, S. (2019). Fuzzy rule based unsupervised sentiment analysis from social media posts. Expert Systems with Applications, 138, 112834.
> [Link](https://www.researchgate.net/profile/Srishti-Vashishtha-2/publication/334622166_Fuzzy_Rule_based_Unsupervised_Sentiment_Analysis_from_Social_Media_Posts/links/5ece42174585152945149e5b/Fuzzy-Rule-based-Unsupervised-Sentiment-Analysis-from-Social-Media-Posts.pdf)

## Integrantes
- **Esteban Gabriel Fernández Arrúa**
- **José Sebastián Vera Arrieta**

## Ejecución
Primeramente instalar las dependencias necesarias para el proyecto
```shell
pip install -r requirements.txt
```
Luego ejecutar el archivo `main.py`, previamente se puede editar las variables de configuración para utilizar otro dataset o cambiar el path de salida
```shell
python main.py
```
## Resultados
- Los resultados procesados se guardarán por defecto en `data/result.csv`.
- Por consola se imprimirán los cálculos de métricas como tiempo promedio de ejecución e inferencia, conteo de tweets positivos, negativos y neutrales y la ubicación de los archivos generados de resultados y benchmarks.
