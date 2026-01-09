# Sobre el Proyecto

Este proyecto consiste en el desarrollo de una **Red Neuronal Convolucional (CNN)** entrenada desde cero para la clasificación de imágenes en **10 categorías**.  
El modelo alcanza aproximadamente **80–86% de accuracy**, priorizando la **generalización** y la mitigación del **overfitting** mediante una correcta separación de datos y regularización.

El objetivo principal es **educativo**, buscando implementar un pipeline completo de *machine learning*:  
descarga de datos, preprocesamiento, entrenamiento, evaluación y aplicación práctica.

Como caso de uso, el modelo permite **ordenar automáticamente una carpeta de imágenes**, clasificándolas en subcarpetas correspondientes a cada categoría.

---

## Estructura del Proyecto

- `model.py` ->  definición de la arquitectura de la CNN  
- `train.ipynb` ->  entrenamiento del modelo y sanity check  
- `eval.ipynb` ->  evaluación final sobre un conjunto de test independiente  
- `download_data.py` ->  descarga y preparación del dataset  
- `.bat` scripts ->  automatización del setup y ejecución (Windows)
- `ModelHadle.py` ->  definicion de la clase "Class10" para uso practic
---

## Setup

Crear y activar el entorno virtual (Windows):

```bash
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
```
## Dataset
El dataset utilizado es CIFAR-10, descargado automáticamente mediante el script (Windows):

```bash
call venv\Scripts\activate
cd RedNeuronal
python download_data.py
```

## Creditos

Gracias a huggingface y p2pfl proporicionando el dataset.
<a>https://huggingface.co/datasets/p2pfl/CIFAR10</a>

