import requests
import pandas as pd
import os

def DescargarDataset():
    training_request = requests.get("https://huggingface.co/api/datasets/p2pfl/CIFAR10/parquet/default/train/0.parquet")
    testing_request = requests.get("https://huggingface.co/api/datasets/p2pfl/CIFAR10/parquet/default/test/0.parquet")

    #Este bloque entero es para verificar la creacion de carpetas y hacerlas.

    class CarpetaNoCreada(Exception):
        pass

    Index =  {0 : 'AirPlane',2 : 'Bird',1 : 'Car',3 : 'Cat',4 : 'Deer',5 :'Dog',6 : 'Frog',7 : 'Horse',8 : 'Ship',9 : 'Truck'}

    try:
        os.mkdir("training_set")
    except WindowsError:
        pass
        
            
    try:
        os.mkdir("testing_set")
    except WindowsError:
        pass

    try:
        for _,value in Index.items():
            os.mkdir("./training_set/"+value)
    except WindowsError:
        pass
        
    try:
        for _,value in Index.items():
            os.mkdir("./testing_set/"+value)
    except WindowsError:
        pass

    if not os.path.isdir("./training_set"): raise CarpetaNoCreada("Carpeta training_set no pudo crear.")
    if not os.path.isdir("./testing_set"): raise CarpetaNoCreada("Carpeta testing_set no pudo crear.")

    lista_testing = os.listdir("./testing_set")
    lista_training = os.listdir("./training_set")

    for _,value in Index.items():

        
        try:
            lista_testing.index(value)
            lista_testing.remove(value)
        except ValueError:
            pass
        
        try:
            lista_training.index(value)
            lista_training.remove(value)
        except ValueError:
            pass
        
    if len(lista_testing) != 0:
        text = "No se pudieron crear las siguentes carpetas en testing/ : "
        for name in lista_testing:
            text + name + " "
        raise CarpetaNoCreada(text)

    if len(lista_training) != 0:
        text = "No se pudieron crear las siguentes carpetas en testing/ : "
        for name in lista_training:
            text + name + " "
        raise CarpetaNoCreada(text)

    if training_request.ok:
        
        with open("./training.parquet","wb") as f:
            f.write(training_request.content)
            
        training_request.close()
        
    if testing_request.ok:

        with open("./testing.parquet","wb") as f:
            f.write(testing_request.content)
            
        testing_request.close()

    test_df = pd.read_parquet("./testing.parquet")
    training_df = pd.read_parquet("./training.parquet")

    for row in test_df.itertuples():
        with open("./testing_set/"+Index[row[1]]+"/"+str(row[0])+".png","wb") as f:
            f.write(row[2])

    for row in training_df.itertuples():
        with open("./training_set/"+Index[row[1]]+"/"+str(row[0])+".png","wb") as f:
            f.write(row[2])


    del test_df
    del training_df
    os.remove("./testing.parquet")
    os.remove("./training.parquet")
    
if __name__ == "__main__":
    DescargarDataset() #Al abrir el archivo se descarga el dataset