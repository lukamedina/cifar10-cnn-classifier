from RedNeuronal.handler import Class10,NotImageType
from pathlib import Path
import shutil
import tkinter as tk
from tkinter import filedialog

print("DogsAndCats-1 un modelo CNN basico")
print("Los formatos aceptados son : jpeg,jpg y png. si no son ninguno de estos la saltara")

Model = Class10()

root = tk.Tk()
root.withdraw() 

def MainLoop():
    while True:
        Ruta = filedialog.askdirectory(
                    title="Selecciona una carpeta"
                )
        
        if Ruta.lower() == "":
            salir =  input("Exit? Si/No : ")
                        
            if salir.lower() == "si":
                exit()
                break
            
            MainLoop()
        
        if Model.CarpetasSalida(Ruta) == False:
            print("Ingrese una ruta o carpeta existente porfavor.")
            MainLoop()
        
        Ruta = Path(Ruta)
        
        for archivo in Ruta.iterdir():
            if not archivo.is_file():
                continue
            
            try:
                predic = Model.Categorice(str(archivo))
            except NotImageType:
                continue
            except Exception:
                continue #Por el momento no me interesa mucho separar las exepciones
                        
            carpeta_destino = Ruta / predic
            
            shutil.move(
                str(archivo),
                str(carpeta_destino / archivo.name)
            )
            
        print("Carpetas ordenadas.")
            
        exit_op =  input("Exit? Si/No : ")
        
        if exit_op.lower() == "si":
            exit()
            break
        
            
        
if __name__ == "__main__":
    MainLoop()
        
        