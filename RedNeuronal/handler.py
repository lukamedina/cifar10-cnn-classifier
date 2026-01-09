import torch
from torchvision import transforms
from PIL import Image
from RedNeuronal.Model import NeuronalNewtork
import os

Index =  [
    'AirPlane', 
    'Bird', 
    'Car', 
    'Cat', 
    'Deer', 
    'Dog', 
    'Frog', 
    'Horse', 
    'Ship', 
    'Truck'
]

trasf_fn = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5),(.5,.5,.5)) 
])

class NotImageType(Exception):
    pass


class InvalidImagePath(Exception):
    pass

def ImgTrasformFromPath(png):
    
    if not png:
        raise InvalidImagePath
    
    
    if png.find("\\") != -1:
        png.replace("\\","/")
        
    if not os.path.exists(png):
        raise InvalidImagePath
    
    _,extencion = os.path.splitext(png)
        
    match extencion:
        case ".png":
            pass
        case ".jpg":
            pass
        case ".jpeg":
            pass
        case _:
            raise NotImageType #Al parecer necesitamos verificar la extencion, basicamente si esto no es una foto tira error
            

    with Image.open(png,"r") as img:
        return trasf_fn(img)


class Class10:
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.Model = NeuronalNewtork().to(self.device)
        
        models = os.listdir("./RedNeuronal/Modelos")
        
        models = [x for x in models if os.path.splitext(x)[1] == ".pth"]
                
        if len(models) == 0:
            print("Modelo/os no encontrado/os, porfavor revise")
            input()
            exit()
            
        if len(models) == 1:
            self.Model.load_state_dict(torch.load("./RedNeuronal/Modelos/"+models[0],map_location="cpu" if not torch.cuda.is_available() else None))            
        else:
            def VariosModelos():
                print("Varios modelos encontrados:")
                for x,y in enumerate(models): print(f"Modelo {y} indice {x}")
                index = input("Seleccione modelo : ")
                
                try:
                    index = int(index)
                except Exception:
                    print("El indice debe ser un numero entero mayor o igual a 0")
                    VariosModelos()
                
                if index <= 0:
                    print("Debe ser mayor o igual a 0")
                    VariosModelos()
                    
                self.Model.load_state_dict(torch.load("./RedNeuronal/Modelos/"+models[index],map_location="cpu" if not torch.cuda.is_available() else None))            
 
            VariosModelos()
                                    
        self.Model.eval()
        total_params = sum(p.numel() for p in self.Model.parameters())
        trainable_params = sum(p.numel() for p in self.Model.parameters() if p.requires_grad)

        print("Total:", total_params)
        print("Entrenables:", trainable_params)
            
    def Categorice(self,Png):
        
        tensor_image = ImgTrasformFromPath(Png)
        tensor_image = tensor_image.to(self.device)
        tensor_image = tensor_image.unsqueeze(0) #Al parecer esto tiene ya pre-hecho que se van a cargar modelos, por ende aplicamos una pequeña trasformacion lineal para agregar un 0
        
        with torch.no_grad():
            y_result = self.Model(tensor_image)
        
        _, predicted = torch.max(y_result, 1)
            
        return Index[predicted]
    
    def CarpetasSalida(self,path):
        
        if not os.path.isdir(path):
            return False
        
        for key in Index:
            try:
                os.mkdir(os.path.join(path,key))
            except:
                pass
            
        return True