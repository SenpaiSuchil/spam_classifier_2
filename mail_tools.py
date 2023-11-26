import os
import numpy as np
import random

def mail_extractor(dir):
    mails=[]
    for archivo in os.listdir(dir):
        with open(os.path.join(dir, archivo), "r", encoding="latin1") as file:
            contenido = file.read()
            mails.append(contenido)
    return mails


def mail_labeler(array1, array2):
    #emails = [(correo, 1) for correo in array1] + [(correo, 0) for correo in array2]
    
    #random.shuffle(emails)

    
    
    #correos = [correo[0] for correo in emails]
    correos = array1+array2
    
    etiquetas = np.concatenate((np.ones(len(array1)), np.zeros(len(array2))))

    return correos,etiquetas

    
    