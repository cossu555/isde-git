import numpy as np
from sklearn.metrics import pairwise_distances

    #viene utilizzata per calcolare le distanze tra coppie di punti dati.
    # In pratica, prende in ingresso due matrici o vettori e restituisce una matrice di distanze
    #la distanza tra i punti è euclidea, ovvero traccio una linea e la misuro

def softmax(x): #svolgo la funzione su x
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

class NMC:
    def __init__(self):
        self._centroids = None #definisc i centroids

    @property
    def centroids(self):
        return self._centroids
    #il setter poi va commentato, probabilemnte per dei test
    #@centroids.setter
    #def centroids(self,value):
    #    self._centroids = value

    def fit(self, xtr,ytr): #riceve i dati di train

        n_dimensions = xtr.shape[1]                                 #prendo la dimensione dell'immagine, [0] n immagini, [1] dimensione immagine = 784
        n_classes = np.unique(ytr).size                             #prendo i valori unici di ytr e li conto -> 0-9 quindi 10
        self._centroids = np.zeros(shape=(n_classes, n_dimensions)) #creo il vettore dei centroids 10 * 786

        for k in range(n_classes): #vk va da 0-10
            self._centroids[k, :] = np.mean(xtr[ytr == k, :], axis = 0)
            #prendo da 0-10 e in ogniuno faccio la media dei pixel(784) ma solo quelli che hanno ytr==k

        return self #returna la classe

    def decision_function(self,xts): #prendo le cifre e predico cosa può essere

        if self._centroids is None: #se il modello non è stato adddestrato
            raise ValueError("Centroids have not been estimated. Call 'fit' first")

        dist = pairwise_distances(xts, self._centroids) #ho 2 matrici e verifico le distanze tra i punto

        #più sono vicini e meno grandi saranno le distanze

        sim = 1/(1e-3 + dist)
        #1e-3 è usata per evitare divisioni per 0
        #1/ stiamo invertendo la somiglianza, quindi più si somigliano e più sim è grnade

        return sim

    def predict(self, xts):
        scores = self.decision_function(xts) #predico
        ypred = np.argmax(scores,axis=1)     #prendo il più vicino
        return ypred
