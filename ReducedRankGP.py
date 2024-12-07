#!/usr/bin/python3

import numpy as np
from scipy.special import kv, gamma
from scipy.linalg import cholesky, cho_solve
from scipy import linalg
from numba import njit, jit, prange, cuda, float64
import time

@njit(parallel=True)
def compute_eigenfunctions(xy, sortedOrder, boundaryL, inputDim, numE):

    ret = np.zeros((numE, xy.shape[0]))
    boundaryL_inv_sqrt = (1 / np.sqrt(boundaryL))**inputDim

    for idx in prange(numE):
        i = sortedOrder[idx, 0]
        j = sortedOrder[idx, 1]
        ret[idx, :] = boundaryL_inv_sqrt * np.sin(np.pi * (i + 1) * (xy[:, 0] + boundaryL) / (2 * boundaryL)) * np.sin(np.pi * (j + 1) * (xy[:, 1] + boundaryL) / (2 * boundaryL))
    
    return ret
 
class ReducedRankGP:
    def __init__(self, config):
      self.boundaryL = config['boundaryL']
      self.numE = config['numE'] # the number of elements of E
      self.sensorNoise = config['sensorNoise']
      self.inputDim = config['inputDim']        
      
      self.trainInputMat = np.zeros((self.numE, self.numE))
      self.trainOutputVec = np.zeros((self.numE, 1))
      
      ########### Compute eigenvalues and init eigenfunctions ###########
      # Spectral density matrix of matern kernel could be initialized in advance
      self.buildSpectDensityMat(config['covFunc'], config['kernParams'])

    def modelUpdate(self, trainXY, trainZ):
      """
      inputs: trainXY (numTrain, inputDim)
              trainZ (numTrain, 1)
              predXY (numPred, inputDim)
      outputs: predZ (numPred, 1)
      """
      # Compute eigenfunctions
      # print("Computing eigenfunctions...")
      start = time.time()
      trainPhi = self.buildEigenfunctions(trainXY) # (numE, numTrain)
      end = time.time()
      # print("Elapsed time: ", end - start)
            
      self.trainInputMat += np.dot(trainPhi, trainPhi.T) # (numE, numE)
      self.trainOutputVec += trainPhi@trainZ # (numE x numTrain) x (numTrain x 1) -> (numE, 1) 
      
      # print("Computing midTerm...")
      start = time.time()
      tempMat= self.trainInputMat + self.sensorNoise * self.inverseSpectDensity
      LMat = cholesky(tempMat, lower=True)
      self.midTerm = cho_solve((LMat, True), np.eye(self.numE))
      end = time.time()
      # print("Elapsed time: ", end - start)
      
    def predict(self, testXY, predPhi = None):
      """
      inputs: testXY (numTest, inputDim)
      outputs: predZ (numTest, 1)
      """
      if predPhi is None:
        predPhi = self.buildEigenfunctions(testXY)
      
      mean = predPhi.T @ (self.midTerm@self.trainOutputVec) # (numTest, 1)
      var = self.sensorNoise * np.diag(np.dot(predPhi.T, np.dot(self.midTerm,predPhi))) # (numTest, numTest)
      return mean, var
    
    def buildSpectDensityMat(self, conFunc, kernParams):
      index = np.arange(0,500) # the number of elements of E
      index = index.astype(np.uint64)
      index = index.reshape(-1,1)

      ########### Compute eigenvalues using Hilbert space method ###########
      eigenValues = (np.pi * (index + 1) / (2*self.boundaryL))**2
      eigenValues = eigenValues.reshape(-1,1) # (numE, 1)
      frequency = np.sqrt(eigenValues) # (numE, 1)
      
      # Combination of integer orders 
      # sortedFreq, self.sortedOrder = self.sortFrequency(frequency, index) # (numE, 1) , (numE, inputDim)
      spectDensity = self.SpectralDensityFunc(conFunc, kernParams, frequency) # (numE, 1)
      sortedDensity, self.sortedOrder = self.sortSpecDensity(spectDensity, index)
      self.spectDensity = np.diag(sortedDensity[:,0]) # (numE, numE)
      self.inverseSpectDensity = linalg.inv(self.spectDensity) # (numE, numE)
      
    def buildEigenfunctions(self, xy):
      """
      inputs: xy ( len(xy) , inputDim)
      outputs: phi (numE, len(xy))
      """

      if self.inputDim == 1:
          ret = np.zeros((self.numE, xy.shape[0]))
          boundaryL_inv_sqrt = (1 / np.sqrt(self.boundaryL))**self.inputDim

          for idx in prange(self.numE):
              i = self.sortedOrder[idx, 0]
              ret[idx, :] = boundaryL_inv_sqrt * np.sin(np.pi * (i + 1) * (xy + self.boundaryL) / (2 * self.boundaryL))
          return ret
      elif self.inputDim == 2:
          return compute_eigenfunctions(xy, self.sortedOrder, self.boundaryL, self.inputDim, self.numE)
      elif self.inputDim == 3:
        pass
      
      return None
    
    
    def SpectralDensityFunc(self, covFunc, kernParams, frequency):
      """
      inputs: frequency (numE, 1)
      outputs: spectral density vector (numE, 1)
      """
      kernType = covFunc['kernType']
      if kernType == 'Matern':
        nominator = 2**self.inputDim * np.pi**(self.inputDim/2) * gamma(kernParams['v'] + self.inputDim/2) * (2 * kernParams['v'])**(kernParams['v'])
        denominator = gamma(kernParams['v']) * kernParams['l']**(2*kernParams['v']) * (2*kernParams['v']/kernParams['l']**2 + frequency**2)**( kernParams['v'] + self.inputDim/2 )
        ret = nominator/denominator
      elif kernType == 'SE':
        ret = kernParams['strength'] * (2*np.pi*kernParams['l']**2)**(self.inputDim/2) * np.exp(-(kernParams['l']*frequency)**2 / 2)
      elif kernType == 'Laplace':
        nominator = 2 * kernParams['l'] 
        denominator = (1 + (kernParams['l'] * frequency)**2)
        ret = nominator/denominator
      elif kernType == 'GE':
        nominator = 2 * kernParams['strength']**2 * kernParams['l'] * gamma(2/kernParams['gamma'])
        denominator = kernParams['gamma'] * (1 + (frequency * kernParams['l'])**(2/kernParams['gamma']))
        ret = nominator/denominator
      elif kernType == 'RQ':
        # nominator = 2**(self.inputDim) * np.pi**(self.inputDim/2) * gamma(kernParams['alpha'] + self.inputDim/2) * kernParams['l']**(2*kernParams['alpha'])
        # denominator = gamma(kernParams['alpha'])
        # last = (1 + (frequency**2 * kernParams['l']**2)/(2*kernParams['alpha']))**(-kernParams['alpha'] - self.inputDim/2)
        # nominator = (2*np.pi)**(self.inputDim/2) * gamma(kernParams['alpha'] + self.inputDim/2) * (2*kernParams['alpha']*(kernParams['l']**2))**(self.inputDim/2)
        # denominator = gamma(kernParams['alpha'])
        # last = (1 + (frequency**2 * kernParams['l']**2)/(2*kernParams['alpha']))**(-kernParams['alpha'] - self.inputDim/2)        
        # ret = nominator/denominator * last
        
        first = (4*np.pi*kernParams['alpha']*kernParams['l']**2)**(self.inputDim/2)
        second = gamma(kernParams['alpha']-self.inputDim/2)/gamma(kernParams['alpha']) * (frequency * kernParams['l'] * np.sqrt(2*kernParams['alpha']))**(self.inputDim/2-kernParams['alpha'])
        last = kv(kernParams['alpha']-self.inputDim/2, frequency * kernParams['l'] * np.sqrt(2*kernParams['alpha']))
        ret = first * second * last
      return ret
    
    def sortSpecDensity(self, specDensity, index):
      """
      inputs: specDensity (80, 1)
      outputs: sorted spectral density vector (numE, 1)
               sorted index matrix (numE, inputDim)
      """
      if self.inputDim == 1:
        tempMat = np.append(specDensity.reshape(-1,1), index, axis=1)
        filteredMat = tempMat[~np.isnan(tempMat).any(axis=1)]
        numE = filteredMat.shape[0]
        if self.numE > numE:
            self.numE = numE
            print("numE is changed to ", numE)
        sortedMat = filteredMat[np.argsort(filteredMat[:,0])[::-1]]
        sortedOrder = sortedMat[:self.numE,1:].astype(np.uint64)
        sortedDensity = sortedMat[:self.numE,0].reshape(-1, 1)
        
      elif self.inputDim == 2:
        # (i, j)
        xIndex, yIndex = np.meshgrid(index, index)
        xyIndices = np.vstack([xIndex.ravel(), yIndex.ravel()]).T
        
        # (Density_i, Density_j)
        xDensity, yDensity = np.meshgrid(specDensity, specDensity)
        xyDensity = xDensity.ravel() * yDensity.ravel()

        # Sorting the xyDensity
        tempMat = np.append(xyDensity.reshape(-1,1), xyIndices, axis=1)
        
        sortedMat = tempMat[np.lexsort(np.fliplr(tempMat).T)[::-1]]
        sortedOrder = sortedMat[:self.numE,1:].astype(np.uint64)
        sortedDensity = sortedMat[:self.numE,0].reshape(-1, 1)
        
      elif self.inputDim == 3:
        pass
      return sortedDensity, sortedOrder    