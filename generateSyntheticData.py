# used to generate synthetic data for the example plot (but is not used for the simulation results of Section 5)

import numpy
import scipy.stats

def getNrOutlierAndRndOutlierIds(n, OUTLIER_RATIO):
    nrOutliers = int(n * OUTLIER_RATIO)
    rndOutlierIds = numpy.arange(n)
    numpy.random.shuffle(rndOutlierIds)
    rndOutlierIds = rndOutlierIds[0:nrOutliers]
    
    trueOutlierIds_zeroOne = numpy.zeros(n)
    trueOutlierIds_zeroOne[rndOutlierIds] = 1
    return nrOutliers, rndOutlierIds, trueOutlierIds_zeroOne

        
def addUniformNoise(y, nrOutliers, rndOutlierIds):
    y_std = numpy.std(y)
    OUTLIER_LENGTH = 12.0 * y_std
    
    trueOutlierSamplesRaw = numpy.random.uniform(low=0.0, high=OUTLIER_LENGTH, size=nrOutliers)
    lowerOutliers = -trueOutlierSamplesRaw[trueOutlierSamplesRaw < OUTLIER_LENGTH/2] - (y_std*3.0)  
    higherOutliers = trueOutlierSamplesRaw[trueOutlierSamplesRaw >= OUTLIER_LENGTH/2] - (OUTLIER_LENGTH/2) + (y_std*3.0)
    noise = numpy.hstack((lowerOutliers, higherOutliers))
    y[rndOutlierIds] += noise   
    return y

def generateDataAdvanced_UniformOutliers(NR_SAMPLES, OUTLIER_RATIO):
    assert(OUTLIER_RATIO >= 0.0 and OUTLIER_RATIO <= 0.5)
    
    INLIER_MEAN = 0.0
    INLIER_STD = 1.0
    
    y = scipy.stats.norm.rvs(loc=INLIER_MEAN, scale=INLIER_STD, size=NR_SAMPLES)
    nrOutliers, rndOutlierIds, trueOutlierIds_zeroOne = getNrOutlierAndRndOutlierIds(NR_SAMPLES, OUTLIER_RATIO)
    
    y = addUniformNoise(y, nrOutliers, rndOutlierIds)
    
    return INLIER_MEAN, INLIER_STD, y.astype(numpy.float32), trueOutlierIds_zeroOne

