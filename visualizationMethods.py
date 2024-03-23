# used for creating the plot that visualizes the bias of the estimator with trimmed samples

import numpy
import scipy.stats


def visualize_advanced(mu, sigma, observations, outlierIds, estMu, estSigma, estOutlierIds, filename = None):
    import matplotlib.pyplot as plt
    
    scatterY = numpy.zeros(observations.shape[0])
    allSizes = numpy.ones(observations.shape[0]) * 200.0
    
    colors = numpy.asarray(['#1f77b4'] * observations.shape[0]) 
    colors[outlierIds == 1] = '#d62728'

    plt.scatter(observations, scatterY, alpha=0.2, s = allSizes, c = colors) 
    plt.scatter(observations[estOutlierIds == 0], scatterY[estOutlierIds == 0], s = allSizes[estOutlierIds == 0], alpha=0.2, edgecolor = "k", facecolors='none', linestyle='dashed')
    
    
    xRange = numpy.arange(numpy.min(observations) - 1.0, numpy.max(observations) + 1.0, 0.01)
    yValues = scipy.stats.norm.pdf(x = xRange, loc = mu, scale = sigma)
    plt.plot(xRange, yValues)
    
    yValuesEst = scipy.stats.norm.pdf(x = xRange, loc = estMu, scale = estSigma)
    plt.plot(xRange, yValuesEst, color = "k", linestyle='dashed')
    
    plt.xlabel('x')
    
    if filename is None:
        plt.show()
    else:
        plt.savefig("latex/plots/" + filename)
        
    plt.clf()
    return


if __name__ == "__main__":
    import generateSyntheticData
    
    numpy.random.seed(3523421)
    
    METHOD = "fdrBaseline"
    noiseType = "uniform"
    TRUE_OUTLIER_RATIO = 0.05
    NR_SAMPLES = 200
    
    trueMu, trueSigma, allSamples, trueOutlierIndices = generateSyntheticData.generateDataAdvanced_UniformOutliers(NR_SAMPLES, TRUE_OUTLIER_RATIO)
        
    inlierMin = int(NR_SAMPLES / 2)
    
    trimmedModelMu = 0.0

    inlierIds = numpy.argsort(numpy.abs(allSamples))[0:inlierMin]
    estimatedOutlierIds = numpy.ones(NR_SAMPLES)
    estimatedOutlierIds[inlierIds] = 0

    trimmedModelSigma = numpy.sqrt(numpy.mean(numpy.square(allSamples[inlierIds])))

    print("nr of samples used for estimating simga = ", inlierMin)
    print("nr of true inliers = ", NR_SAMPLES - numpy.sum(trueOutlierIndices))
    
    visualize_advanced(trueMu, trueSigma, allSamples, trueOutlierIndices, trimmedModelMu, trimmedModelSigma, estimatedOutlierIds, filename = "groundTruth_" + noiseType + "_" + str(NR_SAMPLES) + "_noCorrection.pdf")
    