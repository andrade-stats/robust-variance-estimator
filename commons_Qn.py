# define constants for the estimators with UNKNOWN mean

from enum import Enum
CorrectionType = Enum("CorrectionType", ["MAD", "QN_FINITE", "QN_ASYMPTOTIC", "QN_OPT_LIN", "QN_OPT_MSE"])

CORRECTION_TYPES_FOR_TABLES = [CorrectionType.MAD, CorrectionType.QN_FINITE, CorrectionType.QN_ASYMPTOTIC, CorrectionType.QN_OPT_LIN, CorrectionType.QN_OPT_MSE]

correctionType_to_label = {}
correctionType_to_label[CorrectionType.MAD] = r'$\text{\normalfont MAD}^2$'
correctionType_to_label[CorrectionType.QN_FINITE] = r'$Q_n^2$ (finite)'
correctionType_to_label[CorrectionType.QN_ASYMPTOTIC] = r'$Q_n^2$ (asymptotic)'
correctionType_to_label[CorrectionType.QN_OPT_LIN] = r'$Q_{\text{\normalfont opt-lin}}^2$'
correctionType_to_label[CorrectionType.QN_OPT_MSE] = r'$Q_{\text{\normalfont opt-mse}}^2$'
