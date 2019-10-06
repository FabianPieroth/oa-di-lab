# ------------------------------------------------------------------------------
#  File: oa.py
#  Author: Jan Kukacka
#  Date: 11/2018
#  Email: jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Common functions for processing optoacoustic data
# ------------------------------------------------------------------------------

import numpy as np
import scipy.io
import scipy.optimize

from pathlib import Path


def _get_default_spectra():
    '''
    Function to get the default spectra used by functions in this module
    Default are clinical spectra w/o collagen

    # Returns:
        Clinical spectra of Hb, HbO2, Fat and Water.
        Shape is (4,28)
    '''
    project_root_dir = str(Path().resolve().parents[1])
    spectra = scipy.io.loadmat(project_root_dir + '/src/' + 'logger/oa_spectra_analysis/clinical_spectra.mat')['spectra_L2'].T
    return spectra[:4]


def linear_unmixing(data, spectra=None, non_negative=False, return_error=False,
                    use_constant=False):
    '''
    Performs linear unmixing of given spectral image.

    # Arguments
        - data: array of shape (n_pixels,n_wavelengths) or (height,width,
            n_wavelengths) containing the multispectral image.
        - spectra: array of shape (n_chromophores, n_wavelengths) with absorption
            spectra of chromophores to unmix. If None, default spectra of Hb,
            HbO2, Fat and Water will be used.
        - non_negative: boolean. If true, non negative least squares will be
            used, otherwise normal linear squares. Default False.
        - return_error: boolean. If true, function will return also error value.
            Default False.
        - use_constant: boolean. If true, add constant component to the linear
            model and remove it before returning coefficients. To get also the
            coefficients of the constant component, add it manually to the
            spectra passed as parameter to this function.

    # Returns
        - unmixed: array of shape (n_pixels, n_chromophores) or (height, width,
            n_chromophores) (based on the input shape) with unmixed coefficients
        - error: array of shape (n_pixels) or  (height, width) containing errors
            of the least square fit (residuals). Returned only if return_error
            is True.
    '''
    data_orig_shape = data.shape
    if len(data_orig_shape) < 2 or len(data_orig_shape) > 3:
        raise ValueError('Invalid shape for parameter data: ' + str(data.shape))

    data = np.reshape(data, (-1, data_orig_shape[-1]))
    n_pixels, n_wavelengths = data.shape

    if spectra is None:
        spectra = _get_default_spectra()
    else:
        if len(spectra.shape) != 2 or spectra.shape[1] != n_wavelengths:
            raise ValueError('Invalid shape for parameter spectra: ' + str(spectra.shape))


    if use_constant:
        ## Add the extra constant component
        spectra = np.concatenate([np.ones((1,n_wavelengths)), spectra], axis=0)

    n_chromophores = spectra.shape[0]

    if not non_negative:
        result = np.linalg.lstsq(spectra.T, data.T, rcond=None)
        unmixed = result[0].T
        error = result[1]
    else:
        unmixed = np.empty((n_pixels, n_chromophores))
        error = np.empty(n_pixels)
        for i in range(n_pixels):
            ## scipy.optimize.nnls does not allow matrix valued results
            ## so we have to go pixel per pixel
            unmixed[i], error[i] = scipy.optimize.nnls(spectra.T, data[i])

    if use_constant:
        ## Remove the extra coefficients of constant term
        unmixed = unmixed[:,1:]
        n_chromophores -= 1

    ## Reshape output to the original shape (n_pixels) or (height, width)
    unmixed = np.reshape(unmixed, data_orig_shape[:-1] + (n_chromophores,))
    error = np.reshape(error, data_orig_shape[:-1])

    if return_error:
        return unmixed, error
    else:
        return unmixed


def spectral_F_test(data, **kwargs):
    '''
    Computes if the spectra fit the data significantly better than a simple
    constant model.

    # Arguments:
        - same as to linear_unmixing. use_constant and return_error are ignored.

    # Returns:
        - pValues: array of shape (height, width) or (n_sampes), based on the
            input shape. Contains p-values of error of rejecting the null
            hypothesis that spectral model explains the data no better than a
            simple constant model.
    '''
    from scipy.stats import f

    # Shape of the default spectra

    if 'spectra' in kwargs:
        n_components = kwargs['spectra'].shape[0]
    else:
        n_components = _get_default_spectra().shape[0]
    n_wavelengths = data.shape[-1]

    kwargs['use_constant'] = True
    kwargs['return_error'] = True

    # Compute sum of squared residuals of the spectral model
    _, error = linear_unmixing(data, **kwargs)

    ## Compute sum of squared residuals of a constant models
    error_linear = np.sum(np.power(data - np.mean(data, axis=-1, keepdims=True),
                                   2), axis=-1)
    # Error may be zero but we don't mind, so set division by zero warning to
    # ignore (don't want to have warnings leaking from the lib functions)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Formula from https://en.wikipedia.org/wiki/F-test#Regression_problems
        F = np.nan_to_num(((error_linear - error) * (n_wavelengths - n_components-1))
                          / (error * n_components))

    # Compute p values under F distribution
    pValues = 1.0 - f.cdf(F,n_components,n_wavelengths-n_components-1)
    return pValues
