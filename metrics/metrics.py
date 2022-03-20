import numpy as np


def check_images(fusioned, original):
    assert len(fusioned) == len(original), "Supplied images have different sizes " + \
    str(fusioned.shape) + " and " + str(original.shape)
    if(len(fusioned.shape) == len(original.shape)):
        estado = 'mtom'
        if(len(fusioned.shape) == 2):
            fusioned = fusioned[:,:,np.newaxis]
            original = original[:,:,np.newaxis]
        else:
            assert fusioned.shape[2] == original.shape[2], "Supplied images have different number of bands "
    else:
        estado = 'mtop'
    return estado, fusioned, original


def mse(fusioned, original):
    """calculates mean squared error (mse).
    :param GT: first (original) input image.
    :param P: second (deformed) input image.
    :returns:  float -- mse value.
    """
    array_mse = []
    mode, fusioned, original = check_images(fusioned, original)
    if(mode == 'mtom'):
        for i in range(fusioned.shape[2]):
            aux_val = np.mean((fusioned[:,:,i].astype(np.float64)-original[:,:,i].astype(np.float64))**2)
            array_mse.append(aux_val)
    else:
        for i in range(fusioned.shape[2]):
            aux_val = np.mean((fusioned[:,:,i].astype(np.float64)-original.astype(np.float64))**2)
            array_mse.append(aux_val)

    return np.array(array_mse)


def rmse(fusioned, original):
    return np.sqrt(mse(fusioned, original))


def bias(fusioned, original):
    array_bias = []
    mode, fusioned, original = check_images(fusioned, original)
    if(mode == 'mtom'):
        for i in range(fusioned.shape[2]):
            aux_val = 1 - ((np.mean(fusioned[:,:,i].astype(np.float64)))/ (np.mean(original[:,:,i].astype(np.float64))))
            array_bias.append(aux_val)
    else:
        for i in range(fusioned.shape[2]):
            aux_val = 1 - ((np.mean(fusioned[:,:,i].astype(np.float64)))/ (np.mean(original.astype(np.float64))))
            array_bias.append(aux_val)
    return array_bias


def correlation_coeff(fusioned, original):
    array_corrcoef = []
    mode, fusioned, original = check_images(fusioned, original)
    if(mode == 'mtom'):
        for i in range(fusioned.shape[2]):
            aux_val = np.corrcoef(fusioned[:,:,i].astype(np.float64).flat, original[:,:,i].astype(np.float64).flat)
            array_corrcoef.append(aux_val[0][1])
    else:
        for i in range(fusioned.shape[2]):
            aux_val = np.corrcoef(fusioned[:,:,i].astype(np.float64).flat, original.astype(np.float64).flat)
            array_corrcoef.append(aux_val[0][1])
    return array_corrcoef
