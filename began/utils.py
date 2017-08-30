import os
import numpy as np
import keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

def inverseRescale(X):
    return (X * 255.)

def saveModelWeights(generator_model, discriminator_model, e, localPath):
    
    if(localPath.endswith('/')):
        raise Exception('Path must not end with /')

    model_path = os.getcwd() + localPath

    gen_weights_path = os.path.join(model_path, 'gen_epoch'+ str(e) +'.h5')
    generator_model.save_weights(gen_weights_path, overwrite=True)

    disc_weights_path = os.path.join(model_path, 'disc_epoch'+ str(e) +'.h5')
    discriminator_model.save_weights(disc_weights_path, overwrite=True)

def loadModelWeights(generator_model, discriminator_model, fileName, localPath):
    
    if(localPath.endswith('/')):
        raise Exception('Path must not end with /')

    model_path = os.getcwd() + localPath

    gen_weights_path = os.path.join(model_path, fileName)
    generator_model.save_weights(gen_weights_path)

    disc_weights_path = os.path.join(model_path, fileName)
    discriminator_model.save_weights(disc_weights_path)


def plotGeneratedBatch(X_real, X_gen, localPath):

    if(not localPath.endswith('.png')):
        raise Exception('Must be .png file')

    X_real = inverseRescale(X_real)
    X_gen = inverseRescale(X_gen)

    Xg = X_gen[:8]
    Xr = X_real[:8]
        
    ax = 0 if K.image_dim_ordering() == "tf" else 1
        
    X = np.concatenate((Xg, Xr), axis=0)
    list_rows = []
    for i in range(int(X.shape[0] / 4)):
        Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=ax+1)
        list_rows.append(Xr)

    Xr = np.concatenate(list_rows, axis=ax)
    if(ax == 1):
        Xr = Xr.transpose(1,2,0)

    if Xr.shape[-1] == 1:
        plt.imshow(Xr[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xr)
    plt.savefig(os.getcwd() + localPath)
    plt.clf()
    plt.close()
