import os
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np
import models
import train
import utils
from tensorflow import flags
from tensorflow import app

FLAGS = flags.FLAGS

if __name__ == "__main__":
    # Dataset flags.
    flags.DEFINE_string("train_data", "./faces",
                        "The directory to save the model files in.")

    flags.DEFINE_string("image_dir", "./out",
                        "The directory to save the generated image files in.")

    flags.DEFINE_string("epochs", 100,
                        "epochs")

    flags.DEFINE_string("batches_per_epoch", 150,
                        "Batches per epoch")

    flags.DEFINE_string("batch_size", 10,
                        "Batch size")

    flags.DEFINE_string("gamma", .5,
                        "Gamma")

    flags.DEFINE_string("img_size", 50,
                        "Size of square image")

    flags.DEFINE_string("channels", 1,
                        "Channel of image")

    flags.DEFINE_string("save_file", None,
                        "H5 saved file name")

def main(unused_argv):
    #Training parameters
    epochs = FLAGS.epochs
    batches_per_epoch = FLAGS.batches_per_epoch
    batch_size = FLAGS.batch_size
    gamma = FLAGS.gamma #between 0 and 1

    #image parameters
    img_size = FLAGS.img_size #Size of square image
    channels = FLAGS.channels #1 for grayscale

    #Model parameters
    z = 100 #Generator input
    h = 128 #Autoencoder hidden representation
    adam = Adam(lr=0.00005) #lr: between 0.0001 and 0.00005
    #In the paper, Adam's learning rate decays if M stalls.  This is not
    #implemented.

    #Build models
    generator = models.decoder(z, img_size, channels)
    discriminator = models.autoencoder(h, img_size, channels)
    gan = models.gan(generator, discriminator)

    generator.compile(loss=models.l1Loss, optimizer=adam)
    discriminator.compile(loss=models.l1Loss, optimizer=adam)
    gan.compile(loss=models.l1Loss, optimizer=adam)

    #Load data
    train_datagen = image.ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(FLAGS.train_data,
                                                    target_size=(img_size, img_size),
                                                    batch_size=batch_size,
                                                    color_mode='grayscale',
                                                    class_mode='categorical')

    trainer = train.GANTrainer(generator, discriminator, gan, train_generator,
                                                    FLAGS.save_file,
                                                    saveModelFrequency=1,
                                                    saveSampleSwatch=True)
    trainer.train(epochs, batches_per_epoch, batch_size, gamma, FLAGS.image_dir)

if __name__ == "__main__":
    exec("""
if not os.path.exists(FLAGS.image_dir):
    os.makedirs(FLAGS.image_dir)""")

    app.run()