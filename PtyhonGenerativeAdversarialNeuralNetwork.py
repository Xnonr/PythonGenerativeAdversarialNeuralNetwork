#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Based off of the 'Build a Generative Adversarial Neural Network' tutorial by Nicholas Renotte
# Tutorial Video Link: 'https://www.youtube.com/watch?v=AALBGpLbj6Q'


# In[ ]:


# Installing Dependencies
#!pip3 install ipywidgets
#!pip3 install matplotlib
#!pip3 install tensorflow
#!pip3 install tensorflow-datasets
#!pip3 install tensorflow-gpu


# In[ ]:


# Displaying Available & Installed PIP Packages
#!pip3 list


# In[ ]:


# Importing Dependencies
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds

from matplotlib import pyplot as plt

from tensorflow.keras.callbacks import Callback

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import UpSampling2D

from tensorflow.keras.losses import BinaryCrossentropy

# Importing the base model class to subclass the training step
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import array_to_img


# In[ ]:


# Downloading Fashion Dataset

# Utilizing the TensorFlow Datasets API so as to bring in the data source
fashion_ds = tfds.load('fashion_mnist', split = 'train')


# In[ ]:


type(fashion_ds)


# In[ ]:


'''
Is a classification dataset, with an image being associated with some form of clothing
    such as shirts, pants, boots, shoes, etc.
Think of the code below as using a pipeline, with a set of repeatable calls used in order
    to bring in the fashion data back via the TensorFlow Dataset API
An iterator, similar to a loop, keeps having batches of images called up one after the other
    with '.next()'
''' 
fashion_ds.as_numpy_iterator().next().keys()


# In[ ]:


# Visualizing The Dataset

# Setting up an iterator and data connection
data_iterator = fashion_ds.as_numpy_iterator()


# In[ ]:


'''
Going to the next image via the previously established iterator
Each time the below command is run a new image and its associated data is fetched
Helps preserve the available amount of local memory upon the current computer by calling
    and fetching the data as needed, rather than loading everything all at once into memory
Data is being retrieved from a pipeline
'''
data_iterator.next()


# In[ ]:


'''
Setting up the subplots formatting, with a total of 4 columns 
    and each plots size being 20 by 20 pixels in size
The whole display is 'figure', whilst each individual subplot is represent by 'subplot'
'''
figure, subplot = plt.subplots(ncols = 4, figsize = (20, 20))

# Displaying 4 images from the fashion dataset as examples
for index in range(4):
    
    # Grabs an image along with its associated label
    batch = data_iterator.next()
    
    '''
    Plots the image for visual display utilizing a specific subplot
    Original image arrays 'squeezed' down from 3 to 2 dimensions for easier
        visualization, and so some minor data transformation has occured
    '''
    subplot[index].imshow(np.squeeze(batch['image']))
    
    # Appens the image label as the specific subplot's associated title
    subplot[index].title.set_text(batch['label'])


# In[ ]:


'''
Even squeezed down, the images are represented from values of 0 to 255, 
    for neural networks and other good deep learning models these should
    be scaled down to become values of either 0 or 1
'''

# Scales own and returns the images only, without their associated labels
def scale_images(data):
    image = data['image']
    return (image / 255)


# In[ ]:


'''
Building The Dataset

Typical steps for building up a TensorFlow Pipeline:
    1. Map
    2. Cache
    3. Shuffle
    4. Batch
    5. Prefetch
'''

# Reloads the dataset if not already done
#fashion_ds = tfds.load('fashion_mnist', split = 'train')

# Running the dataset through the 'scale_images' function preprocessing step
fashion_ds = fashion_ds.map(scale_images)

# Caching the dataset for the given batch
fashion_ds = fashion_ds.cache()

'''
Shuffles the dataset similarly to shuffling a deck of cards to avoid repeatedly
    looking at the same set of data every time and therefore making erroneus conclusions
'''
fashion_ds = fashion_ds.shuffle(60000)

# Batches of 128 images are gathered per sample
fashion_ds = fashion_ds.batch(128)

# Reduces the likelihood of bottlenecking during the process
fashion_ds = fashion_ds.prefetch(64)


# In[ ]:


'''
The iterator should retrieve a batch of 128 images all of dimensions 28 by 28 pixels 
    by 1 channel, signaling their being grayscale in nature
'''
fashion_ds.as_numpy_iterator().next().shape


# In[ ]:


# Building The Neural Network

# Building The Synthetic Data Generator
def build_generator():
    model = Sequential()
    
    '''
    Specifies what the input layer is going to be
    Dense, fully connected layer, with 128 random values being passed in in order
        to help the generator decide what to create in terms of 7 by 7 pixel sized 
        images by giving it some latent space context
    '''
    model.add(Dense(7 * 7 * 128, input_dim = 128))
    
    # Leaky ReLU activation, recommended in data synthesis and generation neural networks
    model.add(LeakyReLU(0.2))
    
    '''
    Reshaping the dense layer's output into the begginings of an image, which at this point
        is 7 by 7 pixels with 128 different channels
    '''
    model.add(Reshape((7, 7, 128)))
    
    '''
    Upsampling blocks are meant to push the newly generated beginnings of 
        images closer to what the desired output should actually be in terms of pixel
        size and channels
    '''
    
    # Upsampling Block 1
    
    '''
    Reshapes the generated model images by upsampling them to 14 by 14 pixel 
        images with 128 layers, in this case likely simply duplicating pixels
        side by side to generate a large image
    '''
    model.add(UpSampling2D())
    
    '''
    Convolutional layer condenses this upsampled output back down, removing the upsampling
        paramater data to limit the amount of information
    Padding avoids undesired image cropping within this scenario
    '''
    model.add(Conv2D(128, 5, padding = 'same'))
    model.add(LeakyReLU(0.2))
    
    # Upsampling Block 2
    # Images become 28 by 28 pixels with 128 layers
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding = 'same'))
    model.add(LeakyReLU(0.2))
    
    '''
    Convolutional blocks made in order to provide more paramaters for the model to 
        play around with when generating synthetic images as data
    '''
    
    # Convolutional Block 1
    model.add(Conv2D(128, 4, padding = 'same'))
    model.add(LeakyReLU(0.2))
    
    # Convolutional Block 2
    model.add(Conv2D(128, 4, padding = 'same'))
    model.add(LeakyReLU(0.2))
    
    # Convolutional layers to reach an image with only 1 channel
    model.add(Conv2D(1, 4, padding = 'same', activation = 'sigmoid'))
    
    return model


# In[ ]:


generator = build_generator()


# In[ ]:


generator.summary()


# In[ ]:


# Generating 4 synthetic images
test_image = generator.predict(np.random.randn(4, 128, 1))

# Visualizing newly generated synthetic images data from the generator
figure, subplot = plt.subplots(ncols = 4, figsize = (20, 20))
for index, image in enumerate(test_image):
    subplot[index].imshow(np.squeeze(image))
    subplot[index].title.set_text(index)


# In[ ]:


test_image.shape


# In[ ]:


# Building The Discriminator

# Essential acts as an image classifier
def build_discriminator():
    model = Sequential()
    
    # Convolutional Block 1
    '''
    Convolutional layer has 32 filters with a shape of 5 by 5
    Input shape is that of a proper desired fashion image
    '''
    model.add(Conv2D(32, 5, input_shape = (28, 28, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    # Convolutional Block 2
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    # Convolutional Block 3
    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    # Convolutional Block 4
    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    # Flattens and passes into a dense layer
    model.add(Flatten())
    model.add(Dropout(0.4))
    
    '''
    An output value of '1' represents a False, synthesized and generated image,
        whilst a '0' represents a True, meaning the image comes from a real life
        piece of clothing
    '''
    model.add(Dense(1, activation = 'sigmoid'))
    
    return model


# In[ ]:


discriminator = build_discriminator()


# In[ ]:


discriminator.summary()


# In[ ]:


# Predicts whether or not the synthesized and generated test images are real or fake
discriminator.predict(test_image)


# In[ ]:


'''
Constructing The Training Loop

Generators have to train the generator and discriminator side by side in order to 
    properly learn, and so simply using the typical '.fit()' function does not 
    work in this scenario
A specific training sequence, or loop, is therefore needed for these types of neural networks
'''

'''
Setting Up Losses & Optimizers

The generator model will be rewarded for tricking the discriminator, whilst the 
    discriminator will be rewarded for sniffing out the generator's bogus fashion
    items
The learning rate for the discriminator will be slower than the generator in order to prevent
    the former discriminator from completely crushing the generator before the latter has time to
    properly learn to make convincing synthetic data
'''

generator_optimizer = Adam(learning_rate = 0.0001)
discriminator_optimizer = Adam(learning_rate = 0.00001)

generator_loss = BinaryCrossentropy()
discriminator_loss = BinaryCrossentropy()


# In[ ]:


# Setting Up The Subclassed Model

class FashionGAN(Model):
    
    def __init__(self, generator, discriminator, *args, **kwargs):
        
        # Passes through the positional and keyword arguments into the base Model class
        super().__init__(*args, **kwargs)
        
        # Creates attributes for the generator and discriminator respectively
        self.generator = generator
        self.discriminator = discriminator
    
    def compile(self, generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss,
                *args, **kwargs):
        
        # Compiles with the base Model class
        super().compile(*args, **kwargs)
        
        # Creates attributes for the generator's and discriminator's optimizers and losses
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
    
    def train_step(self, batch):
        
        # Retrives the data
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training = False)
        
        # Trains the discriminator
        with tf.GradientTape() as discriminator_tape:
            
            #Passes in the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_images, training = True)
            yhat_fake = self.discriminator(fake_images, training = True)
            
            # Predicitions from the discriminator
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis = 0)
            
            # Creates labels for both the respective real and fake images, actual results
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis = 0)
            
            # Adds some noise to the outputs to prevent the discriminator from learning too quickly
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += 0.15 * tf.concat([noise_real, noise_fake], axis = 0)
            
            # Calculates the training loss
            total_discriminator_loss = self.discriminator_loss(y_realfake, yhat_realfake)
            
        # Applies backpropogation, which effectively allows for the neural network to learn
        discriminator_gradient = discriminator_tape.gradient(total_discriminator_loss, 
                                                             self.discriminator.trainable_variables)
        
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradient, 
                                                         self.discriminator.trainable_variables))
        
        # Trains the generator
        with tf.GradientTape() as generator_tape:

            # Generates new imahes
            generated_images = self.generator(tf.random.normal((128, 128, 1)), training = True)

            # Creates the predicted labels
            predicted_labels = self.discriminator(generated_images, training = False)
            
            '''
            Calculates loss
            The generator attemps to trick the discriminator by labeling its synthesized clothing
                images as true, and will be rewarded if it does so successfully, whilst labeling
                the real images as false
            '''
            total_generator_loss = self.generator_loss(tf.zeros_like(predicted_labels), predicted_labels)
            
        # Applies backpropogation
        generator_gradient = generator_tape.gradient(total_generator_loss,
                                                     self.generator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(zip(generator_gradient, 
                                                     self.generator.trainable_variables))
        
        return {"discriminator_loss":total_discriminator_loss, "generator_loss":total_generator_loss}


# In[ ]:


# Creates an instance of the subclassed model
fashion_gan = FashionGAN(generator, discriminator)


# In[ ]:


# Compiles the instantiated model
fashion_gan.compile(generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss)


# In[ ]:


# Building Callback
class ModelMonitor(Callback):
    
    def __init__(self, number_of_images = 3, latent_dim = 128):
        self.number_of_images = number_of_images
        self.latent_dim = latent_dim
        
    def on_epoch_end(self, epoch, logs = None):
        random_latent_vectors = tf.random.uniform((self.number_of_images, self.latent_dim, 1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        
        for index in range(self.number_of_images):
            image = array_to_img(generated_images[index])
            image.save(os.path.join('GANImages', f'generated_img_{epoch}_{index}.png'))


# In[ ]:


# Training Models

# 2000 Epochs Recommended
history = fashion_gan.fit(fashion_ds, epochs = 20, callbacks = [ModelMonitor()])


# In[ ]:


# Review Model Performances
plt.suptitle('Loss')
plt.plot(hist.history['discriminator_loss'], label = 'Discriminator Loss')
plt.plot(hist.history['generator_loss'], label = 'Generator Loss')
plt.legend()
plt.show()


# In[ ]:


generator.load_weights(os.path.join('Models', 'generatormodel.h5'))


# In[ ]:


# Testing Out The Generator
generated_images = generator.predict(tf.random.normal((16, 128, 1)))


# In[ ]:


# Displaying the generated synthetic images from a trained generator model
figure, subplot = plt.subplots(ncols = 4, nrows = 4, figsize = (20, 20))

for row in range(4):
    for column in range(4)
        subplot[row][column].imshow(generated_images[(row + 1) * (column + 1) - 1])


# In[ ]:


# Saving The Model
generator.save('generator.h5')
discriminator.save('discriminator.h5')

