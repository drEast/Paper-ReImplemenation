import tensorflow as tf
import numpy as np
import time
import datetime
import os
import PIL.Image


EPOCHS = 30
STEPS_EPOCHS = 100
ALPHA_CONTENT = 1e4  # -3
ALPHA_STYLE = 1e-2
LEARNING_RATE = .02

PATH_STYLE = 'night.jpg'
PATH_CONTENT = 'sphinx.jpg'
OUTPUT = 'samples/' + datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S") + '/'


class Generator:
    def __init__(self):
        os.mkdir(OUTPUT)

        # setup data, model and optimizer
        self.model = load_model()
        img_cont, img_style, self.img_gen = get_data()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.99, epsilon=1e-1)

        # calculate content and style from corresponding images
        self.features_content = self.model_call(img_cont)[-1:]
        self.gram_style = gram_matrix(self.model_call(img_style)[:-1])

    def optimize_img(self):
        """ Conducts the optimization loop. """
        print('--- Start optimizing ---')
        for it in range(EPOCHS):
            start = time.time()
            losses = None
            for step in range(STEPS_EPOCHS):
                losses = self.optimization_step()
            self.save_image(it)
            end = time.time()
            # use the last loss as epoch information
            print_epoch_information(it, losses, end-start)
        print('--- End optimizing ---')

    def optimization_step(self):
        """ One step of Minimizing the loss function. """
        # calculate the loss
        with tf.GradientTape() as tape:
            feat_gen = self.model_call(self.img_gen)
            loss_content = self.content_loss(feat_gen[-1:]) * ALPHA_CONTENT
            loss_style = self.style_loss(feat_gen[:-1]) * ALPHA_STYLE
            loss_total = loss_content + loss_style

        # optimize the image
        grad = tape.gradient(loss_total, self.img_gen)
        self.optimizer.apply_gradients([(grad, self.img_gen)])

        tf.clip_by_value(self.img_gen, clip_value_min=0., clip_value_max=1.)  # pixel values remain in range
        return [loss_total, loss_content, loss_style]

    def content_loss(self, generated):
        """ Calculates the content loss between features of the content and generated image.

        Simple MSE loss
        """
        loss = tf.add_n([tf.reduce_mean(tf.square(layer_cont-layer_gen))
                        for layer_cont, layer_gen in zip(self.features_content, generated)])
        return loss / len(generated)

    def style_loss(self, generated):
        """ Calculate the style loss between features of the style and generated image.

        MSE-loss
        """
        g_gen = gram_matrix(generated)
        loss = tf.add_n([tf.reduce_mean(tf.square(g1-g2))
                         for g1, g2 in zip(self.gram_style, g_gen)])
        return loss

    def save_image(self, iteration):
        """ Save the image of this iteration to the output folder. """
        img_name = str(iteration) + '.jpg'
        img_name = os.path.join(OUTPUT, img_name)
        img = self.img_gen * 255
        img = np.array(img, dtype=np.uint8)[0]
        PIL.Image.fromarray(img).save(img_name)

    def model_call(self, inputs):
        """ Image preprocessing and vgg-pass. """
        x = inputs * 255
        x = tf.keras.applications.vgg19.preprocess_input(x)
        x = self.model(x)
        return x


def load_model():
    """ Loads the trained VGG-Model which is used as a feature extractor. """
    trained_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    trained_model.trainable = False

    output_layers = ['block1_conv1',
                     'block2_conv1',
                     'block3_conv1',
                     'block4_conv1',
                     'block5_conv1',
                     'block5_conv2']
    net_output = [trained_model.get_layer(name).output for name in output_layers]
    return tf.keras.Model(trained_model.input, net_output)


def get_data():
    """ Prepare the content and style image. """
    content_img = load_image(PATH_CONTENT)
    style_img = load_image(PATH_STYLE)
    generated_img = tf.Variable(content_img)
    return content_img, style_img, generated_img


def load_image(path_img):
    """ Load image from path to correct dimensions. """
    img = tf.io.read_file(path_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    longer_side = max(shape)
    new_shape = tf.cast(shape * 512/longer_side, tf.int32)
    img = tf.image.resize(img, new_shape)
    return img[tf.newaxis, :]


def gram_matrix(matrix_list):
    """ Calculates the Gram matrix of an input matrix.

    G = 1/(x*y) * sum(M_lxyc * M_lxyd)
    = average (over positions) of feature vector for each layer
    """
    gram_total = []
    for matrix in matrix_list:
        gram = tf.linalg.einsum('lxyc,lxyd->lcd', matrix, matrix)  # layer, x-dim, y-dim, channels
        matrix_shape = tf.shape(matrix)
        positions = tf.cast(matrix_shape[1]*matrix_shape[2], tf.float32)
        gram_total.append(gram/positions)
    return gram_total


def print_epoch_information(iteration, losses, duration):
    """ Console output of Epoch, losses and timing. """
    text = ['Epoch: %d/%d' % (iteration+1, EPOCHS),
            'total loss: %.2f' % losses[0],
            'content loss: %.2f' % losses[1],
            'style loss: %.2f' % losses[2],
            'in: %.2f' % duration]
    print(' '.join(text))


if __name__ == '__main__':
    gen = Generator()
    gen.optimize_img()
