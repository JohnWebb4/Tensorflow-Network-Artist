"""
Run Network Artist.
Will generate images based off of previous images
"""
import os
import nartist
from scipy import misc
import tensorflow as tf

image_path = "Images"  # path to images


def print_help():
    """
    Write help
    :return: None
    """
    print("Type 'Exit' to exit.")  # write exit
    print("Type 'Help' to see this message.")  # write help
    print("Type 'Predict' to predict the image.")  # write predict
    print("Type 'Load' to load a network.")  # write load
    print("Type 'Save' to save the network.")  # write save
    print("Type 'Train' to train the network of all images in /Images folder.")  # write train


class Program:
    """
    Main class for network artist program
    """

    def __init__(self):
        """
        Initialize program
        """
        self.network = nartist.NArtist(width=256,
                                       height=256,
                                       channels=3,
                                       learn_rate=0.001,
                                       batch_size=10,
                                       title_length_max=100,
                                       clarity=20)  # Create network

    def load(self):
        """
        Load a network
        :return: None
        """
        file = input("Enter the network to load (.ckpt): ")  # get file
        saver = tf.train.Saver()  # create saver
        try:
            saver.restore(self.network.sess, file)  # restore
            print("Loaded {0}.".format(file))  # write
        except:
            print("Failed to load {0}.".format(file))  # write

    def predict(self):
        """
        Predict picture
        :return: None
        """
        # Predict
        title = input("Enter the title: ")  # get title
        image = self.network.predict(title.lower())  # get prediction

        # Save
        while True:  # loop for save
            filename = input("Enter the file to save to: ")  # get filename
            try:
                misc.imsave(filename, image)  # attempt save
                print("Saved {0}.".format(filename))  # write
                break  # break
            except ValueError as e:
                print("Could not save {0}.\n{1}".format(filename, e.args))  # write
            except FileNotFoundError as e:
                print("Folder does not exist for {0}.\n{1}".format(filename, e.args))  # write

    def run(self):
        """
        Run the program
        :return: None
        """
        if not os.path.exists(image_path):  # if images folder does not exist
            os.makedirs(image_path)  # make images folder
            print("Created images folder. Populate with training data.")  # write creations

        is_exited = False  # flag to exit

        print_help()  # write help

        while not is_exited:  # till exit
            comm = input("Enter command: ").lower()  # get command

            if comm == "exit":  # if exit
                is_exited = True  # exit
            elif comm == "help":  # if help
                print_help()  # write help
            elif comm == "predict":  # if predict
                self.predict()  # predict
            elif comm == "load":  # if load
                self.load()  # load
            elif comm == "save":  # if save
                self.save()  # save
            elif comm == "train":  # if train
                self.train()  # train

    def save(self):
        """
        Save network
        :return: None
        """
        file = input("Enter the filename (.ckpt): ")
        saver = tf.train.Saver()  # create saver
        try:
            saver.save(self.network.sess, file)  # save
            print("Saved {0}.".format(file))  # write
        except:
            print("Failed to save {0}.".format(file))  # write

    def train(self):
        """
        Train network
        :return: None
        """

        # Get images in "Images" directory
        try:
            filenames = \
                [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]  # get images

            # Load images
            images = []  # array of images
            for filename in filenames:  # cycle through files
                images.append(misc.imread(os.path.join(image_path, filename)))  # get image

            # train
            epochs = int(input("Enter number of epochs: "))
            self.network.train([filename.lower() for filename in filenames],
                               images,
                               epochs,
                               progress_every=max(int(epochs / 100), 1))
        except FileNotFoundError as e:  # can't find folder
            os.makedirs(image_path)  # make path
            print("Images path does not exist. Created images folder. Populate with training data and call again.")
            print(e.args)  # write args
        except ValueError:  # can't convert epochs
            print("Can't convert value to number of epochs.")  # write


if __name__ == "__main__":
    program = Program()  # create program
    program.run()  # run
