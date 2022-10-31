from numpy import argmax
import tensorflow
from keras.utils.image_utils import img_to_array
from keras.models import load_model
from keras.utils.image_utils import load_img
 
# load and prepare the image
def load_image(filename):
	# load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
	# reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img
 
# load an image and predict the class
def run_example():
	# load the image
	img = load_image('K:\sem_5\DBMS\project\Automated_cheque_processing\Cheque_Image_Dataset\cheques\cheque_309066.tif')
	# load model
	model = load_model('bestmodel.h5')
	# predict the class
	predict_value = model.predict(img)
	digit = argmax(predict_value)
	print(digit)
 
# entry point, run the example
run_example()