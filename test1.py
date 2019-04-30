from __future__ import division

import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np
import argparse
from bg_extract import background_extractor, person_extractor
sPath = "/projects1/pi/sbseong/pythonwork/FCN_demo/"
sys.path.append(sPath)
sys.path.append(sPath + "models/slim/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
				description = "Enter your FCN_demo path")
    parser.add_argument("--dir", type=str , help = "Enter your FCN_demo path, exclusing / at the end")
    args = parser.parse_args()

fcn_16s_checkpoint_path = sPath + 'tf_image_segmentation/saver/model_fcn8s_final.ckpt'

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


from tf_image_segmentation.models.fcn_8s import FCN_8s
from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut

number_of_classes = 21


image_data_folder_path = sPath + "data/"
image_files = [x for x in os.listdir(image_data_folder_path) if x.endswith((".jpg",".png"))]


slim = tf.contrib.slim
	
image_filename_placeholder = tf.placeholder(tf.string)

image_tensor = tf.read_file(image_filename_placeholder)

image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)

# Fake batch for image and annotation by adding
# leading empty axis.
image_batch_tensor = tf.expand_dims(image_tensor, axis=0)

# Be careful: after adaptation, network returns final labels
# and not logits
FCN_8s = adapt_network_for_any_size_input(FCN_8s, 32)


pred, fcn_16s_variables_mapping = FCN_8s(image_batch_tensor=image_batch_tensor,
	                                  number_of_classes=number_of_classes,
	                                  is_training=False)

# The op for initializing the variables.
initializer = tf.local_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    
    sess.run(initializer)

    saver.restore(sess,
     sPath + "tf_image_segmentation/saver/model_fcn8s_final.ckpt")
    
    
    for image in image_files:
        image_filename =  image_data_folder_path + image
    
        feed_dict_to_use = {image_filename_placeholder: image_filename}
    
        image_np, pred_np = sess.run([image_tensor, pred], feed_dict=feed_dict_to_use)

        io.imsave(sPath+"tf_image_segmentation/generated/pred_{}".format(image), np.invert(pred_np.squeeze()))

        background_extractor(sPath+"/tf_image_segmentation/background_extracted/",image, image_np, np.invert(pred_np.squeeze()))
        person_extractor(sPath+"/tf_image_segmentation/person_extracted/",image,  image_np, np.invert( pred_np.squeeze()))

