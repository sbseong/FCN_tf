import skimage
import numpy as np
import skimage.io as io
def background_extractor(save_dir,imagefilename, image, seg_image):  # image: real image # seg_image: pred_np.squeeze()
    
    mask = np.zeros(shape=image.shape, dtype=int)
    bg_true = seg_image == -1
    mask[bg_true] = 1

    b_image = image*mask
    
    io.imsave(save_dir+"bg_of_{}".format(imagefilename.split(".")[0]+".png"),b_image)

def person_extractor(save_dir,imagefilename, image, seg_image):
    
    mask = np.zeros(shape=image.shape, dtype=int)
    p_true = seg_image == -16
    mask[p_true] = 1

    p_image = image*mask
    
    io.imsave(save_dir+"p_of_{}".format(imagefilename.split(".")[0]+".png"),p_image)
