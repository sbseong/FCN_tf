# FCN_tf
## virtual environments setting
conda create -n segFCN python=3.5
source activate segFCN

(segFCN)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc0-cp35-cp35m-linux_x86_64.whl

(segFCN)$ pip install --upgrade $TF_BINARY_URL

(segFCN)$ pip install scikit-image Pillow matplotlib numpy tensorflow

## do it yourself
git clone https://github.com/sbseong/FCN_tf

mv model models

VGG16 checkpoint PATH 생성 (http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)

edit 'test1.py' sPath = [your path]

(segFCN)$ python3 test.py --dir [your path]/FCN_tf

## input image
[FCN_tf] / data /  _____.jpg

## output image
[FCN_tf] / tf_image_segmentation / generated / pred_{}.jpg
