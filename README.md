# FCN_tf
## virtual environments setting
conda create -n segFCN python=3.5
source activate segFCN

(segFCN)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc0-cp35-cp35m-linux_x86_64.whl
(segFCN)$ pip install --upgrade $TF_BINARY_URL
(segFCN)$ pip install scikit-image Pillow matplotlib numpy
(segFCN)$ python3 test.py --dir /projects1/pi/sbseong/pythonwork/FCN_demo

