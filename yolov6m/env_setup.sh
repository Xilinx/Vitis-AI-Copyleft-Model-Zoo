ROOT_DIR=${PWD}
echo "Install all the python dependencies using pip"
pip install --trusted-host xcdpython.xilinx.com -r yolov6/requirements.txt

cd torch_rewriters
python setup.py install

cd ${ROOT_DIR}
