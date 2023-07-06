ROOT_DIR=${PWD}
echo "Install all the python dependencies using pip"
pip install --trusted-host xcdpython.xilinx.com -r yolov7/requirements.txt

cd ${ROOT_DIR}
