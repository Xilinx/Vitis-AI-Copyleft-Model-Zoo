echo "Install all the python dependencies using pip"
pip install --trusted-host xcdpython.xilinx.com -r requirements_ptq.txt
cd code/
python setup.py develop
