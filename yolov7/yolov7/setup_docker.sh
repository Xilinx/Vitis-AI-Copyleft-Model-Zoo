# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mkdir -p ~/.config/pip
echo "[global]
trusted-host = xcdpython.xilinx.com
index-url = https://xcdpython.xilinx.com/simple
# For Ubuntu servers
cert = /etc/ssl/certs/ca-certificates.crt
# For CentOS servers
# cert = /etc/pki/tls/cert.pem
[search]
index = http:s//xcdpython.xilinx.com/pypisearch/" > ~/.config/pip/pip.conf
echo "show_channel_urls: true
channels:
  - https://xcdconda/pytorch/
  - https://xcdengvm244030/omnia/
  - https://xcdengvm244030/conda-forge/
  - https://xcdengvm244030/bioconda/
  - https://xcdengvm244030/defaults/
ssl_verify: /etc/ssl/certs/ca-certificates.crt
report_errors: false
auto_activate_base: false" > ~/.condarc
sudo mv  /etc/conda/condarc  /etc/conda/condarc_bak
conda config --set ssl_verify no
sudo cp /etc/apt/sources.list /etc/apt/sources.list.backup
echo "deb http://xap-engapt-sg/mirror/us.archive.ubuntu.com/ubuntu/ bionic-backports main restricted universe multiverse
deb http://xap-engapt-sg/mirror/us.archive.ubuntu.com/ubuntu/ bionic main restricted universe multiverse
deb [arch=amd64 trusted=yes] http://xap-engapt-sg/mirror/developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" | sudo tee /etc/apt/sources.list
