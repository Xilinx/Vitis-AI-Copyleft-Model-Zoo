<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

![Release Version](https://img.shields.io/github/v/release/Xilinx/Vitis-AI-Copyleft-Model-Zoo)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr-raw/Xilinx/Vitis-AI-Copyleft-Model-Zoo)
[![Documentation](https://img.shields.io/badge/documentation-github.IO-blue.svg)](https://xilinx.github.io/Vitis-AI/)
![Repo Size](https://img.shields.io/github/repo-size/Xilinx/Vitis-AI-Copyleft-Model-Zoo)

# Vitis AI Copyleft Model Zoo

The purpose of this repsitory is to release models that are not compatible with the Vitis™ AI Apache 2.0 license.  The original source code for these models was released via a copyleft, or reciprocal license.  For Vitis AI, these models have been released in this separate repository in order to clearly identify the source license for these models.

The directory structure associated each model is uniquely associated with a specific license file so that the user can evaluate whether the license for the model is compatible with their own company policies and license requirements.  Pre-compiled versions of these models are not provided because the Vitis AI Compiler is not open-source.  Users must leverage the scripts provide as a template to compile the model for their target using the Vitis AI Compiler
        
    Vitis-AI-License-Restricted-Model-Zoo
        │
        ├── README.md            # This file
        │
        ├── Model A  
        │    │    
        │    ├── LICENSE.md      # License file for Model A
        │    ├── code            # Model source code 
        │    └── data            # Dataset placeholder directory      
        ├── Model B 
        │    │    
        │    ├── LICENSE.md      # License file for Model B
        │    ├── code            # Model source code 
        │    └── data            # Dataset placeholder directory  
        ...
        
Additional Model Zoo documentation and performance benchmarks are available on **[Github.io](https://xilinx.github.io/Vitis-AI/docs/workflow-model-zoo)** or **[OFFLINE](../docs/docs/workflow-model-zoo.html)**.

## Contributing

We welcome community contributions. When contributing to this repository, first discuss the change you wish to make via:

-  [GitHub Issues](https://github.com/Xilinx/Vitis-AI/issues)
-  [Vitis AI Forums](https://forums.xilinx.com/t5/AI-and-Vitis-AI/bd-p/AI)
-  <a href="mailto:xilinx_ai_model_zoo@xilinx.com">Email</a>

You can also submit a pull request with details on how to improve the product. Prior to submitting your pull request, ensure that you can build the product and run all the demos with your patch. In case of a larger feature, provide a relevant demo.


