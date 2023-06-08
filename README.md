<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

![Release Version](https://img.shields.io/github/v/release/Xilinx/Vitis-AI-Copyleft-Model-Zoo)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr-raw/Xilinx/Vitis-AI-Copyleft-Model-Zoo)
[![Documentation](https://img.shields.io/badge/documentation-github.IO-blue.svg)](https://xilinx.github.io/Vitis-AI/docs/workflow-model-zoo)
![Repo Size](https://img.shields.io/github/repo-size/Xilinx/Vitis-AI-Copyleft-Model-Zoo)

# Vitis AI Copyleft Model Zoo

The purpose of this repository is to provide enablement to developers choosing to use popular models that are not compatible with the Vitis™ AI Apache 2.0 license.  The original source code for these models was released under a copyleft, reciprocal, or otherwise non-permissive license.  The purpose of releasing these models in this separate repository is to clearly identify the source license (inherited license) for each model.

In order for users to leverage these models with Vitis AI, model source code modifications are required.  This repository provides users with the required source-code modifications to ensure compatibility with Vitis AI. Pre-compiled versions of these models are not provided because the Vitis AI Compiler is not open-source.  Users, at their discretion, and subject to the terms of the inherited license, may leverage the scripts provided as a template to compile the model for their target using the Vitis AI Compiler.

The directory structure of this repository is such that each model is uniquely associated with a specific license file, allowing the user to evaluate the license requirements for each model.  It is the user's responsibility to verify that the model license is compatible with their own company policies and legal requirements.  

AMD is releasing each model under the terms of the inherited license for that model.  No additional license is associated with, or otherwise implied, for the contents of this repository.

        
    Vitis-AI-Copyleft-Model-Zoo
        │
        ├── README.md            # This file
        │
        ├── Model A  
        │    │    
        │    ├── LICENSE         # License file for Model A
        │    └── ...             # Sources for Model A
        │
        ├── Model B 
        │    │    
        │    ├── LICENSE         # License file for Model B
        │    └── ...             # Sources for Model B
        │
        ...
        
Additional Model Zoo documentation and performance benchmarks are available on **[GITHUB.IO](https://xilinx.github.io/Vitis-AI/docs/workflow-model-zoo)** or **[OFFLINE](../docs/docs/workflow-model-zoo.html)**.

## Contributing

We welcome community contributions. When contributing to this repository, first discuss the change you wish to make via:

-  [GitHub Issues](https://github.com/Xilinx/Vitis-AI/issues)
-  [Vitis AI Forums](https://support.xilinx.com/s/topic/0TO2E000000YKY9WAO/vitis-ai-ai?language=en_US)
-  <a href="mailto:xilinx_ai_model_zoo@amd.com">Email</a>

You can also submit a pull request with details on how to improve the product. Prior to submitting your pull request, ensure that you can build the product and run all the demos with your patch. In case of a larger feature, provide a relevant demo.


