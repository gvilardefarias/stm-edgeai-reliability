# StellarE AI Battery Management System (BMS)

## KerasID001 AI BMS Test Application

### Description:

This application shows an example of application based on the SR5E1-AI Component.

In particular this application uses the SR5E1-AI to deploy a neural network
able to predict the releasable capacity (Ampere hour) of a rechargeable battery.
The releasable capacity is a fundamental parameter for a battery health
monitoring system [1][2].
This neural network has been designed using tensorflow and trained using the
NASA Li-ion Battery Aging Dataset available at
https://data.nasa.gov/dataset/Li-ion-Battery-Aging-Datasets/uj5r-zjdb .
A subset of this dataset has been used to feed the neural network and show the
functionality.

[1] G. Crocioni, D. Pau and G. Gruosso, "Li-Ion Batteries Releasable Capacity
Estimation with Neural Networks on Intelligent IoT Microcontrollers", 2020 IEEE
20th Mediterranean Electrotechnical Conference ( MELECON), Palermo, Italy, 2020,
pp. 153-158, doi: 10.1109/MELECON48756.2020.9140634.

[2] G. Crocioni, D. Pau, J. Delorme and G. Gruosso, "Li-Ion Batteries Parameter
Estimation With Tiny Neural Networks Embedded on Intelligent IoT Microcontrollers",
in IEEE Access, vol. 8, pp. 122135-122146, 2020, doi: 10.1109/ACCESS.2020.3007046.

If the options Allocate Inputs/Allocate Outputs are enabled in the Network
Settings, the activations buffer will be also used to handle the input buffers/
output buffers. This allows to reduce the memory usage. Please, refer to the
SR5E1-AI User Guide for more details.

If the Validate is enabled, a specific Validate Project will be compiled.

In order to receive from serial, please set the baud rate of the COM port on
your PC to 115200.

### Supported Compilers:

- GNU ARM Embedded Toolchain 13.3.1-2024.06
- HighTec clang version 8.1.0
- IAR ANSI C/C++ Compiler V9.30.1.335/W64

### Supported Platforms:

---

#### EVBE7000P

Requirements:

1.  From the SR5E1_AI_Demo_7000P.ai (SR5E1 AI) model, select the BMS Keras neural network model
    "CNN_LSTM.h5" file under the "source\model_lstm" folder
    - the Name of the network must be network
    - the Type of the network must be Keras
2.  Analyze the neural network chosen pressing the "Analyze" button (optional)
    - a report will be generated under a new "SR5E1_AI_Demo_7000P\cfg" folder
3.  Generate the neural network files of the SR5E1_AI_Demo_7000P.ai model pressing the "Generate" button
    - the "cfg" and "lib" folders will be created automatically under a new SR5E1_AI_Demo_7000P folder
    - the content of the "cfg" folder depends on the validate check box (if enabled or not)
      - for default the validate check box is enabled
      - for SR5E1-EVBE7000P, you have to select the SR5E1_UART3 as serial port for validation (UART via USB_PORT - P26 on board)
    - a makefile will be also automatically created to build the SR5E1_AI_Demo_7000P Test Application
4.  Build the SR5E1_AI_Demo_7000P Test Application using one of the supported compilers
5.  Connect the Host serial port to UART via USB_PORT (P26)
6.  Configure the Host serial port baudrate to 115200
7.  - If Validation is disabled:
      - run the SR5E1_AI_Demo_7000P inference demo test application
    - If Validation is enabled:
      - run the SR5E1_AI_Demo_7000P valiadtion test application
        - some info will be displayed on the serial terminal and the application
          will wait for the validation data from the serial
      - disconnect/close the Host serial port
      - select the Com Port from the SR5E1_AI_Demo_7000P.ai model Validate Tab
        and send the valiadation data (random or not) pressing the "Validate" button
      - a final validation report will be generated under a new "SR5E1_AI_Demo_7000P\cfg" folder

---
