Commented IRQ_HANDLER(IRQ_TIM2_HANDLER) in
	Modules/Drivers/Timers/TIM/src/sr5e1/tim_dev.c:612
	
Modified files
	SR5E1_AI_Demo_7000P/lib/validation/src/aiValidation_ST_AI.c
	
Assumptions for generalize Memory protections

	- A single activations buffer, max 8 KB. (If does not match, it does not compile)
	- Input in activations and float32 outputs: It works if the network is generated with 
		allocate-inputs/outputs and is not quantized.
	- no state buffer in RAM
	- Inputs large up to 1 KB. (If does not match, it does not compile)
	- No other RAM area witten by library during layersr. To verify it, just launch a Validation with
		active MPU and no bit flip. fi_n_crash = 0.
	
What changes with a different Neural Network

In fault_injection.h
	- FI_NODE_IDS/ FI_N_NODES: check Id layer and so number of layers (SR5E1_AI_Demo_7000P/cfg/inc/network_details.h)
	- FI_INPUT_NODE_ID: define layer that reads input (cfg/network_c_info.json)
	- FI_LAYERS_US: layer time period of the inference (cfg/network_validate_report.txt)	
	
In fault_injection.c
-	FI_IN_OFFSET: define input position of activations and geometry of regions 6 & 7
	(If does not match, it does not compile)

