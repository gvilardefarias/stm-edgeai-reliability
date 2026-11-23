################################################################################
# Project makefile for AI validation.
################################################################################

# AI auto-genereted folder
DIR_NAME := SR5E1_AI_Demo_7000P

################################################################################
# Add project files
################################################################################


# C validation sources
C_SRCS += \
	$(DIR_NAME)/lib/stellar/src/stellar_ai.c \
	$(DIR_NAME)/lib/validation/src/aiPbIO.c \
	$(DIR_NAME)/lib/validation/src/aiPbMemRWServices.c \
	$(DIR_NAME)/lib/validation/src/aiPbMgr.c \
	$(DIR_NAME)/lib/validation/src/aiValidation_ST_AI.c \
	$(DIR_NAME)/lib/validation/src/pb_common.c \
	$(DIR_NAME)/lib/validation/src/pb_decode.c \
	$(DIR_NAME)/lib/validation/src/pb_encode.c \
	$(DIR_NAME)/lib/validation/src/stellar_ai_validate.c \
	$(DIR_NAME)/lib/validation/src/stm32msg.pb.c \
	$(DIR_NAME)/lib/misc/src/ai_device_adaptor.c \
	$(DIR_NAME)/lib/misc/src/aiTestHelper_ST_AI.c \
	$(DIR_NAME)/lib/misc/src/aiTestUtility.c \
	$(DIR_NAME)/lib/misc/src/syscalls.c

# C network sources
C_SRCS += \
	$(DIR_NAME)/cfg/src/network_data.c \
	$(DIR_NAME)/cfg/src/network.c

C_SRCS += \
	$(DIR_NAME)/cfg/app_stellar-studio-ai.c

# C network and validation includes
C_INCS += \
	$(DIR_NAME)/lib/ai/inc \
	$(DIR_NAME)/lib/stellar/inc \
	$(DIR_NAME)/cfg/inc \
	$(DIR_NAME)/cfg

# C validation includes
C_INCS += \
	$(DIR_NAME)/lib/validation/inc \
	$(DIR_NAME)/lib/misc/inc

# Network runtime library includes
ifeq ($(TOOLCHAIN), ARM)
LIBS += \
	$(DIR_NAME)/lib/ai/Lib/GCC/StellarE/NetworkRuntime1200_CM7_GCC.a \
	-lm
else ifeq ($(TOOLCHAIN), HIGHTEC)
LIBS += \
	$(DIR_NAME)/lib/ai/Lib/HT/StellarE/NetworkRuntime1200_CM7_HT.a
else ifeq ($(TOOLCHAIN), ARMCLANG)
LIBS += \
	$(DIR_NAME)/lib/ai/Lib/ARMCLANG/StellarE/NetworkRuntime1200_CM7_ARMCLANG.a
else ifeq ($(TOOLCHAIN), IAR)
LIBS += \
	$(DIR_NAME)/lib/ai/Lib/IAR/StellarE/NetworkRuntime1200_CM7_IAR.a
endif

# C target mcu macros
C_DEFS += \
	SR5E1


# C network info macros
C_DEFS += HAVE_NETWORK_INFO
