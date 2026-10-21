################################################################################
# Project makefile.
################################################################################

# 'Stellar SDK' project values
PROJECTNAME := SR5E1_AI_Demo_7000P
SDKID := StellarESDK-1.10.0

# Please Update it if you move your workspace
STELLAR_E_SDK_RELATIVE_PATH := ../../SDKS/$(SDKID)
PROJECT_COMMON_DIR := $(STELLAR_E_SDK_RELATIVE_PATH)/Projects/SDKTests/CommonBuild

CONFIG_TARGET_MEMORY = flash
CONFIG_BOARD = evbe7000p

include $(PROJECT_COMMON_DIR)/make/test_defs.mk

ifeq ($(wildcard $(STELLAR_E_SDK_BUILD_SYSTEM_DIR)/StellarESDK.mk),)
$(error $(STELLAR_E_SDK_BUILD_SYSTEM_DIR)/StellarESDK.mk does not exist)
$(error Install the $(SDKID) or Correct STELLAR_E_SDK_RELATIVE_PATH)
endif

################################################################################
# Project builds
################################################################################
BUILD_OS_OSAL                  := 1
BUILD_DRIVERS_DWT              := 1
BUILD_DRIVERS_DMA              := 1
BUILD_DRIVERS_UART             := 1
BUILD_RUNTIME_IO               := 1
BUILD_DRIVERS_TIM              := 1
BUILD_DRIVERS_IRQ              := 1

################################################################################
# Add project files
################################################################################

# Application name
APP_NAME := $(PROJECTNAME)

################################################################################
# Include 'Stellar AI' project makefile
################################################################################

# C sources
C_SRCS += \
    main.c \
    app/src/fault_injection.c

# Headers files
C_INCS += app/inc

include ./stellar_ai.mk

# Fault injection: intercetta run e set_callback dell'API multi-network
ifeq ($(TOOLCHAIN), ARM)
L_OPTS += -Wl,--wrap=stai_mnetwork_run -Wl,--wrap=stai_mnetwork_set_callback
endif

################################################################################
# Linker script files
################################################################################

# Linker script include
ifeq ($(TOOLCHAIN), ARM)
LINKER_SCRIPT_INCS := ./rsc/$(CONFIG_DEVICE)/$(TARGET_BUILD)
else ifeq ($(TOOLCHAIN), HIGHTEC)
LINKER_SCRIPT_INCS := ./rsc/$(CONFIG_DEVICE)/$(TARGET_BUILD)
endif

################################################################################
# Include 'Stellar SDK' top level makefile
################################################################################
include $(STELLAR_E_SDK_BUILD_SYSTEM_DIR)/StellarESDK.mk

################################################################################
# Define 'Stellar SDK' compiling target
################################################################################
all: default-all
