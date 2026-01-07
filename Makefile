CC = gcc
CFLAGS = -Iinclude -Wall -Wextra -g -MMD -MP -O3
LDFLAGS = -lm
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Common sources (Math, Dataset, Layers, VAE)
COMMON_SRCS = src/math_extra.c \
              src/dataset.c \
              src/vae.c \
              $(wildcard src/layers/*.c)

COMMON_OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(COMMON_SRCS))

# Train target
TRAIN_TARGET = $(BIN_DIR)/train_vae
TRAIN_SRC = src/train.c
TRAIN_OBJ = $(OBJ_DIR)/train.o

# Arithmetic target
ARITHMETIC_TARGET = $(BIN_DIR)/arithmetic
ARITHMETIC_SRC = src/arithmetic_demo.c
ARITHMETIC_OBJ = $(OBJ_DIR)/arithmetic_demo.o

# Dummy Net (Old)
DUMMY_TARGET = $(BIN_DIR)/dummy_net
DUMMY_SRC = src/dummy_net.c src/main.c
DUMMY_OBJ = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(DUMMY_SRC))

DEPS = $(COMMON_OBJS:.o=.d) $(TRAIN_OBJ:.o=.d) $(ARITHMETIC_OBJ:.o=.d) $(DUMMY_OBJ:.o=.d)

all: $(TRAIN_TARGET) $(ARITHMETIC_TARGET)

$(TRAIN_TARGET): $(COMMON_OBJS) $(TRAIN_OBJ) | $(BIN_DIR)
	$(CC) $(COMMON_OBJS) $(TRAIN_OBJ) -o $@ $(LDFLAGS)

$(ARITHMETIC_TARGET): $(COMMON_OBJS) $(ARITHMETIC_OBJ) | $(BIN_DIR)
	$(CC) $(COMMON_OBJS) $(ARITHMETIC_OBJ) -o $@ $(LDFLAGS)

$(DUMMY_TARGET): $(COMMON_OBJS) $(DUMMY_OBJ) | $(BIN_DIR)
	$(CC) $(COMMON_OBJS) $(DUMMY_OBJ) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

-include $(DEPS)

.PHONY: all clean
