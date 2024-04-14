# Compiler and flags
CC = clang
CFLAGS = -Wall -Wextra -Werror -I/opt/hombrew/jpeg-turbo/3.0.2/include -I/opt/hombrew/libpng/1.6.43/include
LDFLAGS = -L/opt/hombrew/jpeg-turbo/3.0.2/lib -L/opt/hombrew/libpng/1.6.43/lib
LDLIBS = -lturbojpeg -lpng

# Directories
SRC_DIR = .
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
DEPS = $(OBJS:.o=.d)

# Executable name
TARGET = $(BIN_DIR)/CNN

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

-include $(DEPS)
