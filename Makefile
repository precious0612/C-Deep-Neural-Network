CC = clang
CFLAGS = -Wall -Wextra -I/usr/local/include/stb -I/opt/homebrew/Cellar/json-c/0.17/include -I/opt/homebrew/Cellar/json-c/0.17/include/json-c -I/opt/homebrew/Cellar/jpeg-turbo/3.0.2/include -I/opt/homebrew/opt/libpng/include/libpng16
LDFLAGS = -L/opt/homebrew/Cellar/json-c/0.17/lib -L/opt/homebrew/Cellar/jpeg-turbo/3.0.2/lib -L/opt/homebrew/opt/libpng/lib
LDLIBS = -ljson-c -lturbojpeg -lpng16 -lz

# 排除 main.c
SOURCES = $(filter-out main.c, $(wildcard *.c input/data.c model/model.c model/layer/layer.c optimizer/optimizer.c utils/*.c utils/compute/*.c))
OBJECTS = $(patsubst %.c, %.o, $(SOURCES))
HEADERS = $(wildcard *.h input/*.h model/*.h model/layer/*.h optimizer/*.h utils/*.h utils/compute/*.h)
EXAMPLE_EXECUTABLE = main

all: $(EXAMPLE_EXECUTABLE)

$(EXAMPLE_EXECUTABLE): main.o $(OBJECTS)
	$(CC) $(LDFLAGS) main.o $(OBJECTS) $(LDLIBS) -o $@

main.o: main.c $(HEADERS)
	$(CC) $(CFLAGS) -c main.c -o main.o

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) 

run_example: $(EXAMPLE_EXECUTABLE)
	./$(EXAMPLE_EXECUTABLE)

.PHONY: all clean run_example
