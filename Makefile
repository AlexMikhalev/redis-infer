REDIS_INC ?= /usr/local/include
CC = gcc
CFLAGS = -O3 -fPIC -Wall -I$(REDIS_INC) -pthread
LDFLAGS = -shared -lm -pthread

all: src/redis-infer.so

src/redis-infer.so: src/redis-infer.o
	$(CC) $(LDFLAGS) -o $@ $<

src/redis-infer.o: src/redis-infer.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f src/redis-infer.o src/redis-infer.so
