CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra

covert-channel-send: covert_channel/fast_sender.c covert_channel/message_codec.h covert_channel/interleave_order.h
	$(CC) $(CFLAGS) -pthread -o $@ covert_channel/fast_sender.c

.PHONY: clean
clean:
	rm -f covert-channel-send
