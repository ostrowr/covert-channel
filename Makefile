CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra

.PHONY: all clean
all: covert-channel covert-channel-decoder

build/send.o: covert_channel/fast_sender.c covert_channel/message_codec.h covert_channel/interleave_order.h
	mkdir -p build
	$(CC) $(CFLAGS) -pthread -Dmain=cc_send_main -c -o $@ $<

covert-channel: covert_channel/main.c build/send.o
	$(CC) $(CFLAGS) -pthread -o $@ covert_channel/main.c build/send.o

covert-channel-decoder: covert_channel/decode_cli.py $(wildcard covert_channel/*.py)
	uvx --from pyinstaller --with 'numpy>=2,<3' pyinstaller --noconfirm --clean --onefile \
		--name covert-channel-decoder --paths . --distpath . \
		--workpath build/pyinstaller --specpath build/pyinstaller covert_channel/decode_cli.py

clean:
	rm -f covert-channel covert-channel-decoder
	rm -rf build
