BASE_FONT=../OpenSans-Regular.ttf
TARGET_FONT=llama.ttf
WASM=translatettf_bg.wasm

HG_REPO=https://huggingface.co/lmz/candle-quantized-t5
MODEL=model.gguf
TOKENIZER=tokenizer.json
CONFIG=config.json

$(TARGET_FONT): pkg/$(WASM) $(BASE_FONT)
	../bin/otfsurgeon -i $(BASE_FONT) add -o $(TARGET_FONT) Wasm < pkg/$(WASM)

pkg/$(WASM): src/lib.rs $(MODEL) $(TOKENIZER) $(CONFIG)
	wasm-pack build --target web

$(MODEL):
	curl -s -S -L -o $(MODEL) $(HG_REPO)/resolve/main/$(MODEL)

$(TOKENIZER):
	curl -s -S -L -o $(TOKENIZER) $(HG_REPO)/resolve/main/$(TOKENIZER)

$(CONFIG):
	curl -s -S -L -o $(CONFIG) $(HG_REPO)/resolve/main/$(CONFIG)
