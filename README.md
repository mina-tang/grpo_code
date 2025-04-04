
> [!NOTE] 
> Check out our [blog-post](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) for more detail and benchmarks!

## Installation

```bash
git clone https://github.com/axolotl-ai-cloud/grpo_code.git
cd grpo_code
pip install -e .
pip install axolotl==0.8.0[vllm,flash-attn]
```

## Training

The following environment variables can be used to modify the behaviour of the reward functions:
- `WASM_FUEL` - Controls the amount of fuel (computation resources) allocated to the WASM environment (default: 10000000000)
- `WASM_PATH` - Path to the Python WASM runtime file (default: "./wasm/python-3.12.0.wasm")
- `TIMEOUT` - Maximum execution time in seconds for code evaluation (default: 1)
- `MAX_WORKERS` - Number of parallel workers for multiprocessing reward functions (default: 1)

First, spin up a `vLLM` instance:

```bash
CUDA_VISIBLE_DEVICES=2,3 axolotl vllm-serve r1_acecode.yaml
```

Then, in another terminal, kick off the training process:

```bash
CUDA_VISIBLE_DEVICES=0,1 MAX_WORKERS=64 axolotl train r1_acecode.yaml --num-processes 2
```

This example uses 4 A100 GPUs - adjust `CUDA_VISIBLE_DEVICES`, `MAX_WORKERS`, `cfg.micro_batch_size` and `cfg.gradient_accumulation_steps` as necessary to match your hardware.

## Python WASM Runtime

This project uses Python 3.12.0 compiled to WebAssembly from VMware Labs.

### Verify an Existing Download
If you already have the WASM file and want to verify its integrity:

1. Ensure you have both `python-3.12.0.wasm` and `python-3.12.0.wasm.sha256sum` in the `wasm` directory.
2. Run the verification command:

**Linux/macOS:**
```bash
sha256sum -c ./wasm/python-3.12.0.wasm.sha256sum
```

### Manual Download
To download the runtime files yourself:

1. Download the Python WASM runtime:
   ```bash
   curl -LO https://github.com/vmware-labs/webassembly-language-runtimes/releases/download/python%2F3.12.0%2B20231211-040d5a6/python-3.12.0.wasm -o ./wasm/python-3.12.0.wasm
   ```

2. Download the SHA256 checksum file:
   ```bash
   curl -LO https://github.com/vmware-labs/webassembly-language-runtimes/releases/download/python%2F3.12.0%2B20231211-040d5a6/python-3.12.0.wasm.sha256sum -o ./wasm/python-3.12.0.wasm.sha256sum
   ```

3. Verify the download:
   ```bash
   sha256sum -c ./wasm/python-3.12.0.wasm.sha256sum
   ```

4. Place both files in your project directory or specify the path in your configuration.
