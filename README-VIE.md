[🇺🇸 View English version](README.md)

# RiscV-KyX1-LLM-Demo
Demo chạy mô hình ngôn ngữ lớn (LLM) trên SoC RISC-V KyX1 của Orange Pi RV2. 
Hỗ trợ bởi [Orange Pi Vietnam](https://www.facebook.com/orangepivietnam/)

## Chạy mô hình:
Đây là mã mẫu để chạy các mô hình đã chuyển đổi sẵn trên Orange Pi RV2. Mô hình nhỏ như 1B hoặc 1.5B token có thể chạy trên RV2 với 4GB RAM; nhưng các mô hình lớn như 8B hoặc 3.8B cần RV2 với 8GB RAM.

### Tải về mã nguồn
```bash
https://github.com/thanhtantran/RiscV-KyX1-LLM-Demo
cd RiscV-KyX1-LLM-Demo/python
```

### Cài đặt thư viện phụ thuộc
```bash
pip3 install ./onnxruntime_genai-0.4.0.dev1-cp312-cp312-linux_riscv64.whl ./ky_ort-1.2.2-cp312-cp312-linux_riscv64.whl --break-system-packages
export PATH="$PATH:/home/orangepi/.local/bin"
```

### Tải các mô hình đã chuyển đổi sẵn
```bash
pip install gdown --break-system-packages
sudo chmod +x download_models.sh
bash download_models.sh
```

Có 4 mô hình đã được chuyển đổi sẵn cho bạn
| GDrive ID | File name | Model name |
|---|---| ---|
| 1XVLXUlrJZyOwDlrqOwyH4kxhAd9d5QWz | minicpm-1b-int4-blk64-fusion.tar.gz | |
| 1N9sHii6Cl5UyKS59l8DD3s2W23fF9t6k | phi-3-mini-int4-3.8b.tar.gz | phi3 |
| 1Q7qyorYStCm3gv2jQUNODBO6m63223mQ | llama3-int4-8b-blk64-fusion.tar.gz | llama3 |
| 1g3_Ni7sZg-_JR8u9Kx8hxHzPx9bE-Z-k | qwen2-int4-1.5b.tar.gz | qwen2 |

Lệnh sử dụng
```bash
python3 llm_qa.py --help
usage: llm_qa.py [-h] -m MODEL -e {qwen2,minicpm,tinyllama,phi2,gemma,phi3,llama3} [-i MIN_LENGTH] [-l MAX_LENGTH] [-ds] [-p TOP_P] [-k TOP_K] [-t TEMPERATURE]
                 [-r REPETITION_PENALTY] [-v] [-g]

End-to-end AI Question/Answer example for gen-ai

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Onnx model folder path (must contain config.json and model.onnx)
  -e {qwen2,minicpm,tinyllama,phi2,gemma,phi3,llama3}, --type {qwen2,minicpm,tinyllama,phi2,gemma,phi3,llama3}
                        model type
  -i MIN_LENGTH, --min_length MIN_LENGTH
                        Min number of tokens to generate including the prompt
  -l MAX_LENGTH, --max_length MAX_LENGTH
                        Max number of tokens to generate including the prompt
  -ds, --do_sample      Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false
  -p TOP_P, --top_p TOP_P
                        Top p probability to sample with
  -k TOP_K, --top_k TOP_K
                        Top k tokens to sample from
  -t TEMPERATURE, --temperature TEMPERATURE
                        Temperature to sample with
  -r REPETITION_PENALTY, --repetition_penalty REPETITION_PENALTY
                        Repetition penalty to sample with
  -v, --verbose         Print verbose output and timing information. Defaults to false
  -g, --timings         Print timing information for each generation step. Defaults to false
```

Ví dụ
```bash
python3 llm_qa.py -m ./models/qwen2-int4-1.5b -l 128 -e qwen2 -v -g
```
Kết quả bạn nhận được
```bash
orangepi@orangepirv2:~/RiscV-KyX1-LLM-Demo$ python3 llm_qa.py -m ./models/qwen2-int4-1.5b -l 128 -e qwen2 -v -g
Loading model...
Model loaded
Input: Hello! Who are you?
<|start_header_id|>user<|end_header_id|>Hello! Who are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Generator created
Running generation loop ...

Output: I'm an AI language model. How can I assist you today? Please provide more information about your question or concern.<|eot_1|>

Prompt length: 50, New tokens: 32, Time to first: 5.65s, Prompt tokens per second: 8.86 tps, New tokens per second: 5.36 tps
Input: 
```

## Chuyển đổi mô hình:
Quá trình này nên chạy trên máy X86 mạnh, không phải Orange Pi

### Chuẩn bị môi trường
```bash
cd python/genai-builder
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages
```

### Chuyển đổi mô hình
```bash
python builder.py \
    -m Qwen/Qwen2-1.5B \
    -o qwen2-1.5b-int4-blk64 \
    -p int4 \
    -e cpu \
    -c model_cache \
    --extra_options int4_accuracy_level=4 int4_block_size=64
# Recommended parameter accuracy_level=4 (W4A8 GroupWise quantization, block_size=64)
```

Bạn có thể mua Orange Pi RV2 tại http://orangepi.net
