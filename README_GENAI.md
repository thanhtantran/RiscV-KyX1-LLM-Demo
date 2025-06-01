## GENAI

---

[ONNXRuntime.GENAI](https://onnxruntime.ai/docs/genai/)

---

#### Support Models
- [X] [Qwen2 1.5B](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) [Qwen2 0.5B](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
- [X] [Qwen1.5 4B](https://huggingface.co/Qwen/Qwen1.5-4B-Chat)
- [X] [Phi-2 2.7B](https://huggingface.co/microsoft/phi-2)
- [X] [Phi-3 3.8B](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [X] [Phi-3-V 3.8B](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)
- [X] Llama-3
- [X] Llama-2
- [X] [MiniCPM-llama-1B](https://huggingface.co/openbmb/MiniCPM-S-1B-sft-llama-format)
- [X] Gemma
- [X] Gemma2
- [X] Mistral
- [ ] ChatGLM2
- [ ] ChatGLM3
- [ ] InternLM2

#### Convert model
1. Prepare environment
```bash
cd python/genai-builder
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. Convert the model
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

#### 预构建模型
1. qwen2-1.5b
2. minicpm-llama-1b

#### LLM Chat
1. C & C++
参考samples中的chatllm_demo.cpp

1. python
参考samples中的llm_qa.py
~~~ bash
# 安装genai python版本
pip install onnxruntime_genai-0.4.0.dev0-cp312-cp312-linux_riscv64.whl --break-system-packages
~~~
