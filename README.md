
# Aligning VLM Assistants with Personalized Situated Cognition (ACL 2025 main)

[![GitHub Stars](https://img.shields.io/github/stars/your-username/PCogAlign?style=social)](https://github.com/liyongqi2002/PCogAlign)
[![Hugging Face Dataset](https://img.shields.io/badge/dataset-PCogAlignBench-blue)](https://huggingface.co/datasets/YongqiLi/PCogAlignBench)
[![arXiv](https://img.shields.io/badge/arXiv-2506.00930-orange)](https://arxiv.org/abs/2506.00930)

This repository contains the official code and evaluation tools for our ACL 2025 main paper **"Aligning VLM Assistants with Personalized Situated Cognition"**. 

> ⚠️ This project is for academic research only and not intended for commercial use.



## 1 Abstract

Vision-language models (VLMs) aligned with general human objectives, such as being harmless and hallucination-free, have become valuable assistants of humans in managing visual tasks. 
However, people with diversified backgrounds have different cognition even in the same situation. Consequently, they may have personalized expectations for VLM assistants. 
This highlights the urgent need to align VLM assistants with personalized situated cognition for real-world assistance. 
To study this problem, we first simplify it by characterizing individuals based on the sociological concept of Role-Set. Then, we propose to evaluate the individuals' actions to examine whether the personalized alignment is achieved. 
Further, we construct a benchmark named PCogAlignBench, which includes 18k instances and 20 individuals with different Role-Sets. 
Finally, we present a framework called PCogAlign, which constructs a cognition-aware and action-based reward model for personalized alignment. 
Experimental results and human evaluations demonstrate the reliability of the PCogAlignBench and the effectiveness of our proposed PCogAlign.




## 2 Installation & Setup

```bash
git clone https://github.com/liyongqi2002/PCogAlign.git
cd PCogAlign
pip install -r requirements.txt
```

> ✅ Make sure you're using Python 3.10+ and have CUDA-compatible hardware if running locally.



## 3 Benchmark

Download our benchmark dataset from Hugging Face:

[![Hugging Face Dataset](https://img.shields.io/badge/dataset-PCogAlignBench-blue)](https://huggingface.co/datasets/YongqiLi/PCogAlignBench)

Replace the original empty "PCogAlignBench" with your downloaded one.



## 4 Model Preparation

You can download and place a compatible VLM model like `Qwen/Qwen2-VL-7B-Instruct` directly under the following path:

```bash
./Qwen/Qwen2-VL-7B-Instruct
```

> 💡 If your GPU memory is limited:
> - Use quantized version of the model.
> - Adjust image resolution by modifying `resized_height` / `resized_width` in `utils.py/get_vllm_input()`.



## 5 🚀 Running

To run inference on PCogAlignBench:

```bash
bash run_PCogAlign.sh
```

> This script handles preprocessing, training and test generation. You can inspect the file for more details.



## 6 Evaluation

We provide utilities to submit batch requests to OpenAI-compatible APIs for 
GPT-based evaluations in the `evaluation` folder.

- Use `eval-[Batch]-create.py` to generate a `.jsonl` batch submission file.
- Submit via OpenAI platforms and parse the output file via `eval-[Batch]-parse.py`.

> 📝 We recommend using GPT-4o or similar for best evaluation performance.


[//]: # (## Citation)

[//]: # ()
[//]: # (If you find this work useful, please cite:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@inproceedings{author2025aligning,)

[//]: # (  title={Aligning VLM Assistants with Personalized Situated Cognition},)

[//]: # (  author={Author Names},)

[//]: # (  booktitle={Proceedings of the ACL 2025 Main Conference},)

[//]: # (  year={2025})

[//]: # (})

[//]: # (```)


## 🙌 Acknowledgments

All datasets and models used are obtained through legal and ethical means. For detailed ethical considerations, please refer to our paper's Ethics Statement section.


## 📬 Contact

For any questions or feedback, feel free to reach out to us at [liyongqi@whu.edu.cn].

---

✨ Thank you for your interest in PCogAlign! Stay tuned for more updates.
