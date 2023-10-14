# Scientific Large Language Model (LLM)

This repository collects some scientific LLM papers.

<!-- >What can **Large Language Models (LLMs)** do for Protein? 

🙌 This repository collects some LLM-based protein papers.

😎 Welcome to recommend missing papers through **`Adding Issues`** or **`Pull Requests`**.  -->

<!-- ## 🔔 News
- **2023-07  We create this repository to maintain a paper list on *Large Language Models* appiled in *Protein*.**

*Todo:*
1. - [ ] `Fine-grained classification of papers`
2. - [ ] `Update paper project / code` -->

## Content
- [Scientific Large Language Model (LLM)](#scientific-large-language-model-llm)
  - [Content](#content)
  - [Text LLM Papers](#text-llm-papers)
  - [Protein LLM Papers](#protein-llm-papers)
    - [Protein sequence generation](#protein-sequence-generation)
    - [Protein function prediction](#protein-function-prediction)
  - [Molecule LLM Papers](#molecule-llm-papers)
  - [Genome LLM Papers](#genome-llm-papers)
  - [Multimodal LLM Papers](#multimodal-llm-papers)
  - [Contribution](#contribution)
    - [👥 Contributors](#-contributors)

---

## Text LLM Papers
### General model
- `2018` Improving Language Understanding by Generative Pre-Training,[OpenAI](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- `2019`Language Models are Unsupervised Multitask Learner,[OpenAI](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- `2019` BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,[NAACL](https://aclanthology.org/N19-1423/)
- `2020` Language Models are Few-Shot Learners,[arXive](https://arxiv.org/abs/2005.14165)

### Scientific model
- `2019` SciBERT: A Pretrained Language Model for Scientific Text,[arXiv](https://arxiv.org/abs/1903.10676)
- `2023` Large language models encode clinical knowledge,[nature](https://www.nature.com/articles/s41586-023-06291-2)

## Protein LLM Papers
<!-- 请仿照此格式，最好能对文章分类，然后按时间顺序添加-->
###  Protein sequence generation
- `2020` ProGen: Language Modeling for Protein Generation，[arXiv](https://doi.org/10.48550/arXiv.2004.03497)
- `2022` Large language models generate functional protein sequences across diverse families, [Nature Biotechnology]( https://doi.org/10.1038/s41587-022-01618-2)
- 
###  Protein function prediction
- `2019` A High Efficient Biological Language Model for Predicting Protein–Protein Interactions, [Cells](https://doi.org/10.3390/cells8020122)
- `2020` TRANSFORMER PROTEIN LANGUAGE MODELS ARE UNSUPERVISED STRUCTURE LEARNERS, [bioRxiv](https://doi.org/10.1101/2020.12.15.422761)
- `2021` Highly accurate protein structure prediction with AlphaFold, [nature](https://doi.org/10.1038/s41586-021-03819-2)
- `2022` Language models of protein sequences at the scale of evolution enable accurate structure prediction, [bioRxiv](https://doi.org/10.1101/2022.07.20.500902)
- `2022` Accurate prediction of protein structures and interactions using a 3-track neural network, [Science](https://doi.org/10.1126/science.abj8754)
- 

## Molecule LLM Papers
- 

## Genome LLM Papers
- 

## Multimodal LLM Papers
| Year  | Model   |  Paper   |    Language Model    |Vision Model
| :---: | :---: | :---: | :---: |:---: |
|2023.10.9|**[InternLM-XComposer-VL](https://github.com/InternLM/InternLM-XComposer)** |  **[InternLM-XComposer: A Vision-Language Large Model for Advanced Text-image Comprehension and Composition](https://arxiv.org/abs/2309.15112)**| InternLM-7B |EVA-G |
|2023.10.2|**[MMICL](https://huggingface.co/BleachNick/MMICL-Instructblip-T5-xxl)** |  **[MMICL: Empowering Vision-language Model with Multi-Modal In-Context Learning](https://arxiv.org/abs/2309.07915)**| FLANT5-XXL |EVA-G |
|2023.10.2|**[MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)** |  **[MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models](https://arxiv.org/abs/2304.10592)**| Vicuna 7B |EVA-G |
|2023.10.1|**[Pink](https://github.com/SY-Xuan/Pink)** |  **[Pink: Unveiling the Power of Referential Comprehension for Multi-modal LLMs](https://arxiv.org/abs/2310.00582)**| Vicuna-7B |CLIP ViT-L/14 |
|2023.9.30|**[Cheetor](https://github.com/DCDmllm/Cheetah)** |  **[Fine-tuning Multimodal LLMs to Follow Zero-shot Demonstrative Instructions](https://arxiv.org/abs/2308.04152)**| LLaMA2 |FlanT5-XXL |
|2023.9.29|**[LRV-Instruction](https://github.com/FuxiaoLiu/LRV-Instruction)** |  **[Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning](https://arxiv.org/abs/2306.14565)**|  | |
|2023.9.28|**[LMEye](https://github.com/YunxinLi/LingCloud)** |  **[LMEye: An Interactive Perception Network for Large Language Models](https://arxiv.org/abs/2305.03701)**| FLANT5-XL |CLIP ViT-L/14 |
|2023.9.14|**[Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL)** |  **[Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)**| Qwen-7B |ViT-G/16 |
|2023.9.14|**[Qwen-VL](https://github.com/QwenLM/Qwen-VL)** |  **[Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)**| Qwen-7B |ViT-G/16 |
|2023.8.19|**[BLIVA](https://github.com/mlpc-ucsd/BLIVA)** |  **[BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions](https://arxiv.org/abs/2308.09936)**| FLANT5-XXL |ViT-G/14 |
|2023.8.7|**[OpenFlamingo](https://github.com/mlfoundations/open_flamingo)** |  **[OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models](https://arxiv.org/abs/2308.01390)**| LLaMA 7B |CLIP ViT-L/14 |
|2023.7.30|**[Lynx](https://github.com/bytedance/lynx-llm)** |  **[What Matters in Training a GPT4-Style Language Model with Multimodal Inputs?](https://arxiv.org/abs/2307.02469)**|  | |
|2023.7.3|**[Shikra](https://github.com/shikras/shikra)** |  **[Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic](https://arxiv.org/abs/2306.15195)**| Vicuna 7B |CLIP ViT-L/14 |
|2023.6.18|**[LAMM](https://github.com/OpenLAMM/LAMM)** |  **[LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark](https://arxiv.org/abs/2306.06687)**| LLaMA 7B |CLIP ViT-L/14 |
|2023.6.15|**[InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)** |  **[InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500)**| Vicuna 7B |EVA-G |
|2023.6.15|**[BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)** |  **[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)**|  | |
|2023.6.15|**[LaVIN](https://github.com/luogen1996/LaVIN)** |  **[Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models](https://arxiv.org/abs/2305.15023)**|  | |
|2023.6.13|**[Multimodal-GPT](https://github.com/open-mmlab/Multimodal-GPT)** |  **[MultiModal-GPT: A Vision and Language Model for Dialogue with Humans](https://arxiv.org/abs/2305.04790)**| LLaMA 7B |CLIP ViT-L/14 |
|2023.5.25|**[PandaGPT](https://github.com/yxuansu/PandaGPT)** |  **[PandaGPT: One Model To Instruction-Follow Them All](https://arxiv.org/abs/2305.16355)**| Vicuna 13B |ImageBind ViT-H/14 |
|2023.5.5|**[Otter-I](https://github.com/Luodian/Otter)** |  **[Otter: A Multi-Modal Model with In-Context Instruction Tuning](https://arxiv.org/abs/2305.03726)**| LLaMA 7B |CLIP ViT-L/14 |
|2023.5.2|**[VPGTrans](https://github.com/VPGTrans/VPGTrans)** |  **[Transfer Visual Prompt Generator across LLMs](https://arxiv.org/abs/2305.01278)**|  | |
|2023.4.28|**[LLaMA-Adapter-v2](https://github.com/ZrrSkywalker/LLaMA-Adapter)** |  **[LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model](https://arxiv.org/abs/2304.15010)**| LLaMA 7B |CLIP ViT-L/14 |
|2023.4.27|**[mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl)** |  **[mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality](https://arxiv.org/abs/2304.14178)**| LLaMA2 7B |CLIP ViT-L/14 |
|2023.4.17|**[LLaVA](https://github.com/haotian-liu/LLaVA)** |  **[Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)**| LLaMA 7B |CLIP ViT-L/14 |
|2022.12.15|**[GIT2](https://github.com/microsoft/GenerativeImage2Text)** |  **[GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100)**|  |CLIP ViT-L/14 |
||**[ImageBind-LLM](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main)** |  **[nan]()**| Open-Chinese-LLaMA-7B |imagebind_huge |
||**[IDEFICS-80B-Instruct](https://huggingface.co/HuggingFaceM4/idefics-80b-instruct)** |  **[nan]()**| LLaMA65B |CLIP ViT-H/14 |
||**[VisualGLM](https://github.com/THUDM/VisualGLM-6B)** |  **[nan]()**| ChatGLM 6B |EVA-CLIP |
||**[WeMM](https://github.com/scenarios/WeMM/tree/main)** |  **[nan]()**|  | |
||**[JiuTian-Tiny/JiuTian](https://github.com/rshaojimmy/JiuTian)** |  **[nan]()**|  | |


## Contribution
### 👥 Contributors


<!-- ### 🎉 Contributing ( welcome ! )

- ✨ Add a new paper or update an existing Protein-related LLM paper.
- 🧐 Use the same format as existing entries to describe the work.
- 😄 A very brief explanation why you think a paper should be added or updated is recommended (Not Neccessary) via **`Adding Issues`** or **`Pull Requests`**.

**Don't worry if you put something wrong, they will be fixed for you. Just feel free to contribute and promote your awesome work here! 🤩 We'll get back to you in time ~ 😉** -->


