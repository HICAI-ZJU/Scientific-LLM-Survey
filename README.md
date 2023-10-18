# Scientific Large Language Models (LLMs)

This repository collects some scientific LLM papers. Welcome to follow and star!

<!-- > 😎 Welcome to recommend missing papers through **`Adding Issues`** or **`Pull Requests`**.  -->

<!-- ## 🔔 News
- **2023-07  We create this repository to maintain a paper list on *Large Language Models* appiled in *Protein*.**

*Todo:*
1. - [ ] `Fine-grained classification of papers`
2. - [ ] `Update paper project / code` -->

## Contents
- [Text LLM Papers](#text-llm-papers)
  <!-- - [General model](#general-model)
  - [Scientific model](#scientific-model) -->
- [Protein LLM Papers](#protein-llm-papers)
  <!-- - [Protein sequence generation](#protein-sequence-generation)
  - [Protein function prediction](#protein-function-prediction) -->
- [Molecule LLM Papers](#molecule-llm-papers)
  <!-- - [General Molecular model](#general-molecular-model)
  - [Drug model](#drug-model) -->
- [Genome LLM Papers](#genome-llm-papers)
- [Multimodal LLM Papers](#multimodal-llm-papers)
  <!-- - [Protein Multimodal Paper](#protein-multimodal-paper)
  - [Molecular Multimodal Paper](#molecular-multimodal-paper)
  - [Gene Multimodal Paper](#gene-multimodal-paper) -->

---

<!-- 请仿照此格式，对文章分类，然后按时间倒序添加-->

## 📖 Text LLM Papers
<!-- ### General model
- `2018` Improving Language Understanding by Generative Pre-Training,[OpenAI](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- `2019` Language Models are Unsupervised Multitask Learner,[OpenAI](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- `2019` BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,[NAACL](https://aclanthology.org/N19-1423/)
- `2020` Language Models are Few-Shot Learners,[arXiv](https://arxiv.org/abs/2005.14165)
- `2023` LLaMA: Open and Efficient Foundation Language Models,[arXiv](https://arxiv.org/abs/2302.13971)
- `2023` Alpaca: A Strong, Replicable Instruction-Following Model,[arXiv](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- `2023` Llama 2: Open Foundation and Fine-Tuned Chat Models,[arXiv](https://arxiv.org/abs/2307.09288) -->

<!-- 自然科学包括数学、物理学、化学、生物学、天文学等基础科学和医学、农学、气象学、材料学等应用科学。我们主要做生命科学(Life science)：蛋白、分子、基因，重点收集这些文章-->

### Comprehensive
- `2023` DARWIN Series: Domain Specific Large Language Models for Natural Science,[arXiv](https://arxiv.org/abs/2308.13565)
- `2019` SciBERT: A Pretrained Language Model for Scientific Text,[arXiv](https://arxiv.org/abs/1903.10676)
- `2022` Galactica: A Large Language Model for Science,[arXiv](https://arxiv.org/abs/2211.09085)
### Biology
- `2020` BioMegatron: Larger Biomedical Domain Language Model,[arXiv](https://arxiv.org/abs/2010.06060)
- `2019` BioBERT: a pre-trained biomedical language representation model for biomedical text mining,[arXiv](https://arxiv.org/abs/1901.08746)
### Chemistry

### Physics
- `2023` Can Language Models Understand Physical Concepts?,[EMNLP](https://arxiv.org/pdf/2305.14057.pdf)
### Medicine
- `2022` A large language model for electronic health records,[nature npj digital medicine](https://www.nature.com/articles/s41746-022-00742-2)
- `2023` Large language models encode clinical knowledge,[nature](https://www.nature.com/articles/s41586-023-06291-2)
- `2023` ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge,[arXiv](https://arxiv.org/abs/2303.14070)
- `2023` DoctorGLM: Fine-tuning your Chinese Doctor is not a Herculean Task,[arXiv](https://arxiv.org/abs/2304.01097)
### Geography
- `2023` K2: A Foundation Language Model for Geoscience Knowledge Understanding and Utilization.[arXiv](https://arxiv.org/abs/2306.05064)
### Materialogy

### Mathematics

### Agriculture

### Others


## 🧬 Protein LLM Papers
<!-- 参考 https://github.com/opendilab/awesome-AI-based-protein-design
https://github.com/yangkky/Machine-learning-for-proteins
https://github.com/LirongWu/awesome-protein-representation-learning -->
###  Protein sequence generation (or design) 
- `2020` ProGen: Language Modeling for Protein Generation，[arXiv](https://doi.org/10.48550/arXiv.2004.03497)
- `2022` Large language models generate functional protein sequences across diverse families, [Nature Biotechnology]( https://doi.org/10.1038/s41587-022-01618-2)
###  Protein function (or property) prediction
- `2019` A High Efficient Biological Language Model for Predicting Protein–Protein Interactions, [Cells](https://doi.org/10.3390/cells8020122)
- `2023` A Multimodal Protein Representation Framework for Quantifying Transferability Across Biochemical Downstream Tasks, [Advance Science](https://onlinelibrary.wiley.com/doi/10.1002/advs.202301223)
- `2023` Enhancing Protein Language Models with Structure-based Encoder and Pre-training, [arXiv](https://arxiv.org/abs/2303.06275)
###  Protein structure prediction
- `2021` Highly accurate protein structure prediction with AlphaFold, [nature](https://doi.org/10.1038/s41586-021-03819-2)
- `2022` Language models of protein sequences at the scale of evolution enable accurate structure prediction, [bioRxiv](https://doi.org/10.1101/2022.07.20.500902)
- `2020` TRANSFORMER PROTEIN LANGUAGE MODELS ARE UNSUPERVISED STRUCTURE LEARNERS, [bioRxiv](https://doi.org/10.1101/2020.12.15.422761)
- `2022` Accurate prediction of protein structures and interactions using a 3-track neural network, [Science](https://doi.org/10.1126/science.abj8754)
- `2023` Enhancing the Protein Tertiary Structure Prediction by Multiple Sequence Alignment Generation, [arXiv](https://arxiv.org/abs/2306.01824)

### Others


## 🧪 Molecule LLM Papers
<!-- 参考 https://github.com/OmicsML/awesome-molecule-protein-pretrain-papers -->
<!-- 只考虑分子单模态的，可以标注一下是基于sequence, graph, or 3D strcuture-->
### Molecule generation (or design/edit) 
- `2023` Domain-Agnostic Molecular Generation with Self-feedback, [arXiv](https://arxiv.org/abs/2301.11259)
- `2021` MolGPT: Molecular Generation Using a Transformer-Decoder Model, [JCIM](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00600)

### Molecule property prediction
- `2023` MolBART: Generative Masked Language Models for Molecular Representations, [ICLR](https://openreview.net/forum?id=-4HJSA3Y2vg)
- `2022` Chemformer: a pre-trained transformer for computational chemistry, [IOP Science](https://iopscience.iop.org/article/10.1088/2632-2153/ac3ffb/meta)
- `2022` Molecular contrastive learning of representations via graph neural networks, [Nature](https://www.nature.com/articles/s41467-022-28494-3)

### Others



## 🦠 Genome LLM Papers 

<!-- - `2023` The status of the human gene catalogue, [Nature](https://www.nature.com/articles/s41586-023-06490-x)
- `2023` PCell-type-specific prediction of 3D chromatin organization enables high-throughput in silico genetic screening, [Nature](https://www.nature.com/articles/s41587-022-01612-8)
- `2022` Pan-cancer integrative histology-genomic analysis via multimodal deep learning, [Cancer Cell](https://www.cell.com/cancer-cell/fulltext/S1535-6108(22)00317-8) -->


## Ⓜ️ Multimodal LLM Papers

### Protein-text
- `2023` ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts, [arXiv](https://arxiv.org/abs/2301.12040)

### Molecule-text
- `2023` MolFM: A Multimodal Molecular Foundation Model, [arXiv](https://arxiv.org/abs/2307.09484)
- `2023` GIT-Mol: A Multi-modal Large Language Model for Molecular Science with Graph, Image, and Text, [arXiv](https://arxiv.org/abs/2308.06911)
- `2023` Empowering Molecule Discovery for Molecule-Caption Translation with Large Language Models: A ChatGPT Perspective, [arXiv](https://arxiv.org/abs/2306.06615)
- `2023` DrugChat: Towards Enabling ChatGPT-Like Capabilities on Drug Molecule Graphs, [arXiv](https://arxiv.org/abs/2309.03907)
- `2023` ChatGPT-powered Conversational Drug Editing Using Retrieval and Domain Feedback, [arXiv](https://arxiv.org/abs/2305.18090)
- `2022` Translation between Molecules and Natural Language, [arXiv](https://arxiv.org/abs/2204.11817)
- `2022` Multi-modal Molecule Structure-text Model for Text-based Retrieval and Editing, [arXiv](https://arxiv.org/abs/2212.10789)
- `2022` A deep-learning system bridging molecule structure and biomedical text with comprehension comparable to human professionals, [Nature](https://www.nature.com/articles/s42256-022-00447-x)
- `2021` Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries, [EMNLP](https://aclanthology.org/2021.emnlp-main.47/)

### Genome-text

### Protein-molecule

### Protein-molecule-text

### Others







## Contribution
### 👥 Contributors


<!-- ### 🎉 Contributing ( welcome ! )

- ✨ Add a new paper or update an existing Protein-related LLM paper.
- 🧐 Use the same format as existing entries to describe the work.
- 😄 A very brief explanation why you think a paper should be added or updated is recommended (Not Neccessary) via **`Adding Issues`** or **`Pull Requests`**.

**Don't worry if you put something wrong, they will be fixed for you. Just feel free to contribute and promote your awesome work here! 🤩 We'll get back to you in time ~ 😉** -->


