# Scientific Large Language Models (LLMs)

This repository collects some scientific LLM papers. Welcome to follow and star!

<!-- > 😎 Welcome to recommend missing papers through **`Adding Issues`** or **`Pull Requests`**.  -->

<!-- ## 🔔 News
- **2023-07  We create this repository to maintain a paper list on *Large Language Models* appiled in *Protein*.**

*Todo:*
1. - [ ] `Fine-grained classification of papers`
2. - [ ] `Update paper project / code` -->

## Contents
- [Scientific Large Language Models (LLMs)](#scientific-large-language-models-llms)
  - [Contents](#contents)
  - [📖 Text LLM Papers](#-text-llm-papers)
    - [Comprehensive](#comprehensive)
    - [Biology](#biology)
    - [Chemistry](#chemistry)
    - [Physics](#physics)
    - [Medicine](#medicine)
    - [Geography](#geography)
    - [Materialogy](#materialogy)
    - [Mathematics](#mathematics)
    - [Agriculture](#agriculture)
    - [Others](#others)
  - [🧬 Protein LLM Papers](#-protein-llm-papers)
    - [Protein sequence representation/generation/design](#protein-sequence-representationgenerationdesign)
    - [Protein function/property prediction](#protein-functionproperty-prediction)
    - [Protein structure prediction](#protein-structure-prediction)
    - [Others](#others-1)
  - [🧪 Molecule LLM Papers](#-molecule-llm-papers)
    - [Molecule generation/design/edit](#molecule-generationdesignedit)
    - [Molecule property prediction](#molecule-property-prediction)
    - [Others](#others-2)
  - [🦠 Genome LLM Papers](#-genome-llm-papers)
    - [General Analysis](#General-Analysis)
    - [Gene Expression and Regulatory Element Prediction]
    - [RNA Analysis and Prediction]
    - [Protein Binding Site Prediction]
    - [Sequence Variation and Evolution Analysis]
    - [Sequence Classification and Feature Selection]
    - [Other Downstream Task]
  - [Ⓜ️ Multimodal LLM Papers](#️-multimodal-llm-papers)
    - [Protein-text](#protein-text)
    - [Molecule-text](#molecule-text)
    - [Genome-text](#genome-text)
    - [Protein-molecule](#protein-molecule)
    - [Protein-molecule-text](#protein-molecule-text)
    - [Others](#others-3)
  - [Contribution](#contribution)
    - [👥 Contributors](#-contributors)



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
- `2019` SciBERT: A Pretrained Language Model for Scientific Text,[arXiv](https://arxiv.org/abs/1903.10676)
- `2022` Galactica: A Large Language Model for Science,[arXiv](https://arxiv.org/abs/2211.09085)
- `2022` Structured information extraction from complex scientific text with fine-tuned large language models.[arXiv](https://arxiv.org/abs/2212.05238)
- `2023` DARWIN Series: Domain Specific Large Language Models for Natural Science,[arXiv](https://arxiv.org/abs/2308.13565)
- `2023` MapperGPT: Large Language Models for Linking and Mapping Entities,[arXiv](https://arxiv.org/abs/2310.03666)
- `2023` SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models,[arXiv](https://arxiv.org/abs/2307.10635)
- 
### Biology
- `2019` BioBERT: a pre-trained biomedical language representation model for biomedical text mining,[arXiv](https://arxiv.org/abs/1901.08746)
- `2020` BioMegatron: Larger Biomedical Domain Language Model,[arXiv](https://arxiv.org/abs/2010.06060)
- `2023` BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining,[arXiv](https://arxiv.org/abs/2210.10341)
- `2023` BioPlanner: Automatic Evaluation of LLMs on Protocol Planning in Biology,[arXiv](https://arxiv.org/abs/2310.10632)
- `2023` An Extensive Benchmark Study on Biomedical Text Generation and Mining with ChatGPT,[PubMed](https://pubmed.ncbi.nlm.nih.gov/37682111/)
- 
### Chemistry
- `2022` Translation between Molecules and Natural Language,[arXiv](https://arxiv.org/abs/2204.11817)
- `2023` Is GPT-3 all you need for low-data discovery in chemistry?,[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/63eb5a669da0bc6b33e97a35)
- `2023` ChemCrow: Augmenting large-language models with chemistry tools,[arXiv](https://arxiv.org/abs/2304.05376)
- `2023` What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks,[arXiv](https://arxiv.org/abs/2305.18365)
- `2023` Transformers and Large Language Models for Chemistry and Drug Discovery,[arXiv](https://arxiv.org/abs/2310.06083)
- 
### Physics
- `2023` Can Language Models Understand Physical Concepts?,[EMNLP](https://arxiv.org/pdf/2305.14057.pdf)
- 
### Medicine
- `2022` A large language model for electronic health records,[nature npj digital medicine](https://www.nature.com/articles/s41746-022-00742-2)
- `2023` Large language models encode clinical knowledge,[nature](https://www.nature.com/articles/s41586-023-06291-2)
- `2023` ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge,[arXiv](https://arxiv.org/abs/2303.14070)
- `2023` DoctorGLM: Fine-tuning your Chinese Doctor is not a Herculean Task,[arXiv](https://arxiv.org/abs/2304.01097)
- `2023` Towards Expert-Level Medical Question Answering with Large Language Models,[arXiv](https://arxiv.org/abs/2305.09617)
- `2023` ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge,[arXiv](https://arxiv.org/abs/2303.14070)
- 
### Geography
- `2023` K2: A Foundation Language Model for Geoscience Knowledge Understanding and Utilization.[arXiv](https://arxiv.org/abs/2306.05064)
- `2023` GPT4GEO:HowaLanguageModel Sees the World’s Geography,[arXiv](https://arxiv.org/abs/2306.00020)
- 
### Materialogy
- `2023` MatChat: A Large Language Model and Application Service Platform for Materials Science,[arXiv](https://arxiv.org/abs/2310.07197)
- `2023` MechGPT, a language-based strategy for mechanics and materials modeling that connects knowledge across scales, disciplines and modalities,[arXiv](https://arxiv.org/abs/2310.10445)
- 
### Mathematics
- `2023` Llemma: An Open Language Model For Mathematics,[arXiv](https://arxiv.org/abs/2310.10631)
- `2023` Improving Large Language Model Fine-tuning for Solving Math Problems,[arXiv](https://arxiv.org/abs/2310.10047)
- `2023` MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models,[arXiv](https://arxiv.org/abs/2309.12284)
- 
### Agriculture

### Others
- `2023` PeptideBERT: A Language Model based on Transformers for Peptide Property Prediction,[arXiv](https://arxiv.org/abs/2309.03099)
- 

## 🧬 Protein LLM Papers
<!-- 参考 https://github.com/opendilab/awesome-AI-based-protein-design
https://github.com/yangkky/Machine-learning-for-proteins
https://github.com/LirongWu/awesome-protein-representation-learning -->
###  Protein sequence representation/generation/design
- `2020` ProGen: Language Modeling for Protein Generation，[arXiv](https://doi.org/10.48550/arXiv.2004.03497)
- `2021` Modeling Protein Using Large-scale Pretrain Language Model, [arXiv](https://doi.org/10.48550/arXiv.2108.07435)(Protein sequence representation)
- `2021` Pre-training Co-evolutionary Protein Representation via A Pairwise Masked Language Model, [arXiv](https://doi.org/10.48550/arXiv.2110.15527)(Protein sequence representation)
- `2022` Large language models generate functional protein sequences across diverse families, [Nature Biotechnology]( https://doi.org/10.1038/s41587-022-01618-2)
- `2022` Controllable protein design with language models, [Nature Machine Intelligence]( https://doi.org/10.1038/s42256-022-00499-z)
- `2022` A deep unsupervised language model for protein design, [bioRxiv](https://doi.org/10.1101/2022.03.09.483666)
- `2022` ProtGPT2 is a deep unsupervised language model for protein design,[Nature Communications](https://doi.org/10.1038/s41467-022-32007-7)
- `2023` Current progress, challenges, and future perspectives of language models for protein representation and protein design, [ScienceDirect](https://doi.org/10.1016/j.xinn.2023.100446)
###  Protein function/property prediction
- `2019` A High Efficient Biological Language Model for Predicting Protein–Protein Interactions, [Cells](https://doi.org/10.3390/cells8020122)
- `2021` Language models enable zero-shot prediction of the effects of mutations on protein function, [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2021/file/f51338d736f95dd42427296047067694-Paper.pdf)
- `2022` Exploring evolution-aware & -free protein language models as protein function predictors, [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/file/fe066022bab2a6c6a3c57032a1623c70-Paper-Conference.pdf)
- `2022` Exploring evolution-aware & -free protein language models as protein function predictors, [arXiv](https://doi.org/10.48550/arXiv.2206.06583)
- `2023` A Multimodal Protein Representation Framework for Quantifying Transferability Across Biochemical Downstream Tasks, [Advance Science](https://onlinelibrary.wiley.com/doi/10.1002/advs.202301223)
- `2023` Enhancing Protein Language Models with Structure-based Encoder and Pre-training, [arXiv](https://arxiv.org/abs/2303.06275)
- `2023` Large language models improve annotation of viral proteins, [Nature Portfolio](https://doi.org/10.21203/rs.3.rs-2852098/v1)
- `2023` Linguistically inspired roadmap for building biologically reliable protein language models, [arXiv](https://doi.org/10.48550/arXiv.2207.00982)
- `2023` Advancing variant effect prediction using protein language models,[Nature Genetics](https://doi.org/10.1038/s41588-023-01470-3)
- `2023` NetGO 3.0: A Protein Language Model Improves Large-scale Functional Annotations, [GPB](https://doi.org/10.1016/j.gpb.2023.04.001)
- `2023` Genome-wide prediction of disease variant effects with a deep protein language model, [Nature Genetics](https://doi.org/10.1038/s41588-023-01465-0)
- `2023` Protein Fitness Prediction Is Impacted by the Interplay of Language Models, Ensemble Learning, and Sampling Methods, [MDPI](https://doi.org/10.3390/pharmaceutics15051337)
###  Protein structure prediction
- `2020` Transformer protein language models are unsupervised structure learners, [bioRxiv](https://doi.org/10.1101/2020.12.15.422761)
- `2021` Highly accurate protein structure prediction with AlphaFold, [nature](https://doi.org/10.1038/s41586-021-03819-2)
- `2022` Language models of protein sequences at the scale of evolution enable accurate structure prediction, [bioRxiv](https://doi.org/10.1101/2022.07.20.500902)
- `2022` Accurate prediction of protein structures and interactions using a 3-track neural network, [Science](https://doi.org/10.1126/science.abj8754)
- `2022` Single-sequence protein structure prediction using a language model and deep learning, [Nature Biotechnology](https://doi.org/10.1038/s41587-022-01432-w)
- `2022` Single-sequence protein structure prediction using supervised transformer protein language models, [Nature Computational Science](https://doi.org/10.1038/s43588-022-00373-3)
- `2022` Improved the Protein Complex Prediction with Protein Language Models, [bioRxiv](https://doi.org/10.1101/2022.09.15.508065)
- `2023` Enhancing the Protein Tertiary Structure Prediction by Multiple Sequence Alignment Generation, [arXiv](https://arxiv.org/abs/2306.01824)
- `2023` Evolutionary-scale prediction of atomic-level protein structure with a language model, [Science](https://doi.org/10.1126/science.ade2574)
- `2023` Bilingual Language Model for Protein Sequence and Structure, [arXiv](https://doi.org/10.1101/2023.07.23.550085)
- `2023` Integration of pre-trained protein language models into geometric deep learning networks, [Communications Biology](https://doi.org/10.1038/s42003-023-05133-1)
- `2023` A method for multiple-sequence-alignment-free protein structure prediction using a protein language model, [Nature Machine Intelligence](https://doi.org/10.1038/s42256-023-00721-6)

### Others
- `2022` Evolutionary velocity with protein language models predicts evolutionary dynamics of diverse proteins, [Cell](https://doi.org/10.1016/j.cels.2022.01.003)(evolutionary prediction)
- `2022` Improving protein succinylation sites prediction using embeddings from protein language model, [Scientific Reports](https://doi.org/10.1038/s41598-022-21366-2)(succinylation sites prediction)
- `2023` OceanGPT: A Large Language Model for Ocean Science Tasks,[arXiv](https://arxiv.org/abs/2310.02031)
- 

## 🧪 Molecule LLM Papers
<!-- 参考 https://github.com/OmicsML/awesome-molecule-protein-pretrain-papers -->
<!-- 只考虑分子单模态的，可以标注一下是基于sequence, graph, or 3D strcuture-->
### Molecule generation/design/edit
- `2023` Domain-Agnostic Molecular Generation with Self-feedback, [arXiv](https://arxiv.org/abs/2301.11259)
- `2021` MolGPT: Molecular Generation Using a Transformer-Decoder Model, [JCIM](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00600)

### Molecule property prediction
- `2023` MolBART: Generative Masked Language Models for Molecular Representations, [ICLR](https://openreview.net/forum?id=-4HJSA3Y2vg)
- `2022` Chemformer: a pre-trained transformer for computational chemistry, [IOP Science](https://iopscience.iop.org/article/10.1088/2632-2153/ac3ffb/meta)
- `2022` Molecular contrastive learning of representations via graph neural networks, [Nature](https://www.nature.com/articles/s41467-022-28494-3)

### Others


## 🦠 Genome LLM Papers 
### General Analysis
- `2023` DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks [BioRxiv](https://www.biorxiv.org/content/10.1101/2023.07.11.548628v2)
- `2023` HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution [arXiv](https://arxiv.org/abs/2306.15794)
- `2023` GeneGPT: Augmenting Large Language Models with Domain Tools for Improved Access to Biomedical Information [arXiv](https://arxiv.org/abs/2304.09667v3)
- `2023` GENA-LM: A Family of Open-Source Foundational Models for Long DNA Sequences [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v1.abstract)
- `2022` MoDNA: motif-oriented pre-training for DNA language model [BCB 2022](https://dl.acm.org/doi/10.1145/3535508.3545512)
- `2022` Integrating convolution and self-attention improves language model of human genome for interpreting non-coding regions at base-resolution [Nucleic Acids Research](https://academic.oup.com/nar/article/50/14/e81/6583232)
- `2022` Fine-Tuning Transformers For Genomic Tasks [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.02.07.479412v1.abstract)
- `2022` iDNA-ABF: multi-scale deep biological language learning model for the interpretable prediction of DNA methylations [Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02780-1)
- `2022` BioSeq-BLM: a platform for analyzing DNA, RNA and protein sequences based on biological language models [Nucleic Acids Research](https://academic.oup.com/nar/article/49/22/e129/6377401)
- `2021` Effective gene expression prediction from sequence by integrating long-range interactions [Nature Methods](https://www.nature.com/articles/s41592-021-01252-x)
- `2021` DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome [Bioinformatics](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680)

### Gene Expression and Regulatory Element Prediction
- `2023` Species-aware DNA language models capture regulatory elements and their evolution [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.01.26.525670v2.abstract)
- `2023` PLPMpro: Enhancing promoter sequence prediction with prompt-learning based pre-trained language model [Computers in Biology and Medicine](https://www.sciencedirect.com/science/article/abs/pii/S0010482523007254)
- `2023` A single-cell gene expression language model [NeurIPS 2022 Workshop](https://arxiv.org/abs/2210.14330)
- `2023` DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome [arXiv](https://arxiv.org/abs/2306.15006)
- `2022` iEnhancer-BERT: A Novel Transfer Learning Architecture Based on DNA-Language Model for Identifying Enhancers and Their Strength [ICIC 2022](https://link.springer.com/chapter/10.1007/978-3-031-13829-4_13)
- `2022` iEnhancer-ELM: improve enhancer identification by extracting position-related multiscale contextual information based on enhancer language models [arXiv](https://arxiv.org/abs/2212.01495)
- `2021` A transformer architecture based on BERT and 2D convolutional neural network to identify DNA enhancers from sequence information [Briefings in Bioinformatics](https://academic.oup.com/bib/article/22/5/bbab005/6128847)

### RNA Analysis and Prediction
- `2023` Multiple sequence-alignment-based RNA language model and its application to structural inference [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.03.15.532863v1.abstract)
- `2023` Prediction of Multiple Types of RNA Modifications via Biological Language Model [IEEE/ACM Transactions on Computational Biology and Bioinformatics](https://ieeexplore.ieee.org/abstract/document/10146457)
- `2023` miProBERT: identification of microRNA promoters based on the pre-trained model BERT [Briefings in Bioinformatics](https://academic.oup.com/bib/article-abstract/24/3/bbad093/7079709)
- `2022` scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data [Nature Machine Intelligence]https://www.nature.com/articles/s42256-022-00534-z
- `2022` Language-Informed Basecalling Architecture for Nanopore Direct RNA Sequencing [PMLR 2022](https://proceedings.mlr.press/v200/sneddon22a.html)
- `2022` ELMo4m6A: A Contextual Language Embedding-Based Predictor for Detecting RNA N6-Methyladenosine Sites [IEEE/ACM Transactions on Computational Biology and Bioinformatics](https://ieeexplore.ieee.org/abstract/document/9771386)

### Protein Binding Site Prediction
- `2023` Improving language model of human genome for DNA–protein binding prediction based on task-specific pre-trainin [Interdisciplinary Sciences: Computational Life Sciences volume](https://link.springer.com/article/10.1007/s12539-022-00537-9)
- `2022` Comprehensive Evaluation of BERT Model for DNA-Language for Prediction of DNA Sequence Binding Specificities in Fine-Tuning Phase [ICIC 2022](https://link.springer.com/chapter/10.1007/978-3-031-13829-4_8)
- `2022` PepNN: a deep attention model for the identification of peptide binding sites [Communications Biology](https://www.nature.com/articles/s42003-022-03445-2)

### Sequence Variation and Evolution Analysis
- `2023` GPN-MSA: an alignment-based DNA language model for genome-wide variant effect prediction [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.10.10.561776v1.abstract)
- `2022` GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.10.10.511571v2)
- `2022` DNA language models are powerful predictors of genome-wide variant effects [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.08.22.504706v3.abstract)

### Sequence Classification and Feature Selection
- `2023` Hamming Encoder: Mining Discriminative k-mers for Discrete Sequence Classification [arXiv](https://arxiv.org/abs/2310.10321)

### Other Downstream Tasks
- `2023` MuLan-Methyl - Multiple Transformer-based Language Models for Accurate DNA Methylation Prediction [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.01.04.522704v2.abstract)



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


