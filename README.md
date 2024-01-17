# Scientific Large Language Models

This repository collects papers of scientific large language models (Sci-LLMs) in the biological and chemical domains. 
> 😎 Welcome to recommend missing papers through **`Adding Issues`** or **`Pull Requests`**. 

## 🔔 News
- 💥 [2024/01] Our survey is released! See [Scientific Large Language Models: A Survey on Biochemical Domain](https://arxiv.org/) for the paper!

## 🌟 Contents
- [Scientific Large Language Models (Sci-LLMs)](#)
  - [📖 Textual Scientific Large Language Models (Text-Sci-LLMs)](#-text-llm-papers)
    - [Biology](#biology)
    - [Chemistry](#chemistry)
    - [Comprehensive](#comprehensive)
    - [Datasets and Benchmarks](#datasets-and-benchmarks)
  - [🧪 Molecular Large Language Models (Mol-LLMs)](#-molecule-llm-papers)
    - [Property/Reaction/Interaction Prediction](#molecule-property-predictionreaction-predictioninteraction-prediction)
    - [Generation/Design/Edit](#molecule-generationdesignedit)
    - [Datasets and Benchmarks](#datasets-and-benchmarks-2)
  - [🧬 Protein Large Language Models (Prot-LLMs)](#-protein-llm-papers)
    - [Function/Property Prediction](#protein-functionproperty-prediction)
    - [Sequence Generation/Design](#protein-sequence-generationdesign)
    - [Datasets and Benchmarks](#datasets-and-benchmarks-1)
  - [🦠 Genomic Large Language Models (Gene-LLMs)](#-genome-llm-papers)
    - [Function Prediction](#)
    - [Structure Prediction](#)
    - [Sequence Generation](#)
    - [Sequence Variation and Evolution Analysis](#)
    - [Datasets and Benchmarks](#)
  - [Ⓜ️ Multimodal Scientific Large Language Models (MM-Sci-LLMs)]
  (#️-multimodal-llm-papers)
    - [Molecule-Text](#molecule-text)
    - [Protein-Text](#protein-text)
    - [Protein-Molecule](#protein-molecule)
    - [Comprehensive](#protein-molecule-text)
    - [Datasets and Benchmarks](#datasets-and-benchmarks-4)
  - [👥 Contributions](#contribution)
    - [Citation](#citation)
    - [Contributors](#contributors)



## 📖 Text-Sci-LLMs
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
- `2020` Don't Stop Pretraining: Adapt Language Models to Domains and Tasks,[arXiv](https://arxiv.org/abs/2004.10964)
- `2022` Galactica: A Large Language Model for Science,[arXiv](https://arxiv.org/abs/2211.09085)
- `2022` Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering,[arXiv](https://arxiv.org/abs/2209.09513)_
<!-- - `2022` Structured information extraction from complex scientific text with fine-tuned large language models.[arXiv](https://arxiv.org/abs/2212.05238) -->
- `2023` DARWIN Series: Domain Specific Large Language Models for Natural Science,[arXiv](https://arxiv.org/abs/2308.13565)
- `2023` Sci-CoT: Leveraging Large Language Models for Enhanced Knowledge Distillation in Small Models for Scientific QA,[arXiv](https://arxiv.org/abs/2308.04679)
<!-- - `2023` MapperGPT: Large Language Models for Linking and Mapping Entities,[arXiv](https://arxiv.org/abs/2310.03666) -->
- 
- 
### Biology
- `2019` BioBERT: a pre-trained biomedical language representation model for biomedical text mining,[arXiv](https://arxiv.org/abs/1901.08746)
- `2020` BioMegatron: Larger Biomedical Domain Language Model,[arXiv](https://arxiv.org/abs/2010.06060)
- `2020` Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing,[arXiv](https://arxiv.org/pdf/2007.15779.pdf)
- `2021` BioM-Transformers: Building Large Biomedical Language Models with BERT, ALBERT and ELECTRA,[ACL Anthology](https://aclanthology.org/2021.bionlp-1.24/)
- `2022` LinkBERT: Pretraining Language Models with Document Links,[arXiv](https://arxiv.org/abs/2203.15827)
- `2023` BioinspiredLLM: Conversational Large Language Model for the Mechanics of Biological and Bio-inspired Materials,[arXiv](https://arxiv.org/abs/2309.08788)
- `2023` BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining,[arXiv](https://arxiv.org/abs/2210.10341)
- `2023` BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine,[arXiv](https://arxiv.org/abs/2308.09442)
- `2023` BioPlanner: Automatic Evaluation of LLMs on Protocol Planning in Biology,[arXiv](https://arxiv.org/abs/2310.10632)
- `2023` An Extensive Benchmark Study on Biomedical Text Generation and Mining with ChatGPT,[PubMed](https://pubmed.ncbi.nlm.nih.gov/37682111/)
<!-- - `2023` BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine,[arXiv](https://arxiv.org/abs/2308.09442v2) -->
- 
### Chemistry
<!-- - `2022` Translation between Molecules and Natural Language,[arXiv](https://arxiv.org/abs/2204.11817) -->
- `2021` Automated Chemical Reaction Extraction from Scientific Literature.[ACS](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00284)
- `2023` Is GPT-3 all you need for low-data discovery in chemistry?,[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/63eb5a669da0bc6b33e97a35)
- `2023` ChemCrow: Augmenting large-language models with chemistry tools,[arXiv](https://arxiv.org/abs/2304.05376)
- `2023` What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks,[arXiv](https://arxiv.org/abs/2305.18365)
- `2023` Transformers and Large Language Models for Chemistry and Drug Discovery,[arXiv](https://arxiv.org/abs/2310.06083)
- `2023` What a Scientific Language Model Knows and Doesn't Know about Chemistry,[OpenReview](https://openreview.net/forum?id=hSmn7BQZ2v)
- 
### Physics
- `2023` Can Language Models Understand Physical Concepts?,[EMNLP](https://arxiv.org/pdf/2305.14057.pdf)
- `2023` MeLM, a generative pretrained language modeling framework that solves forward and inverse mechanics problems,[arXiv](https://www.sciencedirect.com/science/article/abs/pii/S0022509623002582)
- 
### Medicine
- `2022` A large language model for electronic health records,[nature npj digital medicine](https://www.nature.com/articles/s41746-022-00742-2)
- `2022` CancerBERT: a cancer domain-specific language model for extracting breast cancer phenotypes from electronic health records,[JAMIA](https://academic.oup.com/jamia/article/29/7/1208/6554005)
- `2023` Large language models encode clinical knowledge,[nature](https://www.nature.com/articles/s41586-023-06291-2)
- `2023` Theory-Driven Analysis of Natural Language Processing Measures of Thought Disorder Using Generative Language Modeling,[BP:CNNI](https://www.biologicalpsychiatrycnni.org/article/S2451-9022(23)00125-8/fulltext)
- `2023` ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge,[arXiv](https://arxiv.org/abs/2303.14070)
- `2023` Large language models in medicine,[Nat. Med](https://www.nature.com/articles/s41591-023-02448-8)
- `2023` DoctorGLM: Fine-tuning your Chinese Doctor is not a Herculean Task,[arXiv](https://arxiv.org/abs/2304.01097)
- `2023` Towards Expert-Level Medical Question Answering with Large Language Models,[arXiv](https://arxiv.org/abs/2305.09617)- `2023` Xiezhi: An Ever-Updating Benchmark for Holistic Domain Knowledge Evaluation,[arXiv](https://arxiv.org/abs/2306.05783)
- `2023` AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models,[arXiv](https://arxiv.org/abs/2304.06364)
- `2023` Bio-SIEVE: Exploring Instruction Tuning Large Language Models for Systematic Review Automation,[arXiv](https://arxiv.org/abs/2308.06610)
- `2023` ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge,[arXiv](https://arxiv.org/abs/2303.14070)
- 
### Geography
- `2023` K2: A Foundation Language Model for Geoscience Knowledge Understanding and Utilization.[arXiv](https://arxiv.org/abs/2306.05064)
- `2023` GPT4GEO:HowaLanguageModel Sees the World’s Geography,[arXiv](https://arxiv.org/abs/2306.00020)
- 
### Materialogy
- `2022` MatSciBERT: A materials domain language model for text mining and information extraction,[NPJ Comput](https://www.nature.com/articles/s41524-022-00784-w)
- `2023` MatChat: A Large Language Model and Application Service Platform for Materials Science,[arXiv](https://arxiv.org/abs/2310.07197)
- `2023` OpticalBERT and OpticalTable-SQA: Text- and Table-Based Language Models for the Optical-Materials Domain,[JCIM](https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c01259)
- `2023` MechGPT, a language-based strategy for mechanics and materials modeling that connects knowledge across scales, disciplines and modalities,[arXiv](https://arxiv.org/abs/2310.10445)
- 
### Mathematics
- `2022` Continual Pre-training of Language Models for Math Problem Understanding with Syntax-Aware Memory Network,[ACL](https://aclanthology.org/2022.acl-long.408/)
- `2023` Llemma: An Open Language Model For Mathematics,[arXiv](https://arxiv.org/abs/2310.10631)
- `2023` Improving Large Language Model Fine-tuning for Solving Math Problems,[arXiv](https://arxiv.org/abs/2310.10047)
- `2023` MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models,[arXiv](https://arxiv.org/abs/2309.12284)
- `2023` Solving Math Word Problems by Combining Language Models With Symbolic Solvers,[arXiv](https://arxiv.org/abs/2304.09102)
- `2023` Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct,[arXiv](https://arxiv.org/abs/2308.09583)

### Agriculture
- `2020` Agribot: A Natural Language Generative Neural Networks Engine for Agricultural Applications,[IC3A](https://ieeexplore.ieee.org/abstract/document/9077116)
- `2023` Embedding-based Retrieval with LLM for Effective Agriculture Information Extracting from Unstructured Data,[arXiv](https://arxiv.org/abs/2308.03107)
- 

### Datasets and Benchmarks
- `2017` Crowdsourcing Multiple Choice Science Questions,[Senmantic Scholar](https://www.semanticscholar.org/paper/Crowdsourcing-Multiple-Choice-Science-Questions-Welbl-Liu/932a5de79d8a8ebb75ea0c43493450fd9922e738)
- `2019` PubMedQA: A Dataset for Biomedical Research Question Answering,[arXiv](https://arxiv.org/abs/1909.06146)
- `2020` Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing,[arXiv](https://arxiv.org/pdf/2007.15779.pdf)
- `2023` Bioinfo-Bench: A Simple Benchmark Framework for LLM Bioinformatics Skills Evaluation,[BioRxiv](https://www.biorxiv.org/content/10.1101/2023.10.18.563023v1.abstract)
- `2023` BioLLMBench: A Comprehensive Benchmarking of Large Language Models in Bioinformatics,[BioRxiv](https://www.biorxiv.org/content/10.1101/2023.12.19.572483v1.abstract)
- `2023` SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models,[arXiv](https://arxiv.org/abs/2307.10635)

### Others
- `2020` NukeBERT: A Pre-trained language model for Low Resource Nuclear Domain,[arXiv](https://arxiv.org/abs/2003.13821)
<!-- - `2023` PeptideBERT: A Language Model based on Transformers for Peptide Property Prediction,[arXiv](https://arxiv.org/abs/2309.03099) -->
- `2023` OceanGPT: A Large Language Model for Ocean Science Tasks,[arXiv](https://arxiv.org/abs/2310.02031)
<!-- - `2022` Pretrained domain-specific language model for natural language processing tasks in the AEC domain,[Comput Ind](https://www.sciencedirect.com/science/article/pii/S0166361522001300) -->
- 

## 🧬 Prot-LLMs
<!-- 参考 https://github.com/opendilab/awesome-AI-based-protein-design
https://github.com/yangkky/Machine-learning-for-proteins
https://github.com/LirongWu/awesome-protein-representation-learning -->
###  Protein sequence generation/design
- `2020` ProGen: Language Modeling for Protein Generation, [arXiv](https://doi.org/10.48550/arXiv.2004.03497)
- `2020` Bertology meets biology: Interpreting attention in protein language models, [arXiv](https://arxiv.org/abs/2006.15222)
- `2020` Transforming the Language of Life: Transformer Neural Networks for Protein Prediction Tasks, [BCB](https://doi.org/10.1145/3388440.3412467)
- `2021` Generating novel protein sequences using Gibbs sampling of masked language models, [bioRxiv](https://doi.org/10.1101/2021.01.26.428322)
- `2021` BioSeq-BLM: a platform for analyzing DNA, RNA and protein sequences based on biological language models, [Nucleic](https://academic.oup.com/nar/article-abstract/49/22/e129/6377401)
- `2022` Large language models generate functional protein sequences across diverse families, [Nature Biotechnology]( https://doi.org/10.1038/s41587-022-01618-2)
- `2022` Controllable protein design with language models, [Nature Machine Intelligence]( https://doi.org/10.1038/s42256-022-00499-z)
- `2022` A deep unsupervised language model for protein design, [bioRxiv](https://doi.org/10.1101/2022.03.09.483666)
- `2022` ProtGPT2 is a deep unsupervised language model for protein design,[Nature Communications](https://doi.org/10.1038/s41467-022-32007-7)
- `2022` RITA: a Study on Scaling Up Generative Protein Sequence Models,[arXiv](https://doi.org/10.48550/arXiv.2205.05789)
- `2022` A high-level programming language for generative protein design, [bioRxiv](https://doi.org/10.1101/2022.12.21.521526)
- `2023` Current progress, challenges, and future perspectives of language models for protein representation and protein design, [Science](https://doi.org/10.1016/j.xinn.2023.100446)
- `2023` Deciphering the protein landscape with ProtFlash, a lightweight language model, [Science](https://doi.org/10.1016/j.xcrp.2023.101600)
- `2023` Structure-informed Language Models Are Protein Designers, [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.02.03.526917v2)

###  Protein function/property prediction
- `2019` Modeling the language of life – Deep Learning Protein Sequences, [bioRxiv](https://doi.org/10.1101/614313)
- `2019` A High Efficient Biological Language Model for Predicting Protein–Protein Interactions, [Cells](https://doi.org/10.3390/cells8020122)
- `2021` Language models enable zero-shot prediction of the effects of mutations on protein function, [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2021/file/f51338d736f95dd42427296047067694-Paper.pdf)
- `2021` Modeling Protein Using Large-scale Pretrain Language Model, [arXiv](https://doi.org/10.48550/arXiv.2108.07435)
- `2021` Pre-training Co-evolutionary Protein Representation via A Pairwise Masked Language Model, [arXiv](https://doi.org/10.48550/arXiv.2110.15527)
- `2022` Exploring evolution-aware & -free protein language models as protein function predictors, [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/file/fe066022bab2a6c6a3c57032a1623c70-Paper-Conference.pdf)
- `2022` Exploring evolution-aware & -free protein language models as protein function predictors, [arXiv](https://doi.org/10.48550/arXiv.2206.06583)
- `2022` Learning functional properties of proteins with language models, [Nature Machine Intelligence](https://doi.org/10.1038/s42256-022-00457-9)
- `2022` Fast and accurate protein function prediction from sequence through pretrained language model and homology-based label diffusion, [bioRxiv](https://doi.org/10.1101/2022.12.05.519119)
- `2022` SESNet: sequence-structure feature-integrated deep learning method for data-efficient protein engineering, [arXiv](https://arxiv.org/abs/2301.00004)
- `2023` A Multimodal Protein Representation Framework for Quantifying Transferability Across Biochemical Downstream Tasks, [Advance Science](https://onlinelibrary.wiley.com/doi/10.1002/advs.202301223)
- `2023` Enhancing Protein Language Models with Structure-based Encoder and Pre-training, [arXiv](https://arxiv.org/abs/2303.06275)
- `2023` Large language models improve annotation of viral proteins, [Nature Portfolio](https://doi.org/10.21203/rs.3.rs-2852098/v1)
- `2023` Linguistically inspired roadmap for building biologically reliable protein language models, [arXiv](https://doi.org/10.48550/arXiv.2207.00982)
- `2023` Advancing variant effect prediction using protein language models,[Nature Genetics](https://doi.org/10.1038/s41588-023-01470-3)
- `2023` NetGO 3.0: A Protein Language Model Improves Large-scale Functional Annotations, [GPB](https://doi.org/10.1016/j.gpb.2023.04.001)
- `2023` Genome-wide prediction of disease variant effects with a deep protein language model, [Nature Genetics](https://doi.org/10.1038/s41588-023-01465-0)
- `2023` Protein Fitness Prediction Is Impacted by the Interplay of Language Models, Ensemble Learning, and Sampling Methods, [MDPI](https://doi.org/10.3390/pharmaceutics15051337)
- `2023` Convolutions are competitive with transformers for protein sequence pretraining, [bioRxiv](https://doi.org/10.1101/2022.05.19.492714)
- `2023` Protein Representation Learning via Knowledge Enhanced Primary Structure Modeling, [arXiv](https://arxiv.org/abs/2301.13154) 
- `2023` Learning Hierarchical Protein Representations via Complete 3D Graph Networks, [arXiv](https://arxiv.org/abs/2207.12600)[protein-graph]
- `2022` OntoProtein: Protein Pretraining With Gene Ontology Embedding, [arXiv](https://arxiv.org/abs/2201.11147)[protein-KG]

###  Protein structure prediction
- `2020` Transformer protein language models are unsupervised structure learners, [bioRxiv](https://doi.org/10.1101/2020.12.15.422761)
- `2021` Highly accurate protein structure prediction with AlphaFold, [nature](https://doi.org/10.1038/s41586-021-03819-2)
- `2022` Language models of protein sequences at the scale of evolution enable accurate structure prediction, [bioRxiv](https://doi.org/10.1101/2022.07.20.500902)
- `2022` Accurate prediction of protein structures and interactions using a 3-track neural network, [Science](https://doi.org/10.1126/science.abj8754)
- `2022` Single-sequence protein structure prediction using a language model and deep learning, [Nature Biotechnology](https://doi.org/10.1038/s41587-022-01432-w)
- `2022` Single-sequence protein structure prediction using supervised transformer protein language models, [Nature Computational Science](https://doi.org/10.1038/s43588-022-00373-3)
- `2022` Improved the Protein Complex Prediction with Protein Language Models, [bioRxiv](https://doi.org/10.1101/2022.09.15.508065)
- `2022` Evolutionary-scale prediction of atomic level protein structure with a language model, [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)
- `2023` Enhancing the Protein Tertiary Structure Prediction by Multiple Sequence Alignment Generation, [arXiv](https://arxiv.org/abs/2306.01824)
- `2023` Evolutionary-scale prediction of atomic-level protein structure with a language model, [Science](https://doi.org/10.1126/science.ade2574)
- `2023` Bilingual Language Model for Protein Sequence and Structure, [arXiv](https://doi.org/10.1101/2023.07.23.550085)
- `2023` Integration of pre-trained protein language models into geometric deep learning networks, [Communications Biology](https://doi.org/10.1038/s42003-023-05133-1)
- `2023` A method for multiple-sequence-alignment-free protein structure prediction using a protein language model, [Nature Machine Intelligence](https://doi.org/10.1038/s42256-023-00721-6)

### Datasets and Benchmarks
- `2004` The Gene Ontology (GO) database and informatics resource, [NAR](https://doi.org/10.1093/nar/gkh036)
- `2012` ECOH: An Enzyme Commission number predictor using mutual information and a support vector machine, [Bioinformatics](https://doi.org/10.1093/bioinformatics/bts700)
- `2022` Critical Assessment of Protein Structure Prediction dataset, [CASP](https://predictioncenter.org/casp15/index.cgi)

### Others
#### Protein evolutionary prediction
- `2022` Evolutionary velocity with protein language models predicts evolutionary dynamics of diverse proteins, [Cell](https://doi.org/10.1016/j.cels.2022.01.003)
#### Protein location/sites prediction
- `2021` Light Attention Predicts Protein Location from the Language of Life, [bioRxiv](https://doi.org/10.1101/2021.04.25.441334)
- `2022` Improving protein succinylation sites prediction using embeddings from protein language model, [Scientific Reports](https://doi.org/10.1038/s41598-022-21366-2)
- `2023` Language models can identify enzymatic active sites in protein sequences, [ChemRxiv](https://doi.org/10.26434/chemrxiv-2021-m20gg-v3)
<!-- #### Predicting interactions with other molecules
- `2021` Towards a systematic characterization of protein complex function: a natural language processing and machine-learning framework, [bioRxiv](https://doi.org/10.1101/2021.02.24.432789)
- `2022` Predicting the specific substrate for transmembrane transport proteins using BERT language model, [bioRxiv](https://doi.org/10.1101/2022.07.23.501263) -->


## 🧪 Mol-LLMs
<!-- 参考 https://github.com/OmicsML/awesome-molecule-protein-pretrain-papers -->
<!-- 只考虑分子单模态的，可以标注一下是基于sequence, graph, or 3D structure-->

### Molecule property prediction

- `2019` SMILES-BERT: Large Scale Unsupervised Pre-Training for Molecular Property Prediction, [ACM](https://dl.acm.org/doi/10.1145/3307339.3342186)(sequence)
- `2019` SMILES Transformer: Pre-trained Molecular Fingerprint for Low Data Drug Discovery, [arXiv](https://arxiv.org/abs/1911.04738v1)(sequence)
- `2020` ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction, [arXiv](http://arxiv.org/abs/2010.09885)(sequence)
- `2020` Self-Supervised Graph Transformer on Large-Scale Molecular Data, [arXiv](http://arxiv.org/abs/2007.02835)(graph)
- `2021` MG-BERT: leveraging unsupervised atomic representation learning for molecular property prediction, [ResearchGate](https://www.researchgate.net/publication/351363304_MG-BERT_leveraging_unsupervised_atomic_representation_learning_for_molecular_property_prediction)(graph)
- `2021` Mol-BERT: An Effective Molecular Representation with BERT for Molecular Property Prediction, [Wireless Communications and Mobile Computing](https://www.hindawi.com/journals/wcmc/2021/7181815/)(sequence)
- `2021` Algebraic graph-assisted bidirectional transformers for molecular property prediction, [Nature](https://www.nature.com/articles/s41467-021-23720-w)(3D)
<!-- - `2021` Dual-view Molecule Pre-training, [arXiv](https://arxiv.org/abs/2106.10234v2)(sequence+graph) -->
<!-- - `2021` Molecular Contrastive Learning with Chemical Element Knowledge Graph, [arXiv](http://arxiv.org/abs/2112.00544)(graph) -->
- `2022` Chemformer: a pre-trained transformer for computational chemistry, [IOP Science](https://iopscience.iop.org/article/10.1088/2632-2153/ac3ffb/meta)(sequence)
- `2022` ChemBERTa-2: Towards Chemical Foundation Models, [arXiv](http://arxiv.org/abs/2209.01712)(sequence)
- `2022` Large-Scale Chemical Language Representations Capture Molecular Structure and Properties, [arXiv](http://arxiv.org/abs/2106.09553)(sequence)
- `2022` Pushing the Boundaries of Molecular Property Prediction for Drug Discovery with Multitask Learning BERT Enhanced by SMILES Enumeration, [Research](https://spj.science.org/doi/10.34133/research.0004)
- `2022` Large-Scale Distributed Training of Transformers for Chemical Fingerprinting, [ACS Publications](https://doi.org/10.1021/acs.jcim.2c00715)(sequence)
- `2022` MOLBART: GENERATIVE MASKED LANGUAGE MODELS FOR MOLECULAR REPRESENTATIONS, [OpenReview](https://openreview.net/forum?id=-4HJSA3Y2vg)(sequence)
- `2023` Molformer: Motif-based Transformer on 3D Heterogeneous Molecular Graphs, [arXiv](http://arxiv.org/abs/2110.01191)(3D)
<!-- - `2023` ONE TRANSFORMER CAN UNDERSTAND BOTH 2D & 3D MOLECULAR DATA, [arXiv](http://arxiv.org/abs/2210.01765)(graph/3D) -->
- `2023` SELFORMER: MOLECULAR REPRESENTATION LEARNING VIA SELFIES LANGUAGE MODELS, [arXiv](http://arxiv.org/abs/2304.04662)(sequence)
- `2023` Graph-based Molecular Representation Learning, [arXiv](http://arxiv.org/abs/2207.04869)
- `2023` MolRoPE-BERT: An enhanced molecular representation with Rotary Position Embedding for molecular property prediction, [Journal of Molecular Graphics and Modelling](https://linkinghub.elsevier.com/retrieve/pii/S1093326322002236)(sequence)
- `2023` Molecular Descriptors Property Prediction Using Transformer-Based Approach, [IJMS](https://www.mdpi.com/1422-0067/24/15/11948)(sequence)
- `2023` UNI-MOL: A UNIVERSAL 3D MOLECULAR REPRESENTATION LEARNING FRAMEWORK, [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/628e5b4d5d948517f5ce6d72)
<!-- - `2023` GPT-MolBERTa: GPT Molecular Features Language Model for molecular property prediction, [arXiv](http://arxiv.org/abs/2310.03030)(sequence) -->

### Interaction prediction

#### DDI

<!-- - `2020` KGNN: Knowledge Graph Neural Network for Drug-Drug Interaction Prediction, [IJCAI](https://www.ijcai.org/proceedings/2020/380)(KG)
- `2021` MDF-SA-DDI: predicting drug–drug interaction events based on multi-source drug fusion, multi-source feature fusion and transformer self-attention mechanism, [Briefings in Bioinformatics](https://www.researchgate.net/publication/355470462_MDF-SA-DDI_Predicting_drug-drug_interaction_events_based_on_multi-source_drug_fusion_multi-source_feature_fusion_and_transformer_self-attention_mechanism) -->
- `2020` X-MOL: large-scale pre-training for molecular understanding and diverse molecular analysis, [biorxiv](https://www.biorxiv.org/content/10.1101/2020.12.23.424259v2.full)

<!-- #### DTI

- `2019` Self-Attention Based Molecule Representation for Predicting Drug-Target Interaction, [PMLR](https://proceedings.mlr.press/v106/shin19a.html)(sequence)
- `2020` MolTrans: Molecular Interaction Transformer for Drug Target Interaction Prediction, [arXiv](http://arxiv.org/abs/2004.11424)(sequence)
- `2020` Molecule attention transformer, [arXiv](http://arxiv.org/abs/2002.08264)(graph)
- `2021` DeepMGT-DTI: Transformer network incorporating multilayer graph information for Drug–Target interaction prediction, [Computers in Biology and Medicine](https://linkinghub.elsevier.com/retrieve/pii/S0010482522000063)(graph) -->

### Molecule generation/design/edit

- `2021` LigGPT: Molecular Generation using a Transformer-Decoder Model, [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/60c7588e469df48597f456ae)(sequence)
- `2021` Transmol: repurposing a language model for molecular generation, [RSC Advances](https://pubs.rsc.org/en/content/articlelanding/2021/ra/d1ra03086h)(sequence)
- `2021` Generative Chemical Transformer: Neural Machine Learning of Molecular Geometric Structures from Chemical Language via Attention, [ACS Publications](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01289)(sequence)
- `2021` MolGPT: Molecular Generation Using a Transformer-Decoder Model, [JCIM](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00600)(sequence)
- `2021` GENERATIVE PRE-TRAINING FROM MOLECULES, [ChemRxiv](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/6142f60742198e8c31782e9e/original/generative-pre-training-from-molecules.pdf)(sequence)
- `2022` Deep learning approaches for de novo drug design: An overview, [Current Opinion in Structural Biology](https://linkinghub.elsevier.com/retrieve/pii/S0959440X21001433)(sequence)
- `2022` A Pre-trained Conditional Transformer for Target-specific De Novo Molecular Generation, [arXiv](http://arxiv.org/abs/2210.08749)(sequence)
- `2023` DOMAIN-AGNOSTIC MOLECULAR GENERATION WITH SELF-FEEDBACK, [arXiv](https://arxiv.org/abs/2301.11259v5)(sequence)
<!-- - `2023` Probabilistic generative transformer language models for generative design of molecules, [Journal of Cheminformatics](https://doi.org/10.1186/s13321-023-00759-z)(sequence) -->
- `2023` iupacGPT: IUPAC-based large-scale molecular pre-trained model for property prediction and molecule generation, [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/645f49f9a32ceeff2d90c9ae)(sequence)
- `2023` cMolGPT: A Conditional Generative Pre-Trained Transformer for Target-Specific De Novo Molecular Generation, [Molecules](https://www.mdpi.com/1420-3049/28/11/4430)(sequence)
- `2023` Molecule generation using transformers and policy gradient reinforcement learning, [Nature](https://www.nature.com/articles/s41598-023-35648-w)(sequence)

### Reaction prediction

#### Forward reaction prediction

- `2019` Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction, [ACS Publications](https://doi.org/10.1021/acscentsci.9b00576)(sequence)
- `2021` Quantitative interpretation explains machine learning models for chemical reaction prediction and uncovers bias, [Nature Communications](https://www.nature.com/articles/s41467-021-21895-w)
- `2021` Predicting Chemical Reaction Outcomes: A Grammar Ontology-based Transformer Framework, [AIChE Journal](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.17190)
- `2021` Prediction of chemical reaction yields using deep learning, [IOP Science](https://iopscience.iop.org/article/10.1088/2632-2153/abc81d)(sequence)

#### Retrosynthetic analysis

- `2019` Predicting Retrosynthetic Reaction using Self-Corrected Transformer Neural Networks, [arXiv](https://arxiv.org/abs/1907.01356)(sequence)
- `2019` A Transformer Model for Retrosynthesis, [Springer International Publishing](http://link.springer.com/10.1007/978-3-030-30493-5_78)(sequence)
- `2020` State-of-the-art augmented NLP transformer models for direct and single-step retrosynthesis, [Nature Communications](https://www.nature.com/articles/s41467-020-19266-y)
- `2021` Valid, Plausible, and Diverse Retrosynthesis Using Tied Two-Way Transformers with Latent Variables, [Journal of Chemical Information and Modeling](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01074)
- `2021` RetroTRAE: retrosynthetic translation of atomic environments with Transformer, [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6117ed347117505183e94d93)(sequence)
- `2021` Molecular Graph Enhanced Transformer for Retrosynthesis Prediction, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231221009413)(sequence+graph)
- `2022` Retrosynthetic reaction pathway prediction through neural machine translation of atomic environments, [Nature Communications](https://www.nature.com/articles/s41467-022-28857-w)(sequence)
- `2023` Enhancing diversity in language based models for single-step retrosynthesis, [Digital Discovery](http://xlink.rsc.org/?DOI=D2DD00110A)(sequence)
- `2023` Unbiasing Retrosynthesis Language Models with Disconnection Prompts, [ACS Publications](https://doi.org/10.1021/acscentsci.3c00372)(sequence)

#### Reaction prediction and retrosynthesis

- `2019` Molecular Transformer unifies reaction prediction and retrosynthesis across pharma chemical space, [Chemical Communications](http://xlink.rsc.org/?DOI=C9CC05122H)(sequence)
- `2021` PERMUTATION INVARIANT GRAPH-TO-SEQUENCE MODEL FOR TEMPLATE-FREE RETROSYNTHESIS AND REACTION PREDICTION, [arXiv](https://arxiv.org/abs/2110.09681)(graph)

### Datasets and Benchmarks
- `2005` ZINC − A Free Database of Commercially Available Compounds for Virtual Screening, [ACS Publications](https://pubs.acs.org/doi/10.1021/ci049714%2B?ref=PDF)
- `2009` 970 Million Druglike Small Molecules for Virtual Screening in the Chemical Universe Database GDB-13, [ACS Publications](https://pubs.acs.org/doi/full/10.1021/ja902302h)
- `2012` PubChem's BioAssay Database, [Nucleic Acids Research](https://academic.oup.com/nar/article/40/D1/D400/2903189?login=false)
- `2012` Enumeration of 166 Billion Organic Small Molecules in the Chemical Universe Database GDB-17, [ACS Publications](https://pubs.acs.org/doi/10.1021/ci300415d)
- `2012` Extraction of chemical structures and reactions from the literature, [University of Cambridge](http://www.dspace.cam.ac.uk/handle/1810/244727)
- `2015` ZINC 15 – Ligand Discovery for Everyone, [ACS Publications](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00559)
- `2016` PubChem Substance and Compound databases, [Nucleic Acids Research](https://pubchem.ncbi.nlm.nih.gov/)
- `2016` What’s What: The (Nearly) Definitive Guide to Reaction Role Assignment, [ACS Publications](https://doi.org/10.1021/acs.jcim.6b00564)
- `2018` ChEMBL: towards direct deposition of bioassay data, [Nucleic Acids Research](https://academic.oup.com/nar/article/47/D1/D930/5162468?login=false)
- `2018` DrugBank 5.0: a major update to the DrugBank database for 2018, [Nucleic Acids Research](https://academic.oup.com/nar/article/46/D1/D1074/4602867?login=false)
- `2018` Comparative Assessment of Scoring Functions: The CASF-2016 Update, [Journal of chemical information and modeling](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.8b00545)
- `2018` MoleculeNet: a benchmark for molecular machine learning, [Chemical Science](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a)
- `2019` GuacaMol: Benchmarking Models for de Novo Molecular Design, [ACS Publications](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839?ref=PDF)
- `2020` Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models, [frontiers](https://www.frontiersin.org/articles/10.3389/fphar.2020.565644/full)
- `2020` ZINC20—A Free Ultralarge-Scale Chemical Database for Ligand Discovery, [JCIM](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00559)
- `2021` ADMETlab 2.0: an integrated online platform for accurate and comprehensive predictions of ADMET properties, [Nucleic Acids Res](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8262709/)
- `2021` First Place Solution of KDD Cup 2021 & OGB Large-Scale Challenge Graph Prediction Track, [arXiv](http://arxiv.org/abs/2106.08279)
- `2021` OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs, [arXiv](http://arxiv.org/abs/2103.09430)
- `2022` GEOM, energy-annotated molecular conformations for property prediction and molecular generation, [Nature](https://www.nature.com/articles/s41597-022-01288-4)
- `2023` PubChem 2023 update, [Nucleic Acids Research](https://doi.org/10.1093/nar/gkac956)
- `2023` Towards Foundational Models for Molecular Learning on Large-Scale Multi-Task Datasets, [arXiv](https://arxiv.org/pdf/2310.04292v2.pdf)
- `2023` The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods, [Nucleic Acids Research](https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkad1004/7337608)

### Others
- `2023` Molecular language models: RNNs or transformer?, [Briefings in Functional Genomics](https://academic.oup.com/bfg/article/22/4/392/7109964)
- `2023` Language models in molecular discovery, [arXiv](http://arxiv.org/abs/2309.16235)
- `2023` Transformers and Large Language Models for Chemistry and Drug Discovery, [arXiv](http://arxiv.org/abs/2310.06083)

## 🦠 Gene-LLMs 
### General Analysis
- `2023` DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks [BioRxiv](https://www.biorxiv.org/content/10.1101/2023.07.11.548628v2)
- `2023` The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v3.full.pdf+html)
- `2023` HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution [arXiv](https://arxiv.org/abs/2306.15794)
- `2023` GeneGPT: Augmenting Large Language Models with Domain Tools for Improved Access to Biomedical Information [arXiv](https://arxiv.org/abs/2304.09667v3)
- `2023` GENA-LM: A Family of Open-Source Foundational Models for Long DNA Sequences [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v1.abstract)
- `2023` DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome [arXiv](https://arxiv.org/abs/2306.15006)
- `2023` Species-aware DNA language modeling [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.01.26.525670.abstract)
- `2022` MoDNA: motif-oriented pre-training for DNA language model [BCB 2022](https://dl.acm.org/doi/10.1145/3535508.3545512)
- `2022` Integrating convolution and self-attention improves language model of human genome for interpreting non-coding regions at base-resolution [Nucleic Acids Research](https://academic.oup.com/nar/article/50/14/e81/6583232)
- `2022` Fine-Tuning Transformers For Genomic Tasks [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.02.07.479412v1.abstract)
- `2022` iDNA-ABF: multi-scale deep biological language learning model for the interpretable prediction of DNA methylations [Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02780-1)
- `2022` Genomics enters the deep learning era [PeerJ](https://peerj.com/articles/13613/)
- `2021` Effective gene expression prediction from sequence by integrating long-range interactions [Nature Methods](https://www.nature.com/articles/s41592-021-01252-x)
- `2021` DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome [Bioinformatics](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680)

### Gene Expression and Regulatory Element Prediction
- `2023` Species-aware DNA language models capture regulatory elements and their evolution [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.01.26.525670v2.abstract)
- `2023` PLPMpro: Enhancing promoter sequence prediction with prompt-learning based pre-trained language model [Computers in Biology and Medicine](https://www.sciencedirect.com/science/article/abs/pii/S0010482523007254)
- `2023` A single-cell gene expression language model [NeurIPS 2022 Workshop](https://arxiv.org/abs/2210.14330)
- `2022` iEnhancer-BERT: A Novel Transfer Learning Architecture Based on DNA-Language Model for Identifying Enhancers and Their Strength [ICIC 2022](https://link.springer.com/chapter/10.1007/978-3-031-13829-4_13)
- `2022` iEnhancer-ELM: improve enhancer identification by extracting position-related multiscale contextual information based on enhancer language models [arXiv](https://arxiv.org/abs/2212.01495)
<!-- - `2022` ~~iPromoter-Seqvec: identifying promoters using bidirectional long short-term memory and sequence-embedded features [BMC genomics](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-022-08829-6)~~
- `2022` ~~iPro-WAEL: a comprehensive and robust framework for identifying promoters in multiple species [Nucleic Acids Research](https://academic.oup.com/nar/article-abstract/50/18/10278/6717829)~~ -->
- `2022` DeeProPre: A promoter predictor based on deep learning [Computational Biology and Chemistry](https://www.sciencedirect.com/science/article/pii/S1476927122001505)
<!-- - `2022` ~~The evolution, evolvability and engineering of gene regulatory DNA [Nature](https://www.nature.com/articles/s41586-022-04506-6)~~
- `2022` ~~DeePromClass: Delineator for Eukaryotic Core Promoters Employing Deep Neural Networks [IEEE/ACM Transactions on Computational Biology and Bioinformatics](https://ieeexplore.ieee.org/abstract/document/9745351/)~~ -->
- `2021` A transformer architecture based on BERT and 2D convolutional neural network to identify DNA enhancers from sequence information [Briefings in Bioinformatics](https://academic.oup.com/bib/article/22/5/bbab005/6128847)
<!-- - `2021` ~~Computational identification of eukaryotic promoters based on cascaded deep capsule neural networks [Briefings in Bioinformatics](https://academic.oup.com/bib/article-abstract/22/4/bbaa299/5998831)~~
- `2021` ~~TSSFinder—fast and accurate ab initio prediction of the core promoter in eukaryotic genomes [Briefings in Bioinformatics](https://academic.oup.com/bib/article-abstract/22/6/bbab198/6287335)~~
- `2020` ~~A unified framework for integrative study of heterogeneous gene regulatory mechanisms [Nature Machine Intelligence](https://www.nature.com/articles/s42256-020-0205-2)~~
- `2020` ~~Cross-species regulatory sequence activity prediction [Plos Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008050)~~
- `2019` ~~Promoter analysis and prediction in the human genome using sequence-based deep learning models [Bioinformatics](https://academic.oup.com/bioinformatics/article/35/16/2730/5270663)~~
- `2019` ~~SpliceFinder: ab initio prediction of splice sites using convolutional neural network [BMC bioinformatics](https://link.springer.com/article/10.1186/s12859-019-3306-3)~~
- `2018` ~~Sequential regulatory activity prediction across chromosomes with convolutional neural networks [Genome Research](https://genome.cshlp.org/content/early/2018/03/27/gr.227819.117)~~
- `2016` ~~DanQ: a hybrid convolutional and recurrent deep neural network for quantifying the function of DNA sequences [Nucleic acids research](https://academic.oup.com/nar/article-abstract/44/11/e107/2468300)~~ -->

### RNA Analysis and Prediction

- `2023` Multiple sequence-alignment-based RNA language model and its application to structural inference [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.03.15.532863v1.abstract)
- `2023` Prediction of Multiple Types of RNA Modifications via Biological Language Model [IEEE/ACM Transactions on Computational Biology and Bioinformatics](https://ieeexplore.ieee.org/abstract/document/10146457)
- `2023` miProBERT: identification of microRNA promoters based on the pre-trained model BERT [Briefings in Bioinformatics](https://academic.oup.com/bib/article-abstract/24/3/bbad093/7079709)
- `2023` Self-supervised learning on millions of pre-mRNA sequences improves sequence-based RNA splicing prediction [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.01.31.526427.abstract)
- `2022` scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data [Nature Machine Intelligence](https://www.nature.com/articles/s42256-022-00534-z)
- 
### Protein Binding Site Prediction
- `2023` Improving language model of human genome for DNA–protein binding prediction based on task-specific pre-trainin [Interdisciplinary Sciences: Computational Life Sciences volume](https://link.springer.com/article/10.1007/s12539-022-00537-9)
- `2022` Comprehensive Evaluation of BERT Model for DNA-Language for Prediction of DNA Sequence Binding Specificities in Fine-Tuning Phase [ICIC 2022](https://link.springer.com/chapter/10.1007/978-3-031-13829-4_8)
- `2022` PepNN: a deep attention model for the identification of peptide binding sites [Communications Biology](https://www.nature.com/articles/s42003-022-03445-2)
- `2020` DeepSite: bidirectional LSTM and CNN models for predicting DNA–protein binding [International Journal of Machine Learning and Cybernetics](https://link.springer.com/article/10.1007/s13042-019-00990-x)
- `2018` A novel method for improved accuracy of transcription factor binding site prediction [Nucleic acids research](https://academic.oup.com/nar/article-abstract/46/12/e72/4958206)
- `2016` Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks [Genome research](https://genome.cshlp.org/content/26/7/990.short)
- `2015` Predicting the sequence specificities of DNA-and RNA-binding proteins by deep learning [Nature biotechnology](https://www.nature.com/articles/nbt.3300.)
- `2013` Jaccard index based similarity measure to compare transcription factor binding site models [Algorithms for Molecular Biology](https://almob.biomedcentral.com/articles/10.1186/1748-7188-8-23)

### Sequence Variation and Evolution Analysis
- `2023` GPN-MSA: an alignment-based DNA language model for genome-wide variant effect prediction [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.10.10.561776v1.abstract)
- `2022` GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.10.10.511571v2)
- `2022` DNA language models are powerful predictors of genome-wide variant effects [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.08.22.504706v3.abstract)
- `2018` Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk [Nature genetics](https://www.nature.com/articles/s41588-018-0160-6)
- `2015` Predicting effects of noncoding variants with deep learning–based sequence model [Nature methods](https://www.nature.com/articles/nmeth.3547)

### Sequence Classification and Feature Selection
- `2023` Hamming Encoder: Mining Discriminative k-mers for Discrete Sequence Classification [arXiv](https://arxiv.org/abs/2310.10321)

### Datasets and Benchmarks
- `2023` A systematic benchmark of machine learning methods for protein–RNA interaction prediction [Briefings in Bioinformatics](https://academic.oup.com/bib/article-abstract/24/5/bbad307/7252289)
- `2022` DESSO-DB: A web database for sequence and shape motif analyses and identification [Computational and Structural Biotechnology Journal](https://www.sciencedirect.com/science/article/pii/S2001037022002422)
- `2012` ENCODE data in the UCSC Genome Browser: year 5 update [Nucleic acids research](https://academic.oup.com/nar/article-abstract/41/D1/D56/1066727)
- `2012` An integrated encyclopedia of DNA elements in the human genome[Nature](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3439153/)
- `2000` The eukaryotic promoter database [Nucleic acids research](https://academic.oup.com/nar/article-abstract/28/1/302/2384389)

### Other Downstream Tasks
- `2023` MuLan-Methyl - Multiple Transformer-based Language Models for Accurate DNA Methylation Prediction [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.01.04.522704v2.abstract)



## Ⓜ️ MM-Sci-LLMs

### Protein-text
- `2023` InstructProtein: Aligning Human and Protein Language via Knowledge Instruction, [arXiv](https://arxiv.org/abs/2310.03269)
- `2023` Prot2Text: Multimodal Protein's Function Generation with GNNs and Transformers, [arXiv](https://arxiv.org/abs/2307.14367)
- `2023` ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts, [arXiv](https://arxiv.org/abs/2301.12040)
<!-- - `2023` Improved global protein homolog detection with major gains in function identification, [PNAS](https://www.pnas.org/doi/full/10.1073/pnas.2211823120)-->
- `2023` A Text-guided Protein Design Framework, [arXiv](https://arxiv.org/abs/2302.04611)
- `2023` Multilingual translation for zero-shot biomedical classification using BioTranslator, [Nature](https://www.nature.com/articles/s41467-023-36476-2)

### Molecule-text
- `2023` Generating Novel Leads for Drug Discovery using LLMs with Logical Feedback, [bioArxiv](https://www.biorxiv.org/content/10.1101/2023.09.14.557698v1.abstract)
- `2023` GIT-Mol: A Multi-modal Large Language Model for Molecular Science with Graph, Image, and Text, [arXiv](https://arxiv.org/abs/2308.06911)
- `2023` MolFM: A Multimodal Molecular Foundation Model, [arXiv](https://arxiv.org/abs/2307.09484)
- `2023` Can Large Language Models Empower Molecular Property Prediction?, [arXiv](https://arxiv.org/abs/2307.07443)
- `2023` MolXPT: Wrapping Molecules with Text for Generative Pre-training, [ACL](https://aclanthology.org/2023.acl-short.138/)
- `2023` Enhancing Activity Prediction Models in Drug Discovery with the Ability to Understand Human Language, [arXiv](https://arxiv.org/abs/2303.03363)
- `2023` Empowering Molecule Discovery for Molecule-Caption Translation with Large Language Models: A ChatGPT Perspective, [arXiv](https://arxiv.org/abs/2306.06615)
- `2023` Unifying Molecular and Textual Representations via Multi-task Language Modelling, [arXiv](https://arxiv.org/abs/2301.12586)
- `2023` Exploring the potential of AI-Chatbots in organic chemistry: An assessment of ChatGPT and Bard, [Computers and Education: Artificial Intelligence](https://www.sciencedirect.com/science/article/pii/S2666920X23000498?via%3Dihub)
- `2022` Multi-modal Molecule Structure-text Model for Text-based Retrieval and Editing, [arXiv](https://arxiv.org/abs/2212.10789)
- `2022` Translation between Molecules and Natural Language, [arXiv](https://arxiv.org/abs/2204.11817)
- `2022` A Molecular Multimodal Foundation Model Associating Molecule Graphs with Natural Language, [arXiv](https://arxiv.org/abs/2209.05481)
- `2022` A deep-learning system bridging molecule structure and biomedical text with comprehension comparable to human professionals, [nature](https://www.nature.com/articles/s41467-022-28494-3)
- `2021` Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries, [EMNLP](https://aclanthology.org/2021.emnlp-main.47/)
- `2023` DrugChat: Towards Enabling ChatGPT-Like Capabilities on Drug Molecule Graphs, [techRxiv](https://www.techrxiv.org/articles/preprint/DrugChat_Towards_Enabling_ChatGPT-Like_Capabilities_on_Drug_Molecule_Graphs/22945922)
- `2023` Language models in molecular discovery, [arXiv](https://arxiv.org/abs/2309.16235)

<!-- ### Genome-text
- `2023` Evaluation of large language models for discovery of gene set function, [arXiv](https://arxiv.org/abs/2309.04019)
- `2023` Towards Generalist Biomedical AI, [arXiv](https://arxiv.org/abs/2307.14334)
- `2023` Foundation models for generalist medical artificial intelligence, [nature](https://www.nature.com/articles/s41586-023-05881-4) -->

### Protein-molecule
- `2023` General-purpose Pre-trained Model Towards Cross-domain Molecule Learning, [ICLR2024](https://openreview.net/forum?id=tNAucRS0QQ)
- `2023` FABind: Fast and Accurate Protein-Ligand Binding, [arXiv](https://arxiv.org/abs/2310.06763)
- `2023` Self-supervised Pocket Pretraining via Protein Fragment-Surroundings Alignment, [arXiv](https://arxiv.org/abs/2310.07229)
- `2023` DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening, [arXiv](https://arxiv.org/abs/2310.06367)
- `2023` CProMG: controllable protein-oriented molecule generation with desired binding affinity and drug-like properties, [Bioinformatics](https://academic.oup.com/bioinformatics/article/39/Supplement_1/i326/7210458)
- `2023` DrugGPT: A GPT-based Strategy for Designing Potential Ligands Targeting Specific Proteins, [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.06.29.543848v1)
- `2023` Contrastive learning in protein language space predicts interactions between drugs and protein targets, [PANS](https://www.pnas.org/doi/10.1073/pnas.2220778120)
- `2023` Deep generative model for drug design from protein target sequence, [Journal of Cheminformatics](https://link.springer.com/article/10.1186/s13321-023-00702-2)
- `2022` Exploiting pretrained biochemical language models for targeted drug design, [Bioinformatics](https://academic.oup.com/bioinformatics/article/38/Supplement_2/ii155/6702010)
- `2022` Implications of the Essential Role of Small Molecule Ligand Binding Pockets in Protein–Protein Interactions, [The Journal of Physical Chemistry B](https://pubs.acs.org/doi/10.1021/acs.jpcb.2c04525)
- `2021` Transformer neural network for protein-specific de novo drug generation as a machine translation problem, [Scientific Reports volume](https://www.nature.com/articles/s41598-020-79682-4)

### Protein-molecule-text
- `2023` The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4, [arXiv](https://arxiv.org/abs/2311.07361)
- `2023` BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations, [arXiv](https://arxiv.org/abs/2310.07276)
- `2023` Towards Unified AI Drug Discovery with Multiple Knowledge Modalities, [arXiv](https://arxiv.org/abs/2305.01523)
- `2023` Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models, [arXiv](https://arxiv.org/abs/2306.08018)
- `2023` DARWIN Series: Domain Specific Large Language Models for Natural Science, [arXiv](https://arxiv.org/abs/2308.13565)
- `2023` BioMedGPT：A Pre-trained Language Model for Biomedical Text Mining, [arXiv](https://arxiv.org/abs/2308.09442v2)
- `2023` ChatGPT-powered Conversational Drug Editing Using Retrieval and Domain Feedback, [arXiv](https://arxiv.org/abs/2305.18090)
- `2022` Galactica: A Large Language Model for Science, [arXiv](http://arxiv.org/abs/2211.09085)

### Datasets and Benchmarks
#### Text-Protein
- `2023` InstructProtein: Aligning Human and Protein Language via Knowledge Instruction, [arXiv](https://arxiv.org/abs/2310.03269)[InstructProtein]
- `2023` ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts, [arXiv](https://arxiv.org/abs/2301.12040)[ProtDescribe]
- `2023` A Text-guided Protein Design Framework, [arXiv](https://arxiv.org/abs/2302.04611)[SwissProtCLAP]
#### Text-Molecule
- `2021` Text2mol: Cross-modal molecule retrieval with natural language queries, [EMNLP2021](https://aclanthology.org/2021.emnlp-main.47/)[ChEBI-20]
- `2022` Multi-modal Molecule Structure-text Model for Text-based Retrieval and Editing, [arXiv](https://arxiv.org/abs/2212.10789)[PubChemSTM]
- `2022` A Molecular Multimodal Foundation Model Associating Molecule Graphs with Natural Language, [arXiv](https://arxiv.org/abs/2209.05481)[MoMu]
- `2022` A deep-learning system bridging molecule structure and biomedical text with comprehension comparable to human professionals, [nature](https://www.nature.com/articles/s41467-022-28494-3)[PCdes]
#### Protein-Molecule
- `2015` BindingDB in 2015: A public database for medicinal chemistry, computational chemistry and systems pharmacology, [Nucleic Acids Research](https://academic.oup.com/nar/article/44/D1/D1045/2502601)[BindingDB]
- `2012` BioLiP: a semi-manually curated database for biologically relevant ligand–protein interactions, [Nucleic Acids Research](https://academic.oup.com/nar/article/41/D1/D1096/1074898)[BioLiP]
- `2012` Directory of Useful Decoys, Enhanced (DUD-E): Better Ligands and Decoys for Better Benchmarking, [Journal of Medicinal Chemistry](https://pubs.acs.org/doi/10.1021/jm300687e)[DUD-E]
#### Comprehensive
- `2022` Galactica: A Large Language Model for Science, [arXiv](https://arxiv.org/abs/2211.09085)[Galactica]
- `2023` Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models, [arXiv](https://arxiv.org/abs/2306.08018)[Mol-Instructions]
### Others
- `2022` PanGu Drug Model: learn a molecule like a human, [Science China Life Sciences](https://link.springer.com/article/10.1007/s11427-022-2239-y)
- `2023` Symmetry-Informed Geometric Representation for Molecules, Proteins, and Crystalline Materials, [arXiv](https://arxiv.org/abs/2306.09375)

<!-- - `2023` The status of the human gene catalogue, [nature](https://www.nature.com/articles/s41586-023-06490-x)[genome-image]
- `2023` Minimum resolution requirements of digital pathology images for accurate classification, [Medical Image Analysis](https://www.sciencedirect.com/science/article/pii/S1361841523001512)[genome-image]
- `2023` Transformer-based biomarker prediction from colorectal cancer histology: A large-scale multicentric study, [Cancer Cell](https://www.cell.com/cancer-cell/fulltext/S1535-6108(23)00278-7)[genome-image]
- `2023` A visual–language foundation model for pathology image analysis using medical Twitter, [nature](https://www.nature.com/articles/s41591-023-02504-3)[genome-image]
- `2023` Analysis of Specimen Mammography with Artificial Intelligence to Predict Margin Status, [Annals of Surgical Oncology](https://link.springer.com/article/10.1245/s10434-023-14083-1)[genome-image]
- `2023` Spatial cellular architecture predicts prognosis in glioblastoma, [nature](https://www.nature.com/articles/s41467-023-39933-0)[genome-image]
- `2023` SCS: cell segmentation for high-resolution spatial transcriptomics, [nature](https://www.nature.com/articles/s41592-023-01939-3)[genome-image]
- `2023` Cell-type-specific prediction of 3D chromatin organization enables high-throughput in silico genetic screening, [nature](https://www.nature.com/articles/s41587-022-01612-8)[genome-image]
- `2022` Pan-cancer integrative histology-genomic analysis via multimodal deep learning, [Cancer Cell](https://www.cell.com/cancer-cell/fulltext/S1535-6108(22)00317-8)[genome-image]
- `2022` Cross-modal Graph Contrastive Learning with Cellular Images, [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.06.05.494905v2)[genome-image] -->







## 👥 Contributions

### Citation
If you find this repository useful, please cite our paper:

```
@misc{zhang2024sci-llms,
      title={Scientific Large Language Models: A Survey on Biochemical Domain}, 
      author={},
      year={2024},
      eprint={},
      archivePrefix={arXiv},
}
```
### Contributors

- Keyan Ding [@dingkeyan93](https://github.com/dingkeyan93)
- 

### Contact
- 

[![Star History Chart](https://api.star-history.com/svg?repos=HICAI-ZJU/Scientific-LLM-Survey&type=Date)](https://star-history.com/#HICAI-ZJU/Scientific-LLM-Survey&Date)
<!-- ### 🎉 Contributing ( welcome ! )

- ✨ Add a new paper or update an existing Protein-related LLM paper.
- 🧐 Use the same format as existing entries to describe the work.
- 😄 A very brief explanation why you think a paper should be added or updated is recommended (Not Neccessary) via **`Adding Issues`** or **`Pull Requests`**.

**Don't worry if you put something wrong, they will be fixed for you. Just feel free to contribute and promote your awesome work here! 🤩 We'll get back to you in time ~ 😉** -->


