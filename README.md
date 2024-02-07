# Scientific Large Language Models (Sci-LLMs)

This repository collects papers on scientific large language models, particularly in the domains of biology and chemistry.

> 😎 Welcome to recommend missing papers through **`Adding Issues`** or **`Pull Requests`**.

## 🔔 News

- 💥 [2024/01] Our survey is released! See [Scientific Large Language Models: A Survey on Biological &amp; Chemical Domains](https://arxiv.org/abs/2401.14656) for the paper!

![Sci-LLMs-Scopes](figures/sci-llms-scopes.png)
In this survey, we focus on scientific languages (i.e., textual, molecular, protein and genomic languages), as well as their combination (i.e., multimodal language).

## 🌟 Contents

- [Scientific Large Language Models (Sci-LLMs)](#scientific-large-language-models-sci-llms)
  - [🔔 News](#-news)
  - [🌟 Contents](#-contents)
  - [📖 Textual Scientific Large Language Models (Text-Sci-LLMs)](#-textual-scientific-large-language-models-text-sci-llms)
    - [Biology](#biology)
    - [Chemistry](#chemistry)
    - [Comprehensive](#comprehensive)
    - [Datasets and Benchmarks](#datasets-and-benchmarks)
  - [🧪 Molecular Large Language Models (Mol-LLMs)](#-molecular-large-language-models-mol-llms)
    - [Molecule Property Prediction](#molecule-property-prediction)
    - [Interaction Prediction](#interaction-prediction)
    - [Molecule Generation/Design/Edit](#molecule-generationdesignedit)
    - [Reaction Prediction](#reaction-prediction)
    - [Datasets and Benchmarks](#datasets-and-benchmarks-1)
  - [🧬 Protein Large Language Models (Prot-LLMs)](#-protein-large-language-models-prot-llms)
    - [Protein Sequence Representation](#protein-sequence-representation)
    - [Protein Sequence Generation/Design](#protein-sequence-generationdesign)
    - [Datasets and Benchmarks](#datasets-and-benchmarks-2)
  - [🦠 Genomic Large Language Models (Gene-LLMs)](#-genomic-large-language-models-gene-llms)
    - [General](#general)
    - [Function Prediction](#function-prediction)
    - [Variants and Evolution Prediction](#variants-and-evolution-prediction)
    - [DNA-Protein Interaction Prediction](#dna-protein-interaction-prediction)
    - [RNA Prediction](#rna-prediction)
    - [Datasets and Benchmarks](#datasets-and-benchmarks-3)
  - [Ⓜ️ Multimodal Scientific Large Language Models (MM-Sci-LLMs)](#️-multimodal-scientific-large-language-models-mm-sci-llms)
    - [Molecule\&text](#moleculetext)
    - [Protein\&text](#proteintext)
    - [Protein\&molecule](#proteinmolecule)
    - [Comprehensive](#comprehensive-1)
    - [Datasets and Benchmarks](#datasets-and-benchmarks-4)
      - [Molecule\&Text](#moleculetext-1)
      - [Protein\&Text](#proteintext-1)
      - [Protein\&Molecule](#proteinmolecule-1)
      - [Comprehensive](#comprehensive-2)
  - [👥 Contributions](#-contributions)
    - [Citation](#citation)
    - [Contributors](#contributors)
    - [Contact](#contact)

## 📖 Textual Scientific Large Language Models (Text-Sci-LLMs)

### Biology

- `2019.05` BioBERT: a pre-trained biomedical language representation model for biomedical text mining, [arXiv](https://arxiv.org/abs/1901.08746), [Code](https://github.com/dmis-lab/biobert)
- `2019.07` Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets, [arXiv](https://arxiv.org/abs/1906.05474), [Code](https://github.com/ncbi-nlp/bluebert?tab=readme-ov-file)
- `2020.10` BioMegatron: Larger Biomedical Domain Language Model, [arXiv](https://arxiv.org/abs/2010.06060), [Code](https://catalog.ngc.nvidia.com/?filters=&orderBy=weightPopularDESC&query=)
- `2020.10` Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing, [arXiv](https://arxiv.org/pdf/2007.15779.pdf), [Hugging Face](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)
- `2021.06` BioM-Transformers: Building Large Biomedical Language Models with BERT, ALBERT and ELECTRA, [ACL Anthology](https://aclanthology.org/2021.bionlp-1.24/), [Code](https://github.com/salrowili/BioM-Transformers)
- `2022.03` LinkBERT: Pretraining Language Models with Document Links, [arXiv](https://arxiv.org/abs/2203.15827), [Code](https://github.com/salrowili/BioM-Transformers)
- `2023.03` BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining, [arXiv](https://arxiv.org/abs/2210.10341), [Code](https://github.com/microsoft/BioGPT)
- `2023.08` BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine, [arXiv](https://arxiv.org/abs/2308.09442), [Code](https://github.com/PharMolix/OpenBioMed?tab=readme-ov-file)

### Chemistry

- `2021.06` Automated Chemical Reaction Extraction from Scientific Literature. [Journal of Chemical Information and Modeling](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00284), [Code](https://github.com/jiangfeng1124/ChemRxnExtractor)
- `2021.09` MatSciBERT: A materials domain language model for text mining and information extraction, [npj Computational Materials](https://www.nature.com/articles/s41524-022-00784-w), [Code](https://github.com/M3RG-IITD/MatSciBERT?tab=readme-ov-file)
- `2022.09` A general-purpose material property data extraction pipeline from large polymer corpora using natural language processing, [npj Computational Materials](https://www.nature.com/articles/s41524-023-01003-w), [Hugging Face](https://huggingface.co/pranav-s/MaterialsBERT)

### Comprehensive

- `2019.09` SciBERT: A Pretrained Language Model for Scientific Text, [arXiv](https://arxiv.org/abs/1903.10676), [Code](https://github.com/allenai/scibert/)
- `2023.05` The Diminishing Returns of Masked Language Models to Science, [arXiv](https://arxiv.org/abs/2205.11342), [Hugging Face](https://huggingface.co/globuslabs/ScholarBERT)
- `2023.08` DARWIN Series: Domain Specific Large Language Models for Natural Science, [arXiv](https://arxiv.org/abs/2308.13565), [Code](https://github.com/MasterAI-EAM/Darwin)

### Datasets and Benchmarks

- [MMLU](https://huggingface.co/datasets/cais/mmlu), `2020.09`. Measuring Massive Multitask Language Understanding, [arXiv](https://arxiv.org/abs/2009.03300)
- [C-Eval](https://huggingface.co/datasets/ceval/ceval-exam), `2023.05`. C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models, [arXiv](https://arxiv.org/abs/2305.08322)
- [AGIEval](https://huggingface.co/datasets/baber/agieval) `2023.05`. AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models, [arXiv](https://arxiv.org/abs/2304.06364)
- [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA), 2022.09. Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering, [arXiv](https://arxiv.org/abs/2209.09513)
- [Xiezhi](https://github.com/MikeGu721/XiezhiBenchmark), `2023.06`. Xiezhi: An Ever-Updating Benchmark for Holistic Domain Knowledge Evaluation, [arXiv](https://arxiv.org/abs/2306.05783)
- [SciEval](https://github.com/OpenDFM/SciEval), `2023.08`. SciEval: A Multi-Level Large Language Model Evaluation Benchmark for Scientific Research, [arXiv](https://arxiv.org/abs/2308.13149)
- [Bioinfo-Bench](https://github.com/cinnnna/bioinfo-bench), `2023.10`. A Simple Benchmark Framework for LLM Bioinformatics Skills Evaluation, [bioRxiv](https://www.bioRxiv.org/content/10.1101/2023.10.18.563023v1.abstract)
- [BLURB](https://huggingface.co/datasets/EMBO/BLURB), `2020.07`. Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing, [arXiv](https://arxiv.org/abs/2007.15779)
- [ARC](https://huggingface.co/datasets/allenai/ai2_arc), `2018.03`. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge, [arXiv](https://arxiv.org/abs/1803.05457)
- [SciQ](https://huggingface.co/datasets/sciq), `2017.07`. Crowdsourcing Multiple Choice Science Questions, [arXiv](https://arxiv.org/abs/1707.06209v1)

## 🧪 Molecular Large Language Models (Mol-LLMs)

### Molecule Property Prediction

- `2019.09` SMILES-BERT: Large Scale Unsupervised Pre-Training for Molecular Property Prediction, [ACM-BCB](https://dl.acm.org/doi/10.1145/3307339.3342186), [Code](https://github.com/uta-smile/SMILES-BERT)
- `2019.11` SMILES Transformer: Pre-trained Molecular Fingerprint for Low Data Drug Discovery, [arXiv](https://arxiv.org/abs/1911.04738v1), [Code](https://github.com/DSPsleeporg/smiles-transformer)
- `2020.02` Molecule attention transformer, [arXiv](http://arxiv.org/abs/2002.08264), [Code](https://github.com/ardigen/MAT)
- `2020.10` ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction, [arXiv](http://arxiv.org/abs/2010.09885), [Code](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)
- `2020.10` Self-Supervised Graph Transformer on Large-Scale Molecular Data, [arXiv](http://arxiv.org/abs/2007.02835), [Code](https://github.com/tencent-ailab/grover)
- `2021.05` MG-BERT: leveraging unsupervised atomic representation learning for molecular property prediction, [Briefings in Bioinformatics](https://www.researchgate.net/publication/351363304_MG-BERT_leveraging_unsupervised_atomic_representation_learning_for_molecular_property_prediction), [Code](https://github.com/zhang-xuan1314/Molecular-graph-BERT)
- `2021.06` Algebraic graph-assisted bidirectional transformers for molecular property prediction, [Nature Communications](https://www.nature.com/articles/s41467-021-23720-w), [Code](https://github.com/ChenDdon/AGBTcode)
- `2021.09` Mol-BERT: An Effective Molecular Representation with BERT for Molecular Property Prediction, [Wireless Communications and Mobile Computing](https://www.hindawi.com/journals/wcmc/2021/7181815/), [Code](https://github.com/cxfjiang/MolBERT)
- `2022.08` KPGT: Knowledge-Guided Pre-training of Graph Transformer for Molecular Property Prediction, [Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining](https://dl.acm.org/doi/10.1145/3534678.3539426), [Code](https://github.com/lihan97/KPGT)
- `2022.09` ChemBERTa-2: Towards Chemical Foundation Models, [arXiv](http://arxiv.org/abs/2209.01712), [Code](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)
- `2022.01` Chemformer: a pre-trained transformer for computational chemistry, [Mach. Learn.: Sci. Technol.](https://iopscience.iop.org/article/10.1088/2632-2153/ac3ffb/meta), [Code](https://github.com/MolecularAI/Chemformer)
- `2022.10` Large-Scale Distributed Training of Transformers for Chemical Fingerprinting, [JCIM](https://doi.org/10.1021/acs.jcim.2c00715), [Code](https://github.com/GouldGroup/MFBERT)
- `2022.11` BARTSmiles: Generative Masked Language Models for Molecular Representations, [arXiv](https://arxiv.org/abs/2211.16349), [Code](https://github.com/YerevaNN/BARTSmiles/)
- `2022.12` Large-Scale Chemical Language Representations Capture Molecular Structure and Properties, [arXiv](http://arxiv.org/abs/2106.09553), [Code](https://github.com/IBM/molformer)
- `2022.12` Pushing the Boundaries of Molecular Property Prediction for Drug Discovery with Multitask Learning BERT Enhanced by SMILES Enumeration, [Research](https://spj.science.org/doi/10.34133/research.0004), [Code](https://github.com/zhang-xuan1314/MTL-BERT)
- `2023.01` MolRoPE-BERT: An enhanced molecular representation with Rotary Position Embedding for molecular property prediction, [Journal of Molecular Graphics and Modelling](https://linkinghub.elsevier.com/retrieve/pii/S1093326322002236)
- `2023.01` Molformer: Motif-based Transformer on 3D Heterogeneous Molecular Graphs, [arXiv](http://arxiv.org/abs/2110.01191), [Code](https://github.com/smiles724/Molformer)
- `2023.02` UNI-MOL: A UNIVERSAL 3D MOLECULAR REPRESENTATION LEARNING FRAMEWORK, [NeurIPS](https://chemrxiv.org/engage/chemrxiv/article-details/6402990d37e01856dc1d1581), [Code](https://github.com/dptech-corp/Uni-Mol)
- `2023.05` SELFORMER: MOLECULAR REPRESENTATION LEARNING VIA SELFIES LANGUAGE MODELS, [arXiv](http://arxiv.org/abs/2304.04662), [Code](https://github.com/HUBioDataLab/SELFormer)
- `2023.07` Molecular Descriptors Property Prediction Using Transformer-Based Approach, [IJMS](https://www.mdpi.com/1422-0067/24/15/11948)

### Interaction Prediction

- `2020.12` X-MOL: large-scale pre-training for molecular understanding and diverse molecular analysis, [bioRxiv](https://www.bioRxiv.org/content/10.1101/2020.12.23.424259v2.full), [Code](https://github.com/bm2-lab/X-MOL)

### Molecule Generation/Design/Edit

- `2021.05` MolGPT: Molecular Generation Using a Transformer-Decoder Model, [JCIM](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00600), [Code](https://github.com/devalab/molgpt)
- `2021.07` Transmol: repurposing a language model for molecular generation, [RSC Advances](https://pubs.rsc.org/en/content/articlelanding/2021/ra/d1ra03086h), [Code](https://gitlab.com/cheml.io/public/transmol)
- `2021.09` GENERATIVE PRE-TRAINING FROM MOLECULES, [ChemRxiv](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/6142f60742198e8c31782e9e/original/generative-pre-training-from-molecules.pdf), [Code](https://github.com/sanjaradylov/smiles-gpt)
- `2021.12` Generative Chemical Transformer: Neural Machine Learning of Molecular Geometric Structures from Chemical Language via Attention, [JCIM](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01289), [Code](https://github.com/Hyunseung-Kim/molGCT)
- `2022.10` A Pre-trained Conditional Transformer for Target-specific De Novo Molecular Generation, [arXiv](http://arxiv.org/abs/2210.08749)
- `2023.05` iupacGPT: IUPAC-based large-scale molecular pre-trained model for property prediction and molecule generation, [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/645f49f9a32ceeff2d90c9ae), [Code](https://github.com/AspirinCode/iupacGPT)
- `2023.05` cMolGPT: A Conditional Generative Pre-Trained Transformer for Target-Specific De Novo Molecular Generation, [Molecules](https://www.mdpi.com/1420-3049/28/11/4430), [Code](https://github.com/VV123/cMolGPT)
- `2023.05` Molecule generation using transformers and policy gradient reinforcement learning, [Scientific Reports](https://www.nature.com/articles/s41598-023-35648-w), [Code](https://github.com/eyalmazuz/MolGen)
- `2023.10` DOMAIN-AGNOSTIC MOLECULAR GENERATION WITH SELF-FEEDBACK, [arXiv](https://arxiv.org/abs/2301.11259v5), [Code](https://github.com/zjunlp/MolGen)

### Reaction Prediction

- `2019.08` Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction, [ACS Cent. Sci.](https://doi.org/10.1021/acscentsci.9b00576), [Code](https://github.com/pschwllr/MolecularTransformer)
- `2019.08` Molecular Transformer unifies reaction prediction and retrosynthesis across pharma chemical space, [Chemical Communications](http://xlink.rsc.org/?DOI=C9CC05122H)
- `2019.09` A Transformer Model for Retrosynthesis, [ICANN](http://link.springer.com/10.1007/978-3-030-30493-5_78), [Code](https://github.com/bigchem/retrosynthesis)
- `2019.12` Predicting Retrosynthetic Reaction using Self-Corrected Transformer Neural Networks, [arXiv](https://arxiv.org/abs/1907.01356), [Code](https://github.com/Jh-SYSU/SCROP)
- `2020.11` State-of-the-art augmented NLP transformer models for direct and single-step retrosynthesis, [Nature Communications](https://www.nature.com/articles/s41467-020-19266-y), [Code](https://github.com/bigchem/synthesis)
- `2021.01` Valid, Plausible, and Diverse Retrosynthesis Using Tied Two-Way Transformers with Latent Variables, [JCIM](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01074), [Code](https://github.com/ejklike/tied-twoway-transformer/)
- `2021.01` Prediction of chemical reaction yields using deep learning, [Mach. Learn.: Sci. Technol.](https://iopscience.iop.org/article/10.1088/2632-2153/abc81d), [Code](https://rxn4chemistry.github.io/rxn_yields/)
- `2021.03` Predicting Chemical Reaction Outcomes: A Grammar Ontology-based Transformer Framework, [AIChE Journal](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.17190)
- `2021.10` Molecular Graph Enhanced Transformer for Retrosynthesis Prediction, [Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231221009413), [Code](https://github.com/papercodekl/MolecularGET)
- `2021.10` PERMUTATION INVARIANT GRAPH-TO-SEQUENCE MODEL FOR TEMPLATE-FREE RETROSYNTHESIS AND REACTION PREDICTION, [arXiv](https://arxiv.org/abs/2110.09681), [Code](https://github.com/coleygroup/Graph2SMILES)
- `2022.03` Retrosynthetic reaction pathway prediction through neural machine translation of atomic environments, [Nature Communications](https://www.nature.com/articles/s41467-022-28857-w), [Code](https://github.com/knu-lcbc/RetroTRAE)
- `2023.02` Enhancing diversity in language based models for single-step retrosynthesis, [Digital Discovery](http://xlink.rsc.org/?DOI=D2DD00110A), [Code](https://github.com/rxn4chemistry/rxn_cluster_token_prompt)
- `2023.07` Unbiasing Retrosynthesis Language Models with Disconnection Prompts, [ACS Cent. Sci.](https://doi.org/10.1021/acscentsci.3c00372), [Code](https://github.com/rxn4chemistry/disconnection_aware_retrosynthesis)

### Datasets and Benchmarks

- [ZINC 15](https://ZINC15.docking.org/), `2015.10` ZINC 15 – Ligand Discovery for Everyone, [JCIM](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00559)
- [ZINC 20](https://zinc20.docking.org/), `2020.12` ZINC20—A Free Ultralarge-Scale Chemical Database for Ligand Discovery, [JCIM](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00559)
- [ZINC-250k](https://figshare.com/articles/dataset/ZINC_250K_data_sets/17122427/1), `2012.07` ZINC − A Free Database of Commercially Available Compounds for Virtual Screening, [JCIM](https://pubs.acs.org/doi/10.1021/ci049714%2B?ref=PDF)
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/), `2023.01` PubChem 2023 update, [Nucleic Acids Research](https://doi.org/10.1093/nar/gkac956)
- [USPTO](https://developer.uspto.gov/data), [USPTO MIT](https://github.com/wengong-jin/nips17-rexgen), [USPTO-15K](https://github.com/connorcoley/ochem_predict_nn), [USPTO-full](https://github.com/dan2097/patent-reaction-extraction), `2012.10` Extraction of chemical structures and reactions from the literature, [University of Cambridge](http://www.dspace.cam.ac.uk/handle/1810/244727)
- [PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/), `2021.10` OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs, [arXiv](http://arxiv.org/abs/2103.09430)
- [PCQM4M-LSC](https://ogb.stanford.edu/kddcup2021/pcqm4m/), `2021.06` First Place Solution of KDD Cup 2021 & OGB Large-Scale Challenge Graph Prediction Track, [arXiv](http://arxiv.org/abs/2106.08279)
- [GEOM](https://www.aicures.mit.edu/), `2022.04` GEOM, energy-annotated molecular conformations for property prediction and molecular generation, [Nature](https://www.nature.com/articles/s41597-022-01288-4)
- [ToyMix, LargeMix, UltraLarge](https://zenodo.org/record/8372621), `2023.10` Towards Foundational Models for Molecular Learning on Large-Scale Multi-Task Datasets, [arXiv](https://arxiv.org/pdf/2310.04292v2.pdf)
- [ChEMBL](https://www.ebi.ac.uk/chembl/), `2023.05` The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods, [Nucleic Acids Research](https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkad1004/7337608)
- [DrugBank 5.0](https://go.drugbank.com/), `2017.11` DrugBank 5.0: a major update to the DrugBank database for 2018, [Nucleic Acids Research](https://academic.oup.com/nar/article/46/D1/D1074/4602867?login=false)
- [GDB-17](https://gdb.unibe.ch/downloads/), `2012.10` Enumeration of 166 Billion Organic Small Molecules in the Chemical Universe Database GDB-17, [JCIM](https://pubs.acs.org/doi/10.1021/ci300415d)
- [ExCAPE-DB](https://pubchem.ncbi.nlm.nih.gov/), `2017.03` ExCAPE-DB: an integrated large scale dataset facilitating Big Data analysis in chemogenomics, [Journal of Cheminformatics](https://doi.org/10.1186/s13321-017-0203-5)
- [MoleculeNet](https://github.com/deepchem/deepchem), `2017.10` MoleculeNet: a benchmark for molecular machine learning, [Chemical Science](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a)
- [MARCEL](https://github.com/SXKDZ/MARCEL), `2023.09` Learning Over Molecular Conformer Ensembles: Datasets and Benchmarks, [arXiv](https://arxiv.org/abs/2310.00115v1)
- [GuacaMol](https://benevolent.ai/guacamol), `2019.03` GuacaMol: Benchmarking Models for de Novo Molecular Design, [JCIM](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839?ref=PDF)
- [MOSES](https://github.com/molecularsets/moses), `2020.12` Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models, [Frontiers in Pharmacology](https://www.frontiersin.org/articles/10.3389/fphar.2020.565644/full)
- [ADMETlab 2.0](https://admet.scbdd.com/) `2021.04` ADMETlab 2.0: an integrated online platform for accurate and comprehensive predictions of ADMET properties, [Nucleic Acids Research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8262709/)

## 🧬 Protein Large Language Models (Prot-LLMs)

### Protein Sequence Representation

- `2020.02` Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences, [PNAS](https://www.bioRxiv.org/content/10.1101/622803v4), [Code](https://github.com/facebookresearch/esm)
- `2021.02` MSA transformer, [PMLR](http://proceedings.mlr.press/v139/rao21a.html), [Code](https://github.com/facebookresearch/esm)
- `2021.02` Multi-scale representation learning on proteins, [Neurips](https://proceedings.neurips.cc/paper_files/paper/2021/hash/d494020ff8ec181ef98ed97ac3f25453-Abstract.html)
- `2021.02` Language models enable zero-shot prediction of the effects of mutations on protein function, [Neurips](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html), [Code](https://github.com/facebookresearch/esm)
- `2021.07` ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning, [IEEE Transactions on Pattern Analysis and Machine Intelligence](https://ieeexplore.ieee.org/abstract/document/9477085/), [Code](https://github.com/agemagician/ProtTrans)
- `2021.07` Pre-training Co-evolutionary Protein Representation via A Pairwise Masked Language Model, [CoRR](https://arxiv.org/abs/2110.15527)
- `2021.09` Toward more general embeddings for protein design: Harnessing joint representations of sequence and structure, [bioRxiv](https://www.bioRxiv.org/content/10.1101/2021.09.01.458592.abstract)
- `2022.02` ProteinBERT: a universal deep-learning model of protein sequence and function, [bioRxiv](https://academic.oup.com/bioinformatics/article-abstract/38/8/2102/6502274), [Code](https://github.com/nadavbra/protein_bert)
- `2022.04` Lm-gvp: an extensible sequence and structure informed deep learning framework for protein property prediction, [bioRxiv](https://www.bioRxiv.org/content/10.1101/2023.02.22.529597v1), [Code](https://github.com/aws-samples/lm-gvp)
- `2022.05` Retrieved Sequence Augmentation for Protein Representation Learning, [bioRxiv](https://www.bioRxiv.org/content/10.1101/622803v4), [Code](https://github.com/HKUNLP/RSA)
- `2022.06` OntoProtein: Protein Pretraining With Gene Ontology Embedding, [arXiv](https://ui.adsabs.harvard.edu/abs/2022arXiv220111147Z), [Code](https://github.com/zjunlp/OntoProtein)
- `2022.07` Language models of protein sequences at the scale of evolution enable accurate structure prediction, [bioRxiv](https://doi.org/10.1101/2022.07.20.500902), [Code](https://github.com/facebookresearch/esm)
- `2023.02` Multi-level Protein Structure Pre-training via Prompt Learning, [ICLR](https://openreview.net/forum?id=XGagtiJ8XC), [Code](https://github.com/HICAI-ZJU/PromptProtein)
- `2023.02` Protein Representation Learning via Knowledge Enhanced Primary Structure Modeling, [arXiv](https://ui.adsabs.harvard.edu/abs/2023arXiv230113154Z), [Code](https://github.com/RL4M/KeAP)
- `2023.10` Deciphering the protein landscape with ProtFlash, a lightweight language model, [bioRxiv](https://doi.org/10.1016/j.xcrp.2023.101600), [Code](https://github.com/ISYSLAB-HUST/ProtFlash)
- `2023.10` Enhancing protein language models with structure-based encoder and pre-training, [arXiv](https://arxiv.org/abs/2303.06275), [Code](https://github.com/DeepGraphLearning/ESM-GearNet)
- `2023.10` Saprot: Protein language modeling with structure-aware vocabulary, [bioRxiv](https://www.bioRxiv.org/content/10.1101/2023.10.01.560349v2.abstract), [Code](https://github.com/westlake-repl/SaProt)
- `2023.12` ProteinNPT: Improving Protein Property Prediction and Design with Non-Parametric Transformers, [bioRxiv](https://www.bioRxiv.org/content/early/2023/12/07/2023.12.06.570473)

### Protein Sequence Generation/Design

- `2020.03` ProGen: Language Modeling for Protein Generation, [arXiv](https://doi.org/10.48550/arXiv.2004.03497), [Code](https://github.com/salesforce/progen)
- `2021.01` A deep unsupervised language model for protein design, [bioRxiv](https://www.bioRxiv.org/content/early/2022/03/12/2022.03.09.483666), [Code](https://huggingface.co/nferruz/ProtGPT2)
- `2021.01` Fold2seq: A joint sequence (1d)-fold (3d) embedding-based generative model for protein design, [PMLR](https://proceedings.mlr.press/v139/cao21a.html), [Code](https://github.com/IBM/fold2seq)
- `2022.01` ZymCTRL: a conditional language model for the controllable generation of artificial enzymes, [NeurIPS](https://www.mlsb.io/papers_2022/ZymCTRL_a_conditional_language_model_for_the_controllable_generation_of_artificial_enzymes.pdf), [Code](https://huggingface.co/AI4PD/ZymCTRL)
- `2022.04` Few Shot Protein Generation, [arXiv](https://arxiv.org/abs/2204.01168)
- `2022.05` RITA: a Study on Scaling Up Generative Protein Sequence Models, [arXiv]([https://doi.org/10.48550/arXiv.2004.03497](https://ui.adsabs.harvard.edu/abs/2022arXiv220505789H))
- `2022.12` Generative language modeling for antibody design, [arXiv](https://www.bioRxiv.org/content/10.1101/2021.12.13.472419.abstract), [Code](https://github.com/Graylab/IgLM)
- `2023.02` Structure-informed Language Models Are Protein Designers, [bioRxiv](https://www.bioRxiv.org/content/early/2023/02/09/2023.02.03.526917)
- `2023.02` Generative power of a protein language model trained on multiple sequence alignments, [Elife](https://elifesciences.org/articles/79854), [Code](https://doi.org/10.5281/zenodo.7684052)
- `2023.02` Protein sequence design in a latent space via model-based reinforcement learning, [ICLR](https://openreview.net/forum?id=OhjGzRE5N6o)
- `2023.06` Enhancing the Protein Tertiary Structure Prediction by Multiple Sequence Alignment Generation, [arXiv](https://ui.adsabs.harvard.edu/abs/2023arXiv230601824Z), [Code](https://github.com/Magiccircuit/MSA-Augmentor)
- `2023.07` ProstT5: Bilingual Language Model for Protein Sequence and Structure, [bioRxiv](https://www.bioRxiv.org/content/early/2023/07/25/2023.07.23.550085), [Code](https://github.com/mheinzinger/ProstT5)
- `2023.07` xTrimoPGLM: unified 100B-scale pre-trained transformer for deciphering the language of protein, [bioRxiv](https://www.bioRxiv.org/content/10.1101/2023.07.05.547496.abstract)
- `2023.08` Efficient and accurate sequence generation with small-scale protein language models, [bioRxiv](https://www.bioRxiv.org/content/10.1101/2023.08.04.551626.abstract)
- `2023.10` Generative Antibody Design for Complementary Chain Pairing Sequences through Encoder-Decoder Language Model, [NeurIPS](https://openreview.net/forum?id=QrH4bhWhwY)
- `2023.10` ProGen2: exploring the boundaries of protein language models, [Cell](https://www.cell.com/cell-systems/pdf/S2405-4712(23)00272-7.pdf), [Code](https://github.com/salesforce/progen)
- `2023.11` PoET: A generative model of protein families as sequences-of-sequences, [arXiv](https://arxiv.org/abs/2306.06156)

### Datasets and Benchmarks

- [UniRef100, 90, 50](https://www.uniprot.org/uniref?query=*), `2007.03` UniRef: comprehensive and non-redundant UniProt reference clusters, [Bioinformatics](https://academic.oup.com/bioinformatics/article-abstract/23/10/1282/197795)
- [UniProtKB/Swiss-Prot](https://www.uniprot.org/uniprotkb?query=*), `2016.01` UniProtKB/Swiss-Prot, the manually annotated section of the UniProt KnowledgeBase: how to use the entry view, [Springer Plant Bioinformatics](https://link.springer.com/protocol/10.1007/978-1-4939-3167-5_2)
- [UniProtKB/TrEMBL](https://www.uniprot.org/uniprotkb?query=*), `1999.03` EDITtoTrEMBL: a distributed approach to high-quality automated protein sequence annotation, [Bioinformatics](https://link.springer.com/protocol/10.1007/978-1-4939-3167-5_2)
- [UniParc](https://www.uniprot.org/uniparc?query=*), `2022.11` UniProt: the Universal Protein Knowledgebase in 2023, [Bioinformatics](https://doi.org/10.1093/nar/gkac1052)
- [Pfam](https://www.ebi.ac.uk/interpro/entry/pfam/), `1999.03` Pfam: clans, web tools and services, [Nucleic Acids Research](https://academic.oup.com/nar/article-abstract/34/suppl_1/D247/1133922)
- [BFD](https://bfd.mmseqs.com/), `2018.06` Clustering huge protein sequence sets in linear time, [Nature Communications](https://www.nature.com/articles/s41467-018-04964-5)
- [PDB](https://www.rcsb.org/), `2018.10` Protein Data Bank: the single global archive for 3D macromolecular structure data, [Nucleic Acids Research](https://doi.org/10.1093/nar/gky949)
- [AlphaFoldDB](https://alphafold.ebi.ac.uk/), 2021.11.AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models, [Nucleic Acids Research](https://doi.org/10.1093/nar/gkab1061)
- [CASP](https://predictioncenter.org/), Critical assessment of methods of protein structure prediction (CASP)—Round XIV, [PROTEINS](https://doi.org/10.1002/prot.26237)
- [EC](https://www.enzyme-database.org/), `2008.09` ExplorEnz: the primary source of the IUBMB enzyme list, [Nucleic Acids Research](https://academic.oup.com/nar/article-abstract/37/suppl_1/D593/1000297)
- [GO](https://geneontology.org/), `2000.05` Gene ontology: tool for the unification of biology, [Nature Genetics](https://www.nature.com/articles/ng0500_25)
- [CATH](http://www.cathdb.info), `1997.08` CATH--a hierarchic classification of protein domain structures, [NIH](https://pubmed.ncbi.nlm.nih.gov/9309224/)
- [HIPPIE](http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/), `2012.02` HIPPIE: Integrating protein interaction networks with experiment based quality scores, [PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0031826)
- [SCOP](http://scop.berkeley.edu), `2000.01` SCOP: a structural classification of proteins database, [Nucleic Acids Research](https://academic.oup.com/nar/article-abstract/28/1/257/2384406)
- [ProteinGym](https://proteingym.org/), `2023.09` Proteingym: Large-scale benchmarks for protein fitness prediction and design, [NeurIPS](https://openreview.net/forum?id=URoZHqAohf)
- [FLIP](https://benchmark.protein.properties), `2022.01` FLIP: Benchmark tasks in fitness landscape inference for proteins, [bioRxiv](https://www.bioRxiv.org/content/10.1101/2021.11.09.467890v2.abstract)
- [PEER](https://github.com/DeepGraphLearning/PEER_Benchmark), `2022.09` Peer: a comprehensive and multi-task benchmark for protein sequence understanding, [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e467582d42d9c13fa9603df16f31de6d-Abstract-Datasets_and_Benchmarks.html)
- [TAPE](https://github.com/songlab-cal/tape), `2019.09` Evaluating Protein Transfer Learning with TAPE, [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2019/hash/37f65c068b7723cd7809ee2d31d7861c-Abstract.html)

## 🦠 Genomic Large Language Models (Gene-LLMs)

### General

- `2021.02` DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome [Bioinformatics](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680)
- `2023.01` Species-aware DNA language modeling [bioRxiv](https://www.bioRxiv.org/content/10.1101/2023.01.26.525670.abstract)
- `2023.01` The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics [bioRxiv](https://www.bioRxiv.org/content/10.1101/2023.01.11.523679v3.full.pdf+html)
- `2023.06` HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution [arXiv](https://arxiv.org/abs/2306.15794)
- `2023.06` DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome [arXiv](https://arxiv.org/abs/2306.15006)
- `2023.06` GENA-LM: A Family of Open-Source Foundational Models for Long DNA Sequences [bioRxiv](https://www.bioRxiv.org/content/10.1101/2023.06.12.544594v1.abstract)
- `2023.07` EpiGePT: a Pretrained Transformer model for epigenomics [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.07.15.549134v1)
- `2023.08` Understanding the Natural Language of DNA using Encoder-Decoder Foundation Models with Byte-level Precision [bioRxiv](https://arxiv.org/abs/2311.02333)
- `2023.08` DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks [bioRxiv](https://www.bioRxiv.org/content/10.1101/2023.07.11.548628v2)
- `2022.08` MoDNA: motif-oriented pre-training for DNA language model [ACM-BCB](https://dl.acm.org/doi/10.1145/3535508.3545512)

### Function Prediction

- `2021.10` Effective gene expression prediction from sequence by integrating long-range interactions [Nature Methods](https://www.nature.com/articles/s41592-021-01252-x)
- `2022.08` iEnhancer-BERT: A Novel Transfer Learning Architecture Based on DNA-Language Model for Identifying Enhancers and Their Strength [ICIC 2022](https://link.springer.com/chapter/10.1007/978-3-031-13829-4_13)
- `2022.10` iDNA-ABF: multi-scale deep biological language learning model for the interpretable prediction of DNA methylations [Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02780-1)
- `2022.12` iEnhancer-ELM: improve enhancer identification by extracting position-related multiscale contextual information based on enhancer language models [arXiv](https://arxiv.org/abs/2212.01495)
- `2023.03` miProBERT: identification of microRNA promoters based on the pre-trained model BERT [Briefings in Bioinformatics](https://academic.oup.com/bib/article-abstract/24/3/bbad093/7079709)
- `2023.07` PLPMpro: Enhancing promoter sequence prediction with prompt-learning based pre-trained language model [Computers in Biology and Medicine](https://www.sciencedirect.com/science/article/abs/pii/S0010482523007254)

### Variants and Evolution Prediction

- `2022.08` DNA language models are powerful predictors of genome-wide variant effects [bioRxiv](https://www.bioRxiv.org/content/10.1101/2022.08.22.504706v3.abstract)
- `2022.10` GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics [bioRxiv](https://www.bioRxiv.org/content/10.1101/2022.10.10.511571v2)
- `2023.10` GPN-MSA: an alignment-based DNA language model for genome-wide variant effect prediction [bioRxiv](https://www.bioRxiv.org/content/10.1101/2023.10.10.561776v1.abstract)

### DNA-Protein Interaction Prediction

- `2023.05` Improving language model of human genome for DNA–protein binding prediction based on task-specific pre-training [Interdisciplinary Sciences: Computational Life Sciences](https://link.springer.com/article/10.1007/s12539-022-00537-9)

### RNA Prediction

- `2023.02` Self-supervised learning on millions of pre-mRNA sequences improves sequence-based RNA splicing prediction [bioRxiv](https://www.bioRxiv.org/content/10.1101/2023.01.31.526427.abstract)
- `2023.03` Multiple sequence-alignment-based RNA language model and its application to structural inference [bioRxiv](https://www.bioRxiv.org/content/10.1101/2023.03.15.532863v1.abstract)
- `2023.06` Prediction of Multiple Types of RNA Modifications via Biological Language Model [IEEE/ACM Transactions on Computational Biology and Bioinformatics](https://ieeexplore.ieee.org/abstract/document/10146457/)
- `2023.07` Uni-RNA: Universal Pre-trained Models Revolutionize RNA Research [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.07.11.548588v1)

### Datasets and Benchmarks

- `1999.10` The Mammalian Gene Collection [Source](https://pubmed.ncbi.nlm.nih.gov/10521335/) [Science](https://www.science.org/doi/abs/10.1126/science.286.5439.455)
- `2013.12` GRCh38 [Source](https://pubmed.ncbi.nlm.nih.gov/10521335/) [Genomics](https://www.sciencedirect.com/science/article/pii/S0888754317300058)
- `2016.06` 690 ChIP-seq [Bioinformatics](https://academic.oup.com/bioinformatics/article-abstract/32/12/i121/2240609)
- `2017.04` DeepSEA [Source](http://deepsea.princeton.edu/job/analysis/create/) [Nature Methods](https://www.nature.com/articles/nmeth.3547)
- `2017.10` 1000 Genomes Project [Source](https://www.internationalgenome.org) [Nature](https://www.nature.com/articles/nature15393)
- `2019.11` EPDnew [Source](https://epd.expasy.org/epd) [Nucleic Acids Research](https://academic.oup.com/nar/article-abstract/41/D1/D157/1070274)
- `2020.03` Panglao Dataset [Source](https://panglaodb.se) [Database](https://academic.oup.com/database/article-abstract/doi/10.1093/database/baz046/5427041)
- `2020.12` ExPecto [Source](https://hb.flatironinstitute.org/expecto/?tabId=1) [Nature Methods](https://www.nature.com/articles/s41592-018-0087-y)
- `2022.11` UCSC Genome Database [Source](https://genome.ucsc.edu) [Nucleic Acids Research](https://academic.oup.com/nar/article/31/1/51/2401563)
- `2023.01` BV-BRC [Source](https://www.bv-brc.org) [Nucleic Acids Research](https://academic.oup.com/nar/article-abstract/51/D1/D678/6814465)
- `2023.02` Ensembl [Source](https://useast.ensembl.org/index.html) [Nucleic Acids Research](https://academic.oup.com/nar/article-abstract/40/D1/D1202/2903058)
- `2023.07` RNAcmap [Bioinformatics](https://academic.oup.com/bioinformatics/article-abstract/37/20/3494/6281070)
- `2023.09` ENCODE [Source](https://www.encodeproject.org) [Nature](https://www.nature.com/articles/nature11247)
- `2023.10` NCBI Genome Database [Source](https://www.ncbi.nlm.nih.gov/genome/)
- `2023.12` TAIR [Source](https://www.arabidopsis.org) [Nucleic Acids Research](https://academic.oup.com/nar/article-abstract/40/D1/D1202/2903058)
- `2023.12` VGDB [Source](https://www.ncbi.nlm.nih.gov/genome/viruses/) [Bioinformatics](https://academic.oup.com/bioinformatics/article-abstract/16/5/484/192501)
- `2023.07` CAGI5 [Source](http://www.genomeinterpretation.org/cagi5-challenge.html) [Human Mutation](https://onlinelibrary.wiley.com/doi/abs/10.1002/humu.23873)
- `2023.08` Protein–RNA Interaction Prediction [Briefings in Bioinformatics](https://academic.oup.com/bib/article-abstract/24/5/bbad307/7252289)
- `2023.09` The Nucleaotide Transformer Benchmark [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.01.11.523679.abstract)

## Ⓜ️ Multimodal Scientific Large Language Models (MM-Sci-LLMs)

### Molecule&text

- `2021.11` Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries, [EMNLP](https://aclanthology.org/2021.emnlp-main.47/), [Code](https://github.com/cnedwards/text2mol)
- `2022.02` KV-PLM: A deep-learning system bridging molecule structure and biomedical text with comprehension comparable to human professionals, [Nature](https://www.nature.com/articles/s41467-022-28494-3), [Code](https://github.com/thunlp/KV-PLM)
- `2022.09` MoMu: A Molecular Multimodal Foundation Model Associating Molecule Graphs with Natural Language, [arXiv](https://arxiv.org/abs/2209.05481), [Code](https://github.com/BingSu12/MoMu)
- `2022.11` MolT5: Translation between Molecules and Natural Language, [arXiv](https://arxiv.org/abs/2204.11817), [Code](https://github.com/blender-nlp/MolT5)
- `2023.05` Text+Chem T5: Unifying Molecular and Textual Representations via Multi-task Language Modelling, [arXiv](https://arxiv.org/abs/2301.12586), [Code](https://github.com/GT4SD/multitask_text_and_chemistry_t5)
- `2023.05` DrugChat: Towards Enabling ChatGPT-Like Capabilities on Drug Molecule Graphs, [techRxiv](https://www.techrxiv.org/articles/preprint/DrugChat_Towards_Enabling_ChatGPT-Like_Capabilities_on_Drug_Molecule_Graphs/22945922), [Code](https://github.com/UCSD-AI4H/drugchat)
- `2023.06` GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning, [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.05.30.542904v2), [Code](https://github.com/zhao-ht/GIMLET)
- `2023.06` MolReGPT: Empowering Molecule Discovery for Molecule-Caption Translation with Large Language Models: A ChatGPT Perspective, [arXiv](https://arxiv.org/abs/2306.06615), [Code](https://github.com/phenixace/MolReGPT)
- `2023.06` ChatMol: Interactive Molecular Discovery with Natural Language, [arXiv](https://arxiv.org/abs/2306.11976), [Code](https://github.com/Ellenzzn/ChatMol/tree/main)
- `2023.07` MolXPT: Wrapping Molecules with Text for Generative Pre-training, [ACL](https://aclanthology.org/2023.acl-short.138/)
- `2023.07` MolFM: A Multimodal Molecular Foundation Model, [arXiv](https://arxiv.org/abs/2307.09484), [Code](https://github.com/PharMolix/OpenBioMed)
- `2023.08` GIT-Mol: A Multi-modal Large Language Model for Molecular Science with Graph, Image, and Text, [arXiv](https://arxiv.org/abs/2308.06911)
- `2023.10` GPT-MolBERTa: GPT Molecular Features Language Model for molecular property prediction, [arXiv](https://arxiv.org/abs/2310.03030), [Code](https://github.com/Suryanarayanan-Balaji/GPT-MolBERTa)
- `2023.12` MoleculeSTM: Multi-modal Molecule Structure-text Model for Text-based Retrieval and Editing, [arXiv](https://arxiv.org/abs/2212.10789), [Code](https://github.com/chao1224/MoleculeSTM/tree/main)

### Protein&text

- `2022.04` ProTranslator: zero-shot protein function prediction using textual description, [arXiv](https://arxiv.org/abs/2204.10286), [Code](https://github.com/HanwenXuTHU/ProTranslator)
- `2023.02` ProteinDT: A Text-guided Protein Design Framework, [arXiv](https://arxiv.org/abs/2302.04611)
- `2023.07` ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts, [arXiv](https://arxiv.org/abs/2301.12040), [Code](https://github.com/DeepGraphLearning/ProtST)
- `2023.07` Prot2Text: Multimodal Protein's Function Generation with GNNs and Transformers, [arXiv](https://arxiv.org/abs/2307.14367)
- `2023.10` InstructProtein: Aligning Human and Protein Language via Knowledge Instruction, [arXiv](https://arxiv.org/abs/2310.03269)

### Protein&molecule

- `2022.09` ChemBERTaLM: Exploiting pretrained biochemical language models for targeted drug design, [Bioinformatics](https://academic.oup.com/bioinformatics/article/38/Supplement_2/ii155/6702010), [Code](https://github.com/boun-tabi/biochemical-lms-for-drug-design)
- `2023.03` Deep generative model for drug design from protein target sequence, [Journal of Cheminformatics ](https://link.springer.com/article/10.1186/s13321-023-00702-2), [Code](https://github.com/viko-3/TargetGAN)
- `2023.06` DrugGPT: A GPT-based Strategy for Designing Potential Ligands Targeting Specific Proteins, [bioRxiv](https://www.bioRxiv.org/content/10.1101/2023.06.29.543848v1), [Code](https://github.com/LIYUESEN/druggpt)
- `2023.10` DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening, [arXiv](https://arxiv.org/abs/2310.06367)

### Comprehensive

- `2022.11` Galactica: A Large Language Model for Science, [arXiv](http://arxiv.org/abs/2211.09085), [Code](https://galactica.org/mission/)
- `2023.02` BioTranslator: Multilingual translation for zero-shot biomedical classification using BioTranslator, [Nature](https://www.nature.com/articles/s41467-023-36476-2), [Code](https://github.com/HanwenXuTHU/BioTranslatorProject)
- `2023.05` ChatDrug: ChatGPT-powered Conversational Drug Editing Using Retrieval and Domain Feedback, [arXiv](https://arxiv.org/abs/2305.18090), [Code](https://github.com/chao1224/ChatDrug)
- `2023.08` BioMedGPT：A Pre-trained Language Model for Biomedical Text Mining, [arXiv](https://arxiv.org/abs/2308.09442v2), [Code](https://github.com/PharMolix/OpenBioMed)
- `2023.08` DARWIN Series: Domain Specific Large Language Models for Natural Science, [arXiv](https://arxiv.org/abs/2308.13565), [Code](https://github.com/MasterAI-EAM/Darwin)
- `2023.10` BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations, [arXiv](https://arxiv.org/abs/2310.07276), [Code](https://github.com/QizhiPei/BioT5)
- `2023.11` Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models, [arXiv](https://arxiv.org/abs/2306.08018), [Code](https://github.com/zjunlp/Mol-Instructions)
- `2024.01` BioBridge: Bridging Biomedical Foundation Models via Knowledge Graphs, [arXiv](https://arxiv.org/abs/2310.03320), [Code](https://github.com/RyanWangZf/BioBridge)

### Datasets and Benchmarks

#### Molecule&Text

- [ChEBI-20](https://github.com/cnedwards/text2mol), `2021.11` Text2mol: Cross-modal molecule retrieval with natural language queries, [EMNLP2021](https://aclanthology.org/2021.emnlp-main.47/)
- [PCdes](https://github.com/thunlp/KV-PLM), `2022.02` A deep-learning system bridging molecule structure and biomedical text with comprehension comparable to human professionals, [Nature](https://www.nature.com/articles/s41467-022-28494-3)
- [MoMu](https://github.com/BingSu12/MoMu/tree/main/data), `2022.09` A Molecular Multimodal Foundation Model Associating Molecule Graphs with Natural Language, [arXiv](https://arxiv.org/abs/2209.05481)
- [PubChemSTM](https://github.com/chao1224/MoleculeSTM/tree/main/MoleculeSTM/datasets), `2022.12`. Multi-modal Molecule Structure-text Model for Text-based Retrieval and Editing, [arXiv](https://arxiv.org/abs/2212.10789)
- [ChEBL-dia](https://github.com/Ellenzzn/ChatMol/tree/main), `2023.06` ChatMol: Interactive Molecular Discovery with Natural Language, [arXiv](https://arxiv.org/abs/2306.11976)
- [PubChemQA](https://github.com/PharMolix/OpenBioMed), `2023.08` BioMedGPT：A Pre-trained Language Model for Biomedical Text Mining, [arXiv](https://arxiv.org/abs/2308.09442v2)

#### Protein&Text

- SwissProtCLAP, `2023.02` ProteinDT: A Text-guided Protein Design Framework, [arXiv](https://arxiv.org/abs/2302.04611)
- [ProtDescribe](https://github.com/DeepGraphLearning/ProtST), `2023.07` ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts, [arXiv](https://arxiv.org/abs/2301.12040)
- Prot2Text, `2023.07` Prot2Text: Multimodal Protein's Function Generation with GNNs and Transformers, [arXiv](https://arxiv.org/abs/2307.14367)
- [UniProtQA](https://github.com/PharMolix/OpenBioMed), `2023.08` BioMedGPT：A Pre-trained Language Model for Biomedical Text Mining, [arXiv](https://arxiv.org/abs/2308.09442v2)
- InstructProtein, `2023.10` InstructProtein: Aligning Human and Protein Language via Knowledge Instruction, [arXiv](https://arxiv.org/abs/2310.03269)

#### Protein&Molecule

- [DUD-E](https://dude.docking.org/), `2012.06` Directory of Useful Decoys, Enhanced (DUD-E): Better Ligands and Decoys for Better Benchmarking, [Journal of Medicinal Chemistry](https://pubs.acs.org/doi/10.1021/jm300687e)
- [BioLiP](https://zhanggroup.org/BioLiP/index.cgi), `2012.10` BioLiP: a semi-manually curated database for biologically relevant ligand–protein interactions, [Nucleic Acids Research](https://academic.oup.com/nar/article/41/D1/D1096/1074898)
- [BindingDB](https://www.bindingdb.org/rwd/bind/index.jsp), `2016.01` BindingDB in 2015: A public database for medicinal chemistry, computational chemistry and systems pharmacology, [Nucleic Acids Research](https://academic.oup.com/nar/article/44/D1/D1045/2502601)

#### Comprehensive

- Galactica, `2022.11` Galactica: A Large Language Model for Science, [arXiv](https://arxiv.org/abs/2211.09085)
- [Scientific Knowledge Dataset](https://github.com/MasterAI-EAM/Darwin), `2023.08` DARWIN Series: Domain Specific Large Language Models for Natural Science, [arXiv](https://arxiv.org/abs/2308.13565)
- [Mol-Instructions](https://github.com/zjunlp/Mol-Instructions), `2023.10` Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models, [arXiv](https://arxiv.org/abs/2306.08018)

## 👥 Contributions

### Citation

If you find this repository useful, please cite our paper:

```
@misc{zhang2024scientific,
      title={Scientific Large Language Models: A Survey on Biological & Chemical Domains}, 
      author={Qiang Zhang and Keyan Ding and Tianwen Lyv and Xinda Wang and Qingyu Yin and Yiwen Zhang and Jing Yu and Yuhao Wang and Xiaotong Li and Zhuoyi Xiang and Xiang Zhuang and Zeyuan Wang and Ming Qin and Mengyao Zhang and Jinlu Zhang and Jiyu Cui and Renjun Xu and Hongyang Chen and Xiaohui Fan and Huabin Xing and Huajun Chen},
      year={2024},
      eprint={2401.14656},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Contributors

- Keyan Ding [@dingkeyan93](https://github.com/dingkeyan93)
- Jing Yu [@jiing17](https://github.com/jiing17)
- Tianwen Lyu [@smiling-k](https://github.com/smiling-k)
- Yiwen Zhang [@zhangyiwen2002](https://github.com/zhangyiwen2002)
- Xinda Wang [@Wwwduojin](https://github.com/Wwwduojin)
- Qingyu Yin [@MikaStars39](https://github.com/MikaStars39)

### Contact

- Xinda Wang [22351323@zju.edu.cn](mailto:22351323@zju.edu.cn)

![Star History Chart](https://api.star-history.com/svg?repos=HICAI-ZJU/Scientific-LLM-Survey&type=Date)
