# ATPapers
Worth-reading papers and related resources on attention mechanisim, Transformer and pretrained language model (PLM) such as BERT.

Suggestions about fixing errors or adding papers, repositories and other resources are welcomed!

值得一读的注意力机制、Transformer和预训练语言模型论文与相关资源集合。

欢迎修正错误以及新增论文、代码仓库与其他资源等建议！

## Attention
### Papers
- Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (ICML 2015) [[paper]](http://proceedings.mlr.press/v37/xuc15.html) - ***Hard & Soft Attention***
- Effective Approaches to Attention-based Neural Machine Translation (EMNLP 2015) [[paper]](https://www.aclweb.org/anthology/D15-1166/) - ***Global & Local Attention***
- Neural Machine Translation by Jointly Learning to Align and Translate (ICLR 2015) [[paper]](https://arxiv.org/abs/1409.0473)
- Why Self-Attention? A Targeted Evaluation of Neural Machine Translation Architectures (EMNLP 2018) [[paper]](https://www.aclweb.org/anthology/D18-1458/)
- Phrase-level Self-Attention Networks for Universal Sentence Encoding (EMNLP 2018) [[paper]](https://www.aclweb.org/anthology/D18-1408/)
- Bi-Directional Block Self-Attention for Fast and Memory-Efficient Sequence Modeling (ICLR 2018) [[paper]](https://arxiv.org/abs/1804.00857)[[code]](https://github.com/taoshen58/BiBloSA) - ***Bi-BloSAN***
- Leveraging Local and Global Patterns for Self-Attention Networks (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1295/) [[tf code]](https://github.com/scewiner/Leveraging)[[pt code]](https://github.com/galsang/BiBloSA-pytorch)
- Attention over Heads: A Multi-Hop Attention for Neural Machine Translation (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-2030/)
- Are Sixteen Heads Really Better than One? (NeurIPS 2019) [[paper]](https://arxiv.org/abs/1905.10650)
- Synthesizer: Rethinking Self-Attention in Transformer Models (CoRR 2020) [[paper]](https://arxiv.org/abs/2005.00743) - ***Synthesizer***
### Survey & Review
- An Attentive Survey of Attention Models (IJCAI 2019) [[paper]](https://arxiv.org/abs/1904.02874)
### English Blog
- [Illustrated: Self-Attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)
### Chinese Blog
- [深度学习中的注意力模型（2017版）](https://zhuanlan.zhihu.com/p/37601161)
- [科学空间 / Google新作Synthesizer：我们还不够了解自注意力](https://kexue.fm/archives/7430)
### Repositories
- [Keras Attention Layer](https://github.com/thushv89/attention_keras) - Attention layer Keras implementation

## Transformer
### Papers
- Attention is All you Need (NIPS 2017) [[paper]](http://papers.nips.cc/paper/7181-attention-is-all-you-need)[[code]](https://github.com/jadore801120/attention-is-all-you-need-pytorch) - ***Transformer***
- Weighted Transformer Network for Machine Translation (CoRR 2017) [[paper]](https://arxiv.org/abs/1711.02132)[[code]](https://github.com/JayParks/transformer)
- Accelerating Neural Transformer via an Average Attention Network (ACL 2018) [[paper]](https://www.aclweb.org/anthology/P18-1166/)[[code]](https://github.com/bzhangGo/transformer-aan) - ***AAN***
- Self-Attention with Relative Position Representations (NAACL 2018) [[paper]](https://www.aclweb.org/anthology/N18-2074/) [[unoffical code]](https://github.com/THUNLP-MT/THUMT/blob/d4cb62c215d846093e5357aa17b286506b2df1af/thumt/layers/attention.py)
- Universal Transformers (ICLR 2019) [[paper]](https://openreview.net/forum?id=HyzdRiR9Y7)[[code]](https://github.com/andreamad8/Universal-Transformer-Pytorch) - ***Universal Transformer***
- Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1285) - ***Transformer-XL***
- Memory Transformer Networks (CS224n Winter2019 Reports) [[paper]](https://web.stanford.edu/class/cs224n/reports/custom/15778933.pdf)
- Star-Transformer (NAACL 2019) [[paper]](https://arxiv.org/pdf/1902.09113.pdf)
- On Layer Normalization in the Transformer Architecture (ICLR 2020) [[paper]](https://openreview.net/pdf?id=B1x8anVFPr)
- Reformer: The Efficient Transformer (ICLR 2020) [[paper]](https://openreview.net/forum?id=rkgNKkHtvB) [[code 1]](https://pastebin.com/62r5FuEW)[[code 2]](https://github.com/google/trax/tree/master/trax/models/reformer)[[code 3]](https://github.com/lucidrains/reformer-pytorch) - ***Reformer***
- TENER: Adapting Transformer Encoder for Named Entity Recognition (CoRR 2019) [[paper]](https://arxiv.org/abs/1911.04474)
- ReZero is All You Need: Fast Convergence at Large Depth (CoRR 2020) [[paper]](https://arxiv.org/abs/2003.04887) [[code]](https://github.com/majumderb/rezero) [[related Chinese post]](https://zhuanlan.zhihu.com/p/113384612) - ***ReZero***
### Chinese Blog
- [放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)
- [浅谈 Transformer-based 模型中的位置表示](https://zhuanlan.zhihu.com/p/92017824)
- [Transformers Assemble（PART I）](https://zhuanlan.zhihu.com/p/104935987)
### English Blog
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) [[code]](https://github.com/harvardnlp/annotated-transformer)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [How Self-Attention with Relative Position Representations works](https://medium.com/@_init_/how-self-attention-with-relative-position-representations-works-28173b8c245a)
- [Transformer-XL: Unleashing the Potential of Attention Models](https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html)
- [Moving Beyond Translation with the Universal Transformer](https://ai.googleblog.com/2018/08/moving-beyond-translation-with.html)
### Repositories
- [DongjunLee / transformer-tensorflow](https://github.com/DongjunLee/transformer-tensorflow) - Transformer Tensorflow implementation
- [andreamad8 / Universal-Transformer-Pytorch](https://github.com/andreamad8/Universal-Transformer-Pytorch) - Universal Transformer PyTorch implementation
- [sannykim / transformers](https://github.com/sannykim/transformers) - A collection of resources to study Transformers in depth

## Pretrained Language Model
### Models
- Deep Contextualized Word Representations (NAACL 2018) [[paper]](https://aclweb.org/anthology/N18-1202) - ***ELMo***
- Universal Language Model Fine-tuning for Text Classification (ACL 2018) [[paper]](https://www.aclweb.org/anthology/P18-1031/) - ***ULMFit***
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (NAACL 2019) [[paper]](https://www.aclweb.org/anthology/N19-1423)[[code]](https://github.com/google-research/bert)[[official PyTorch code]](https://github.com/codertimo/BERT-pytorch) - ***BERT***
- Improving Language Understanding by Generative Pre-Training (CoRR 2018) [[paper]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) - ***GPT***
- Language Models are Unsupervised Multitask Learners (CoRR 2019) [[paper]](https://www.techbooky.com/wp-content/uploads/2019/02/Better-Language-Models-and-Their-Implications.pdf)[[code]](https://github.com/openai/gpt-2) - ***GPT-2***
- MASS: Masked Sequence to Sequence Pre-training for Language Generation (ICML 2019) [[paper]](http://proceedings.mlr.press/v97/song19d/song19d.pdf)[[code]](https://github.com/microsoft/MASS) - ***MASS***
- Unified Language Model Pre-training for Natural Language Understanding and Generation (CoRR 2019) [[paper]](https://arxiv.org/pdf/1905.03197.pdf)[[code]](https://github.com/microsoft/unilm) - ***UNILM*** 
- Cross-lingual Language Model Pretraining (CoRR 2019) [[paper]](https://arxiv.org/pdf/1901.07291.pdf) - ***XLM***
- Multi-Task Deep Neural Networks for Natural Language Understanding (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1441)[[code]](https://github.com/namisan/mt-dnn) - ***MT-DNN***
- 75 Languages, 1 Model: Parsing Universal Dependencies Universally[[paper]](https://www.aclweb.org/anthology/D19-1279/)[[code]](https://github.com/hyperparticle/udify) - ***UDify***
- ERNIE: Enhanced Language Representation with Informative Entities (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1139)[[code]](https://github.com/thunlp/ERNIE) - ***ERNIE (THU)***
- ERNIE: Enhanced Representation through Knowledge Integration (CoRR 2019) [[paper]](https://arxiv.org/pdf/1904.09223.pdf) - ***ERNIE (Baidu)***
- Defending Against Neural Fake News (CoRR 2019) [[paper]](https://arxiv.org/abs/1905.12616)[[code]](https://rowanzellers.com/grover/) - ***Grover***
- ERNIE 2.0: A Continual Pre-training Framework for Language Understanding (CoRR 2019) [[paper]](https://arxiv.org/pdf/1907.12412.pdf) - ***ERNIE 2.0 (Baidu)***
- Pre-Training with Whole Word Masking for Chinese BERT (CoRR 2019) [[paper]](https://arxiv.org/pdf/1906.08101.pdf) - ***Chinese-BERT-wwm***
- SpanBERT: Improving Pre-training by Representing and Predicting Spans (CoRR 2019) [[paper]](https://arxiv.org/pdf/1907.10529.pdf) - ***SpanBERT***
- XLNet: Generalized Autoregressive Pretraining for Language Understanding  (CoRR 2019) [[paper]](https://arxiv.org/pdf/1906.08237.pdf)[[code]](https://github.com/zihangdai/xlnet) - ***XLNet***
- RoBERTa: A Robustly Optimized BERT Pretraining Approach (CoRR 2019) [[paper]](https://arxiv.org/pdf/1907.11692.pdf) - ***RoBERTa***
- NEZHA: Neural Contextualized Representation for Chinese Language Understanding (CoRR 2019) [[paper]](https://arxiv.org/abs/1909.00204)[[code]](https://github.com/huawei-noah/Pretrained-Language-Model) - ***NEZHA***
- K-BERT: Enabling Language Representation with Knowledge Graph (AAAI 2020) [[paper]](https://arxiv.org/abs/1909.07606)[[code]](https://github.com/autoliuweijie/K-BERT) - ***K-BERT***
- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (CoRR 2019) [[paper]](https://arxiv.org/abs/1909.08053)[[code]](https://github.com/NVIDIA/Megatron-LM) - ***Megatron-LM***
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transforme (CoRR 2019) [[paper]](https://arxiv.org/abs/1910.10683)[[code]](https://github.com/google-research/text-to-text-transfer-transformer) - ***T5***
- BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (CoRR 2019) [[paper]](https://arxiv.org/abs/1910.13461) - ***BART***
- ZEN: Pre-training Chinese Text Encoder Enhanced by N-gram Representations (CoRR 2019) [[paper]](https://arxiv.org/abs/1911.00720)[[code]](https://github.com/sinovation/zen) - ***ZEN***
- The JDDC Corpus: A Large-Scale Multi-Turn Chinese Dialogue Dataset for E-commerce Customer Service (CoRR 2019) [[paper]](https://arxiv.org/pdf/1911.09969.pdf)[[code]](https://github.com/jd-aig/nlp_baai) - ***BAAI-JDAI-BERT***
- MultiFiT: Efficient Multi-lingual Language Model Fine-tuning (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1572/)[[code]](https://github.com/n-waves/multifit) - ***MultiFit***
- Knowledge Enhanced Contextual Word Representations (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1005/) - ***KnowBert***
- UER: An Open-Source Toolkit for Pre-training Models (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-3041/)[[code]](https://github.com/dbiir/UER-py) - ***UER***
- ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (ICLR 2020) [[paper]](https://openreview.net/forum?id=r1xMH1BtvB) - ***ELECTRA***
- StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding (ICLR 2020) [[paper]](https://arxiv.org/abs/1908.04577) - ***StructBERT***
- FreeLB: Enhanced Adversarial Training for Language Understanding (ICLR 2020) [[paper]](https://arxiv.org/abs/1909.11764)[[code]](https://github.com/zhuchen03/FreeLB) - ***FreeLB***
- HUBERT Untangles BERT to Improve Transfer across NLP Tasks (CoRR 2019) [[paper]](https://arxiv.org/pdf/1910.12647.pdf) - ***HUBERT***
- CodeBERT: A Pre-Trained Model for Programming and Natural Languages (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.08155) - ***CodeBERT***
- ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training [[paper]](https://arxiv.org/abs/2001.04063) - ***ProphetNet***


### Vision
- VideoBERT: A Joint Model for Video and Language Representation Learning (ICCV 2019) [[paper]](https://arxiv.org/abs/1904.01766)
- Learning Video Representations using Contrastive Bidirectional Transformer (CoRR 2019) [[paper]](https://arxiv.org/abs/1906.05743) - ***CBT***
- ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks (NeurIPS 2019) [[paper]](https://arxiv.org/abs/1908.02265)[[code]](https://github.com/jiasenlu/vilbert_beta)
- VisualBERT: A Simple and Performant Baseline for Vision and Language (CoRR 2019) [[paper]](https://arxiv.org/abs/1908.03557)[[code]](https://github.com/uclanlp/visualbert)
- Fusion of Detected Objects in Text for Visual Question Answering (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1219/)[[code]](https://github.com/google-research/
language/tree/master/language/question_answering/b2t2) - ***B2T2***
- Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training (AAAI 2020) [[paper]](https://arxiv.org/abs/1908.06066)
- LXMERT: Learning Cross-Modality Encoder Representations from Transformers (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1514/)[[code]](https://github.com/airsplay/lxmert)
- VL-BERT: Pre-training of Generic Visual-Linguistic Representatio (CoRR 2019) [[paper]](https://arxiv.org/abs/1908.08530)[[code]](https://github.com/jackroos/VL-BERT)
- UNITER: Learning UNiversal Image-TExt Representations (CoRR 2019) [[paper]](https://arxiv.org/abs/1909.11740)

### Compression
- Distilling Task-Specific Knowledge from BERT into Simple Neural Networks (CoRR 2019) [[paper]](https://arxiv.org/abs/1903.12136)
- Model Compression with Multi-Task Knowledge Distillation for Web-scale Question Answering System (CoRR 2019) [[paper]](https://arxiv.org/abs/1904.09636) - ***MKDM***
- Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding (CoRR 2019) [[paper]](https://arxiv.org/abs/1904.09482)
- Well-Read Students Learn Better: On the Importance of Pre-training Compact Models (CoRR 2019) [[paper]](https://arxiv.org/abs/1908.08962)
- Small and Practical BERT Models for Sequence Labeling (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1374/)
- Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT (CoRR 2019) [[paper]](https://arxiv.org/abs/1909.05840) - ***Q-BERT***
- Patient Knowledge Distillation for BERT Model Compression (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1441/) - ***BERT-PKD***
- Extreme Language Model Compression with Optimal Subwords and Shared Projections (ICLR 2019) [[paper]](https://arxiv.org/abs/1909.11687)
- DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter [[paper]](https://arxiv.org/pdf/1910.01108.pdf)[[code]](https://github.com/huggingface/transformers/tree/master/examples/distillation) - ***DistilBERT***
- TinyBERT: Distilling BERT for Natural Language Understanding (ICLR 2019) [[paper]](https://arxiv.org/pdf/1909.10351.pdf)[[code]](https://github.com/huawei-noah/Pretrained-Language-Model) - ***TinyBERT***
- Q8BERT: Quantized 8Bit BERT (NeurIPS 2019 Workshop) [[paper]](https://arxiv.org/abs/1910.06188) - ***Q8BERT***
- ALBERT: A Lite BERT for Self-supervised Learning of Language Representations (ICLR 2020) [[paper]](https://arxiv.org/abs/1909.11942)[[code]](https://github.com/google-research/ALBERT) - ***ALBERT***
- Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning (ICLR 2020) [[paper]](https://openreview.net/forum?id=SJlPOCEKvH)[[PyTorch code]](https://github.com/lonePatient/albert_pytorch)
- Reducing Transformer Depth on Demand with Structured Dropout (ICLR 2020) [[paper]](https://arxiv.org/abs/1909.11556) - ***Structured Dropout***
- Multilingual Alignment of Contextual Word Representations (ICLR 202) [[paper]](https://arxiv.org/abs/2002.03518)
- AdaBERT: Task-Adaptive BERT Compression with Differentiable Neural Architecture Search (CoRR 2020) [[paper]](https://arxiv.org/pdf/2001.04246.pdf) - ***AdaBERT***

### Application
- BERT for Joint Intent Classification and Slot Filling (CoRR 2019) [[paper]](https://arxiv.org/abs/1902.10909)
- GPT-based Generation for Classical Chinese Poetry (CoRR 2019) [[paper]](https://arxiv.org/abs/1907.00151)
- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (EMNLP 2019) [[paper]](https://arxiv.org/abs/1908.10084)[[code]](https://github.com/UKPLab/sentence-transformers)
- Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring (ICLR 2020) [[paper]](https://arxiv.org/abs/1905.01969)
- Pre-training Tasks for Embedding-based Large-scale Retrieval (ICLR 2020) [[paper]](https://arxiv.org/abs/2002.03932)
- K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.01808) - ***K-Adapter***
- Keyword-Attentive Deep Semantic Matching (CoRR 2020) [[paper & code]](https://github.com/DataTerminatorX/Keyword-BERT) [[post]](https://zhuanlan.zhihu.com/p/112562420) - ***Keyword BERT***
- Unified Multi-Criteria Chinese Word Segmentation with BERT (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.05808)

### Analysis & Tools
- Probing Neural Network Comprehension of Natural Language Arguments (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1459.pdf)[[code]](https://github.com/IKMLab/arct2)
- Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1334/) [[code]](https://github.com/tommccoy1/hans)
- To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks (RepL4NLP@ACL 2019) [[paper]](https://www.aclweb.org/anthology/W19-4302/)
- Multi-Head Multi-Layer Attention to Deep Language Representations for Grammatical Error Detection (CICLing 2019) [[paper]](https://arxiv.org/abs/1904.07334)
- Understanding the Behaviors of BERT in Ranking (CoRR 2019) [[paper]](https://arxiv.org/abs/1904.07531)
- How to Fine-Tune BERT for Text Classification? (CoRR 2019) [[paper]](https://arxiv.org/abs/1905.05583)
- What Does BERT Look At? An Analysis of BERT's Attention (BlackBoxNLP 2019) [[paper]](https://arxiv.org/abs/1906.04341)[[code]](https://github.com/clarkkev/attention-analysis)
- Visualizing and Understanding the Effectiveness of BERT (EMNLP 2019) [[paper]](https://arxiv.org/pdf/1908.05620.pdf)
- exBERT: A Visual Analysis Tool to Explore Learned Representations in Transformers Models (CoRR 2019) [[paper]](https://arxiv.org/abs/1910.05276) [[code]](exbert.net)
- Transformers: State-of-the-art Natural Language Processing [[paper]](https://arxiv.org/pdf/1910.03771.pdf)[[code]](https://github.com/huggingface/transformers)[[code]](https://github.com/huggingface/transformers)
- Do Attention Heads in BERT Track Syntactic Dependencies? [[paper]](https://arxiv.org/abs/1911.12246)
- Fine-tune BERT with Sparse Self-Attention Mechanism (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1361/)
- How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1006/)
- oLMpics -- On what Language Model Pre-training Captures (CoRR 2019) [[paper]](https://arxiv.org/abs/1912.13283)
- Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment (AAAI 2020) [[paper]](https://arxiv.org/abs/1907.11932)[[code]](https://github.com/jind11/TextFooler) - ***TextFooler***
- A Mutual Information Maximization Perspective of Language Representation Learning (ICLR 2020) [[paper]](https://arxiv.org/abs/1910.08350)
- Cross-Lingual Ability of Multilingual BERT: An Empirical Study (ICLR2020) [[paper]](https://arxiv.org/abs/1912.07840)
- Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.06305)
- How Much Knowledge Can You Pack Into the Parameters of a Language Model? (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.08910)
- BERT Can See Out of the Box: On the Cross-modal Transferability of Text Representations (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.10832)

### Tutorial & Survey
- Transfer Learning in Natural Language Processing (NAACL 2019) [[paper]](https://www.aclweb.org/anthology/N19-5004/)
- Evolution of Transfer Learning in Natural Language Processing (CoRR 2019) [[paper]](https://arxiv.org/abs/1910.07370)
- Transferring NLP Models Across Languages and Domains (DeepLo 2019) [[paper]](https://www.dropbox.com/s/myle46vl64nasg8/Deeplo-talk-2019.pdf?dl=0)
- Pre-trained Models for Natural Language Processing: A Survey (Invited Review of Science China Technological Sciences 2020) [[paper]](https://arxiv.org/pdf/2003.08271.pdf) - ***
-Embeddings in Natural Language Processing (2020) [[book]](http://josecamachocollados.com/book_embNLP_draft.pdf)

### Repository
- [keras-bert](https://github.com/CyberZHG/keras-bert) - CyberZHG's BERT Keras implementation
- [BERT-keras](https://github.com/Separius/BERT-keras) - Separius' BERT Keras implementation
- [bert4keras](https://github.com/bojone/bert4keras) - bojone's (苏神) BERT Keras implementation
- [gpt-2-Pytorch: Simple Text-Generator with OpenAI gpt-2 Pytorch Implementation](https://github.com/graykode/gpt-2-Pytorch)
- [GPT2-Chinese: Chinese version of GPT2 training code, using BERT tokenizer](https://github.com/Morizeyao/GPT2-Chinese)
- [OpenClap: Open Chinese Language Pre-trained Model Zoo](https://github.com/thunlp/OpenCLaP)
- [XLNet Chinese Pretrained Language Model](https://github.com/ymcui/Chinese-PreTrained-XLNet)
- [RoBERTa Chinese Pretrained Language Model](https://github.com/brightmart/roberta_zh)
- [RoBERTa-wwm-base-distill Chinese](https://github.com/policeme/roberta-wwm-base-distill)
- [ALBERT Chinese Pretrained Language Model](https://github.com/brightmart/albert_zh)
- [tomohideshibata / BERT-related-papers](https://github.com/tomohideshibata/BERT-related-papers)
- [Jiakui / awesome-bert](https://github.com/Jiakui/awesome-bert)
- [thunlp / PLMpapers](https://github.com/thunlp/PLMpapers)
- [terrifyzhao/bert-utils](https://github.com/terrifyzhao/bert-utils) - One line generate BERT's sent2vec for classification or matching task
- [hanxiao / bert-as-service](https://github.com/hanxiao/bert-as-service) - Using BERT model as a sentence encoding service
- [CLUEbenchmark / CLUE](https://github.com/CLUEbenchmark/CLUE) - Chinese Language Understanding Evaluation Benchmark
- [jessevig / bertviz](https://github.com/jessevig/bertviz) - BERT Visualization Tool
- [Tencent / TurboTransformers](https://github.com/Tencent/TurboTransformers)

### Chinese Blog
- [从Word Embedding到BERT模型—自然语言处理中的预训练技术发展史](https://zhuanlan.zhihu.com/p/49271699)
- [Bert时代的创新（应用篇）：Bert在NLP各领域的应用进展](https://zhuanlan.zhihu.com/p/68446772)
- [BERT时代的创新：BERT应用模式比较及其它](https://zhuanlan.zhihu.com/p/65470719)
- [效果惊人的GPT 2.0模型：它告诉了我们什么](https://zhuanlan.zhihu.com/p/56865533)
- [XLNet:运行机制及和Bert的异同比较](https://zhuanlan.zhihu.com/p/70257427)
- [BERT和Transformer到底学到了什么 | AI ProCon 2019](https://zhuanlan.zhihu.com/p/90167937)
- [深度长文：NLP的巨人肩膀（上）](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247493520&idx=1&sn=2b04c009ef75291ef3d19e8fe673aa36&chksm=96ea3810a19db10621e7a661974c796e8adeffc31625a769f8db1d87ba803cd58a30d40ad7ce&scene=21#wechat_redirect)
- [深度长文：NLP的巨人肩膀（下）](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247493731&idx=1&sn=51206e4ca3983548436d889590ab5347&chksm=96ea37e3a19dbef5b6db3143eb9df822915126d3d8f61fe73ddb9f8fa329d568ec79a662acb1&token=20831088&lang=zh_CN#rd)
- [BERT 瘦身之路：Distillation，Quantization，Pruning](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650411686&idx=2&sn=efc5abe3647e8d7a743eecd4280cfeec&chksm=becd94fc89ba1dea133f797592c54458f5748428c36a6a3b33760085f648ea9326204c325236&scene=21#wechat_redirect)
- [XLNet原理](http://fancyerii.github.io/2019/06/30/xlnet-theory/)
- [BERT 的演进和应用](https://mp.weixin.qq.com/s/u4k-A3dSb2-6PDodWPePhA)
- [从基础到前沿看迁移学习在NLP中的演化](https://mp.weixin.qq.com/s/Xsh3VNLYCxqh5TH_mK1uXQ)
- [BERT 瘦身之路：Distillation，Quantization，Pruning](https://zhuanlan.zhihu.com/p/86900556)
- [8篇论文梳理BERT相关模型进展与反思](https://www.msra.cn/zh-cn/news/features/bert)
- [BERT为何使用学习的position embedding而非正弦position encoding?](https://www.zhihu.com/question/307293465/answer/1039311514)
- [小米 / BERT适应业务遇难题？这是小米NLP的实战探索](https://mp.weixin.qq.com/s/XnKfqm-bj9tbPqf2lZJc0A)

### English Blog
- [A Fair Comparison Study of XLNet and BERT with Large Models](https://medium.com/@xlnet.team/a-fair-comparison-study-of-xlnet-and-bert-with-large-models-5a4257f59dc0)
- [Compressing BERT for faster prediction](https://blog.rasa.com/compressing-bert-for-faster-prediction-2/)
- [All The Ways You Can Compress BERT](http://mitchgordon.me/machine/learning/2019/11/18/all-the-ways-to-compress-BERT.html)
- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)
