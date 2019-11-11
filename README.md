# ATPapers
Must-read papers and related resources on attention mechanisim, Transformer and pretrained language model (PLM) such as BERT.

Suggestions about fixing errors or adding papers, repositories and other resources are welcomed!

值得一读的注意力机制、Transformer和预训练语言模型论文与相关资源集合。

欢迎修正错误以及新增论文、代码仓库与其他资源等建议！

## Attention
- Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (ICML 2015) [[paper]](http://proceedings.mlr.press/v37/xuc15.html) - *Hard & Soft Attention*
- Effective Approaches to Attention-based Neural Machine Translation (EMNLP 2015) [[paper]](https://www.aclweb.org/anthology/D15-1166/) - *Global & Local Attention*
- Neural Machine Translation by Jointly Learning to Align and Translate (ICLR 2015) [[paper]](https://arxiv.org/abs/1409.0473)
- Why Self-Attention? A Targeted Evaluation of Neural Machine Translation Architectures (EMNLP 2018) [[paper]](https://www.aclweb.org/anthology/D18-1458/)

## Transformer
### Papers
- Attention is All you Need (NIPS 2017) [[paper]](http://papers.nips.cc/paper/7181-attention-is-all-you-need)[[code]](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- Weighted Transformer Network for Machine Translation (CoRR 2017) [[paper]](https://arxiv.org/abs/1711.02132)[[code]](https://github.com/JayParks/transformer)
- Universal Transformers (CoLR 2019) [[paper]](https://openreview.net/forum?id=HyzdRiR9Y7)[[code]](https://github.com/andreamad8/Universal-Transformer-Pytorch)
- Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1285)
- Memory Transformer Networks (CS224n Winter2019 Reports) [[paper]](https://web.stanford.edu/class/cs224n/reports/custom/15778933.pdf)
- Star-Transformer (NAACL 2019) [[paper]](https://arxiv.org/pdf/1902.09113.pdf)
- On Layer Normalization in the Transformer Architecture (ICLR 2020) [[paper]](https://openreview.net/pdf?id=B1x8anVFPr)
### Chinese Blog
- [放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)

## Pretrained Language Model
### Models
- **`ELMo`** Deep Contextualized Word Representations (NAACL 2018) [[paper]](https://aclweb.org/anthology/N18-1202)
- **`ULMFit`** Universal Language Model Fine-tuning for Text Classification (ACL 2018) [[paper]](https://www.aclweb.org/anthology/P18-1031/)
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (NAACL 2019) [[paper]](https://www.aclweb.org/anthology/N19-1423)
- **`GPT`** Improving Language Understanding by Generative Pre-Training (CoRR 2018) [[paper]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- **`GPT-2`** Language Models are Unsupervised Multitask Learners (CoRR 2019) [[paper]](https://www.techbooky.com/wp-content/uploads/2019/02/Better-Language-Models-and-Their-Implications.pdf)[[code]](https://github.com/openai/gpt-2)
- MASS: Masked Sequence to Sequence Pre-training for Language Generation (ICML 2019) [[paper]](http://proceedings.mlr.press/v97/song19d/song19d.pdf)[[code]](https://github.com/microsoft/MASS)
- **`UNILM`** Unified Language Model Pre-training for Natural Language Understanding and Generation (CoRR 2019) [[paper]](https://arxiv.org/pdf/1905.03197.pdf)[[code]](https://github.com/microsoft/unilm)
- **`XLM`** Cross-lingual Language Model Pretraining (CoRR 2019) [[paper]](https://arxiv.org/pdf/1901.07291.pdf)
- **`MT-DNN`** Multi-Task Deep Neural Networks for Natural Language Understanding (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1441)[[code]](https://github.com/namisan/mt-dnn)
- ERNIE: Enhanced Language Representation with Informative Entities (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1139)[[code]](https://github.com/thunlp/ERNIE)
- ERNIE: Enhanced Representation through Knowledge Integration (CoRR 2019) [[paper]](https://arxiv.org/pdf/1904.09223.pdf)
- ERNIE 2.0: A Continual Pre-training Framework for Language Understanding (CoRR 2019) [[paper]](https://arxiv.org/pdf/1907.12412.pdf)
- Pre-Training with Whole Word Masking for Chinese BERT (CoRR 2019) [[paper]](https://arxiv.org/pdf/1906.08101.pdf)
- SpanBERT: Improving Pre-training by Representing and Predicting Spans (CoRR 2019) [[paper]](https://arxiv.org/pdf/1907.10529.pdf)
- XLNet: Generalized Autoregressive Pretraining for Language Understanding  (CoRR 2019) [[paper]](https://arxiv.org/pdf/1906.08237.pdf)
- RoBERTa: A Robustly Optimized BERT Pretraining Approach (CoRR 2019) [[paper]](https://arxiv.org/pdf/1907.11692.pdf)
- NEZHA: Neural Contextualized Representation for Chinese Language Understanding (CoRR 2019) [[paper]](https://arxiv.org/abs/1909.00204)
- **`T5`** Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transforme (CoRR 2019) [[paper]](https://arxiv.org/abs/1910.10683)[[code]](https://github.com/google-research/text-to-text-transfer-transformer)
- ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (ICLR 2020) [[paper]](https://openreview.net/forum?id=r1xMH1BtvB)
- BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (CoRR 2019) [[paper]](https://arxiv.org/abs/1910.13461)
- ZEN: Pre-training Chinese Text Encoder Enhanced by N-gram Representations (CoRR 2019) [[paper]](https://arxiv.org/abs/1911.00720)[[code]](https://github.com/sinovation/zen)
### Compression
- Distilling Task-Specific Knowledge from BERT into Simple Neural Networks (CoRR 2019) [[paper]](https://arxiv.org/abs/1903.12136)
- Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding (CoRR 2019) [[paper]](https://arxiv.org/abs/1904.09482)
- Patient Knowledge Distillation for BERT Model Compression (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1441/)
- Extreme Language Model Compression with Optimal Subwords and Shared Projections (ICLR 2019) [[paper]](https://arxiv.org/abs/1909.11687)
- DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter [[paper]](https://arxiv.org/pdf/1910.01108.pdf)[[code]](https://github.com/huggingface/transformers/tree/master/examples/distillation)
- TinyBERT: Distilling BERT for Natural Language Understanding (ICLR 2019) [[paper]](https://arxiv.org/pdf/1909.10351.pdf)
- ALBERT: A Lite BERT for Self-supervised Learning of Language Representations (ICLR 2019) [[paper]](https://arxiv.org/abs/1909.11942)
### Application
- BERT for Joint Intent Classification and Slot Filling (CoRR 2019) [[paper]](https://arxiv.org/abs/1902.10909)
- GPT-based Generation for Classical Chinese Poetry (CoRR 2019) [[paper]](https://arxiv.org/abs/1907.00151)
- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (EMNLP 2019) [[paper]](https://arxiv.org/abs/1908.10084)
### Analysis & Tools
- Probing Neural Network Comprehension of Natural Language Arguments (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1459.pdf)[[code]](https://github.com/IKMLab/arct2)
- To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks (RepL4NLP@ACL 2019) [[paper]](https://www.aclweb.org/anthology/W19-4302/)
- Multi-Head Multi-Layer Attention to Deep Language Representations for Grammatical Error Detection (CICLing 2019) [[paper]](https://arxiv.org/abs/1904.07334)
- Understanding the Behaviors of BERT in Ranking (CoRR 2019) [[paper]](https://arxiv.org/abs/1904.07531)
- Visualizing and Understanding the Effectiveness of BERT (EMNLP 2019) [[paper]](https://arxiv.org/pdf/1908.05620.pdf)
- exBERT: A Visual Analysis Tool to Explore Learned Representations in Transformers Models (CoRR 2019) [[paper]](https://arxiv.org/abs/1910.05276) [[code]](exbert.net)
- **`huggingface/pytorch-transformers`** Transformers: State-of-the-art Natural Language Processing [[paper]](https://arxiv.org/pdf/1910.03771.pdf) [[code]](https://github.com/huggingface/transformers)
### Tutorial & Survey
- Transfer Learning in Natural Language Processing (NAACL 2019) [[paper]](https://www.aclweb.org/anthology/N19-5004/)
### Repository
- [OpenClap: Open Chinese Language Pre-trained Model Zoo](https://github.com/thunlp/OpenCLaP)
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
### English Blog
- [Compressing BERT for faster prediction](https://blog.rasa.com/compressing-bert-for-faster-prediction-2/)
