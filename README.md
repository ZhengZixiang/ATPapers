# ATPapers
Worth-reading papers and related resources on **A**ttention Mechanism, **T**ransformer and **P**retrained Language Model (PLM) such as BERT.

Suggestions about fixing errors or adding papers, repositories and other resources are welcomed!

*Since I am Chinese, I mainly focus on Chinese resources. Welcome to recommend excellent resources in English or other languages!*

值得一读的注意力机制、Transformer和预训练语言模型论文与相关资源集合。

欢迎修正错误以及新增论文、代码仓库与其他资源等建议！

## Attention
### Papers
- **Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**. *Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio*. (ICML 2015) [[paper]](http://proceedings.mlr.press/v37/xuc15.html) - ***Hard & Soft Attention***
- **Effective Approaches to Attention-based Neural Machine Translation**. *Minh-Thang Luong, Hieu Pham, Christopher D. Manning*. (EMNLP 2015) [[paper]](https://www.aclweb.org/anthology/D15-1166/) - ***Global & Local Attention***
- **Neural Machine Translation by Jointly Learning to Align and Translate**. *Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio*. (ICLR 2015) [[paper]](https://arxiv.org/abs/1409.0473)
- **Non-local Neural Networks**. *Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He*. (CVPR 2018) [[paper]](https://arxiv.org/abs/1711.07971)[[code]](https://github.com/facebookresearch/video-nonlocal-net)
- **Why Self-Attention? A Targeted Evaluation of Neural Machine Translation Architectures**. *Gongbo Tang, Mathias Müller, Annette Rios, Rico Sennrich*. (EMNLP 2018) [[paper]](https://www.aclweb.org/anthology/D18-1458/)
- **Phrase-level Self-Attention Networks for Universal Sentence Encoding**. *Wei Wu, Houfeng Wang, Tianyu Liu, Shuming Ma*. (EMNLP 2018) [[paper]](https://www.aclweb.org/anthology/D18-1408/)
- **Bi-Directional Block Self-Attention for Fast and Memory-Efficient Sequence Modeling**. *Tao Shen, Tianyi Zhou, Guodong Long, Jing Jiang, Chengqi Zhang*. (ICLR 2018) [[paper]](https://arxiv.org/abs/1804.00857)[[code]](https://github.com/taoshen58/BiBloSA) - ***Bi-BloSAN***
- **Efficient Attention: Attention with Linear Complexities**. *Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, Hongsheng Li*. (CoRR 2018) [[paper]](https://arxiv.org/abs/1812.01243)[[code]](https://github.com/cmsflash/efficient-attention)
- **Leveraging Local and Global Patterns for Self-Attention Networks**. *Mingzhou Xu, Derek F. Wong, Baosong Yang, Yue Zhang, Lidia S. Chao*. (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1295/) [[tf code]](https://github.com/scewiner/Leveraging)[[pt code]](https://github.com/galsang/BiBloSA-pytorch)
- **Attention over Heads: A Multi-Hop Attention for Neural Machine Translation**. *Shohei Iida, Ryuichiro Kimura, Hongyi Cui, Po-Hsuan Hung, Takehito Utsuro, Masaaki Nagata*. (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-2030/)
- **Are Sixteen Heads Really Better than One?**. *Paul Michel, Omer Levy, Graham Neubig*. (NeurIPS 2019) [[paper]](https://arxiv.org/abs/1905.10650)
- **Attention is not Explanation**. *Sarthak Jain, Byron C. Wallace*. (NAACL 2019) [[paper]](https://arxiv.org/abs/1902.10186)
- **Attention is not not Explanation**. *Sarah Wiegreffe, Yuval Pinter*. (EMNLP 2019) [[paper]](https://arxiv.org/abs/1908.04626)
- **Is Attention Interpretable?**. *Sofia Serrano, Noah A. Smith*. (ACL 2019) [[paper]](https://arxiv.org/abs/1906.03731)
- **Human Attention Maps for Text Classification: Do Humans and Neural Networks Focus on the Same Words?**. *Cansu Sen, Thomas Hartvigsen, Biao Yin, Xiangnan Kong, Elke Rundensteiner*. (ACL 2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.419/) - ***YELP-HAT***
- **The elephant in the interpretability room: Why use attention as explanation when we have saliency methods?**. *Jasmijn Bastings, Katja Filippova*. (BlackboxNLP 2020) [[paper]](https://arxiv.org/abs/2010.05607)
- **Attention is Not Only a Weight: Analyzing Transformers with Vector Norms**. *Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, Kentaro Inui*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2004.10102)[[code]](https://github.com/gorokoba560/norm-analysis-of-transformer)

### Survey & Review
- **An Attentive Survey of Attention Models**. *Sneha Chaudhari, Gungor Polatkan, Rohan Ramanath, Varun Mithal*. (IJCAI 2019) [[paper]](https://arxiv.org/abs/1904.02874)

### English Blog
- [Illustrated: Self-Attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)
### Chinese Blog
- [JayLou / NLP中的Attention注意力机制+Transformer详解](https://zhuanlan.zhihu.com/p/53682800)
- [张俊林 / 深度学习中的注意力模型（2017版）](https://zhuanlan.zhihu.com/p/37601161)
- [哈工大SCIR / Attention！注意力机制可解释吗？](https://mp.weixin.qq.com/s/bCLLNLA2HB23v0jx3X2rIw)
- [科学空间 / 为节约而生：从标准Attention到稀疏Attention](https://kexue.fm/archives/6853)
- [科学空间 / Google新作Synthesizer：我们还不够了解自注意力](https://kexue.fm/archives/7430)
- [科学空间 / 线性Attention的探索：Attention必须有个Softmax吗？](https://kexue.fm/archives/7546)
### Repositories
- [thushv89 / Keras Attention Layer](https://github.com/thushv89/attention_keras) - Keras Layer implementation of Attention

## Transformer
### Papers
- **Attention is All you Need**. *Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin*. (NIPS 2017) [[paper]](https://arxiv.org/abs/1706.03762)[[code]](https://github.com/jadore801120/attention-is-all-you-need-pytorch) - ***Transformer***
- **Weighted Transformer Network for Machine Translation**. *Karim Ahmed, Nitish Shirish Keskar, Richard Socher*. (CoRR 2017) [[paper]](https://arxiv.org/abs/1711.02132)[[code]](https://github.com/JayParks/transformer)
- **Accelerating Neural Transformer via an Average Attention Network**. *Biao Zhang, Deyi Xiong, Jinsong Su*. (ACL 2018) [[paper]](https://arxiv.org/abs/1805.00631)[[code]](https://github.com/bzhangGo/transformer-aan) - ***AAN***
- **Self-Attention with Relative Position Representations**. *Peter Shaw, Jakob Uszkoreit, Ashish Vaswani*. (NAACL 2018) [[paper]](https://arxiv.org/abs/1803.02155) [[unoffical code]](https://github.com/THUNLP-MT/THUMT/blob/d4cb62c215d846093e5357aa17b286506b2df1af/thumt/layers/attention.py)
- **Universal Transformers**. *Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Lukasz Kaiser*. (ICLR 2019) [[paper]](https://arxiv.org/abs/1807.03819)[[code]](https://github.com/andreamad8/Universal-Transformer-Pytorch)
- **Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context**. *Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G. Carbonell, Quoc Viet Le, Ruslan Salakhutdinov*.  (ACL 2019) [[paper]](https://arxiv.org/abs/1901.02860)
- **Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned**. *Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, Ivan Titov*. (ACL 2019) [[paper]](https://arxiv.org/abs/1905.09418)
- **Star-Transformer**. *Qipeng Guo, Xipeng Qiu, Pengfei Liu, Yunfan Shao, Xiangyang Xue, Zheng Zhang*. (NAACL 2019) [[paper]](https://arxiv.org/pdf/1902.09113.pdf)
- **Generating Long Sequences with Sparse Transformers**. *Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1904.10509)[[code]](https://github.com/openai/sparse_attention)
- **Memory Transformer Networks**. *Jonas Metzger*. (CS224n Winter2019 Reports) [[paper]](https://web.stanford.edu/class/cs224n/reports/custom/15778933.pdf)
- **Transformer Dissection: A Unified Understanding of Transformer's Attention via the Lens of Kernel**. *Yao-Hung Hubert Tsai, Shaojie Bai, Makoto Yamada, Louis-Philippe Morency, Ruslan Salakhutdinov*. (EMNLP 2019) [[paper]](https://arxiv.org/abs/1908.11775)[[code]](https://github.com/yaohungt/TransformerDissection)
- **Transformers without Tears: Improving the Normalization of Self-Attention**. *Toan Q. Nguyen, Julian Salazar*. (IWSLT 2019) [[paper]](https://arxiv.org/abs/1910.05895)[[code]](https://github.com/tnq177/transformers_without_tears)
- **TENER: Adapting Transformer Encoder for Named Entity Recognition**. *Hang Yan, Bocao Deng, Xiaonan Li, Xipeng Qiu*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1911.04474)
- **Explicit Sparse Transformer: Concentrated Attention Through Explicit Selection**. *Guangxiang Zhao, Junyang Lin, Zhiyuan Zhang, Xuancheng Ren, Qi Su, Xu Sun*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1912.11637)[[code]](https://github.com/lancopku/Explicit-Sparse-Transformer)
- **Compressive Transformers for Long-Range Sequence Modelling**. *Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Timothy P. Lillicrap*. (ICLR 2020) [[paper]](https://arxiv.org/abs/1911.05507)[[code]](https://github.com/lucidrains/compressive-transformer-pytorch)
- **Reformer: The Efficient Transformer**. *Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya*. (ICLR 2020) [[paper]](https://arxiv.org/abs/2001.04451) [[code 1]](https://pastebin.com/62r5FuEW)[[code 2]](https://github.com/google/trax/tree/master/trax/models/reformer)[[code 3]](https://github.com/lucidrains/reformer-pytorch)
- **On Layer Normalization in the Transformer Architecture**. *Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu*. (ICML 2020) [[paper]](https://arxiv.org/abs/2002.04745)
- **Lite Transformer with Long-Short Range Attention**. *Zhanghao Wu, Zhijian Liu, Ji Lin, Yujun Lin, Song Han*. (ICLR 2020) [[paper]](https://arxiv.org/abs/2004.11886)[[code]](https://github.com/mit-han-lab/lite-transformer)
- **ReZero is All You Need: Fast Convergence at Large Depth**. *Thomas Bachlechner, Bodhisattwa Prasad Majumder, Huanru Henry Mao, Garrison W. Cottrell, Julian McAuley*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2003.04887) [[code]](https://github.com/majumderb/rezero) [[related Chinese post]](https://zhuanlan.zhihu.com/p/113384612)
- **Improving Transformer Models by Reordering their Sublayers**. *Ofir Press, Noah A. Smith, Omer Levy*. (ACL 2020) [[paper]](https://arxiv.org/abs/1911.03864)
- **Highway Transformer: Self-Gating Enhanced Self-Attentive Networks**. *Yekun Chai, Jin Shuo, Xinwen Hou*. (ACL 2020) [[paper]](https://arxiv.org/abs/2004.08178)[[code]](https://github.com/cyk1337/Highway-Transformer)
- **Efficient Content-Based Sparse Attention with Routing Transformers**. *Aurko Roy, Mohammad Saffar, Ashish Vaswani, David Grangier*. (TACL 2020) [[paper]](https://arxiv.org/abs/2003.05997)[[code]](https://github.com/google-research/google-research/tree/master/routing_transformer)
- **HAT: Hardware-Aware Transformers for Efficient Natural Language Processing**. *Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan, Song Han*. (ACL 2020) [[paper]](https://arxiv.org/abs/2005.14187)[[code]](https://link.zhihu.com/?target=https://github.com/mit-han-lab/hardware-aware-transformers)
- **Longformer: The Long-Document Transformer**. *Iz Beltagy, Matthew E. Peters, Arman Cohan*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.05150)[[code]](https://github.com/allenai/longformer)
- **Talking-Heads Attention**. *Noam Shazeer, Zhenzhong Lan, Youlong Cheng, Nan Ding, Le Hou*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2003.02436)
- **Synthesizer: Rethinking Self-Attention in Transformer Models**. *Yi Tay, Dara Bahri, Donald Metzler, Da-Cheng Juan, Zhe Zhao, Che Zheng*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2005.00743)
- **Linformer: Self-Attention with Linear Complexity**. *Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.04768)
- **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention**. *Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret*. (ICML 2020) [[paper]](https://arxiv.org/abs/2006.16236)[[code]](https://github.com/idiap/fast-transformers)[[project]](https://linear-transformers.com/)
- **Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing**. *Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.03236)[[code]](https://github.com/laiguokun/Funnel-Transformer)
- **Fast Transformers with Clustered Attention**. *Apoorv Vyas, Angelos Katharopoulos, François Fleuret*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2007.04825)[[code]](https://github.com/idiap/fast-transformers)
- **Memory Transformer**. *Mikhail S. Burtsev, Grigory V. Sapunov*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.11527)
- **Multi-Head Attention: Collaborate Instead of Concatenate**. *Jean-Baptiste Cordonnier, Andreas Loukas, Martin Jaggi*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.16362)[[code]](https://github.com/epfml/collaborative-attention)
- **Big Bird: Transformers for Longer Sequences**. *Big Bird: Transformers for Longer Sequences*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2007.14062)
- **Efficient Transformers: A Survey**. *Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2009.06732)
- **Rethinking Attention with Performers**. *Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2009.14794)[[code]](https://github.com/google-research/google-research/tree/master/performer/fast_self_attention)[[pytorch version]](https://github.com/lucidrains/performer-pytorch)[[blog]](https://ai.googleblog.com/2020/10/rethinking-attention-with-performers.html)
- **Long Range Arena: A Benchmark for Efficient Transformers**. *Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, Donald Metzler*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2011.04006)[[code]](https://github.com/google-research/long-range-arena)
- **Very Deep Transformers for Neural Machine Translation**. *Xiaodong Liu, Kevin Duh, Liyuan Liu, Jianfeng Gao*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2008.07772)[[code]](https://github.com/namisan/exdeep-nmt)
- **DeLighT: Very Deep and Light-weight Transformer**. *Sachin Mehta, Marjan Ghazvininejad, Srinivasan Iyer, Luke Zettlemoyer, Hannaneh Hajishirzi*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2008.00623)[[code]](https://github.com/sacmehta/delight)
- **FastFormers: Highly Efficient Transformer Models for Natural Language Understanding**. *Young Jin Kim, Hany Hassan Awadalla*. (SustaiNLP 2020 at EMNLP 2020) [[paper]](https://arxiv.org/abs/2010.13382)[[code]](https://github.com/microsoft/fastformers)

### Chinese Blog
- [Kaiyuan Gao / Transformers Assemble（PART I）](https://zhuanlan.zhihu.com/p/104935987)
- [科学空间 / 突破瓶颈，打造更强大的Transformer](https://kexue.fm/archives/7325)
- [量子位 / 最新Transformer模型大盘点，NLP学习必备，Google AI研究员出品](https://mp.weixin.qq.com/s/ptqSW86lwhB8Ogi3Ak_L9A)
- [美团 / Transformer 在美团搜索排序中的实践](https://tech.meituan.com/2020/04/16/transformer-in-meituan.html)
- [徐啸 / 浅谈 Transformer-based 模型中的位置表示](https://zhuanlan.zhihu.com/p/92017824)
- [张俊林 / 放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)
### English Blog
- [Google / Transformer-XL: Unleashing the Potential of Attention Models](https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html)
- [Google / Moving Beyond Translation with the Universal Transformer](https://ai.googleblog.com/2018/08/moving-beyond-translation-with.html)
- [Havard NLP / The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) [[code]](https://github.com/harvardnlp/annotated-transformer)
- [Jay Alammar / The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Madison May / A Survey of Long-Term Context in Transformers](https://www.pragmatic.ml/a-survey-of-methods-for-incorporating-long-term-context/) [[中文翻译]](https://mp.weixin.qq.com/s/JpBds6NQIBZ0S8GsMo4LEA)
- [Mohd / How Self-Attention with Relative Position Representations works](https://medium.com/@_init_/how-self-attention-with-relative-position-representations-works-28173b8c245a)

### Repositories
- [DongjunLee / transformer-tensorflow](https://github.com/DongjunLee/transformer-tensorflow) - Transformer Tensorflow implementation
- [andreamad8 / Universal-Transformer-Pytorch](https://github.com/andreamad8/Universal-Transformer-Pytorch) - Universal Transformer PyTorch implementation
- [lucidrains / Linear Attention Transformer](https://github.com/lucidrains/linear-attention-transformer) - Transformer based on a variant of attention that is linear complexity in respect to sequence length
- [PapersWithCode / Attention](https://paperswithcode.com/methods/category/attention-mechanisms)
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
- Multi-Task Deep Neural Networks for Natural Language Understanding (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1441)[[code]](https://github.com/namisan/mt-dnn) - ***MT-DNN***
- 75 Languages, 1 Model: Parsing Universal Dependencies Universally[[paper]](https://www.aclweb.org/anthology/D19-1279/)[[code]](https://github.com/hyperparticle/udify) - ***UDify***
- Defending Against Neural Fake News (CoRR 2019) [[paper]](https://arxiv.org/abs/1905.12616)[[code]](https://rowanzellers.com/grover/) - ***Grover***
- ERNIE 2.0: A Continual Pre-training Framework for Language Understanding (CoRR 2019) [[paper]](https://arxiv.org/pdf/1907.12412.pdf) - ***ERNIE 2.0 (Baidu)***
- Pre-Training with Whole Word Masking for Chinese BERT (CoRR 2019) [[paper]](https://arxiv.org/pdf/1906.08101.pdf) - ***Chinese-BERT-wwm***
- SpanBERT: Improving Pre-training by Representing and Predicting Spans (CoRR 2019) [[paper]](https://arxiv.org/pdf/1907.10529.pdf) - ***SpanBERT***
- XLNet: Generalized Autoregressive Pretraining for Language Understanding  (CoRR 2019) [[paper]](https://arxiv.org/pdf/1906.08237.pdf)[[code]](https://github.com/zihangdai/xlnet) - ***XLNet***
- RoBERTa: A Robustly Optimized BERT Pretraining Approach (CoRR 2019) [[paper]](https://arxiv.org/pdf/1907.11692.pdf) - ***RoBERTa***
- NEZHA: Neural Contextualized Representation for Chinese Language Understanding (CoRR 2019) [[paper]](https://arxiv.org/abs/1909.00204)[[code]](https://github.com/huawei-noah/Pretrained-Language-Model) - ***NEZHA***
- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (CoRR 2019) [[paper]](https://arxiv.org/abs/1909.08053)[[code]](https://github.com/NVIDIA/Megatron-LM) - ***Megatron-LM***
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transforme (CoRR 2019) [[paper]](https://arxiv.org/abs/1910.10683)[[code]](https://github.com/google-research/text-to-text-transfer-transformer) - ***T5***
- BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (CoRR 2019) [[paper]](https://arxiv.org/abs/1910.13461) - ***BART***
- ZEN: Pre-training Chinese Text Encoder Enhanced by N-gram Representations (CoRR 2019) [[paper]](https://arxiv.org/abs/1911.00720)[[code]](https://github.com/sinovation/zen) - ***ZEN***
- The JDDC Corpus: A Large-Scale Multi-Turn Chinese Dialogue Dataset for E-commerce Customer Service (CoRR 2019) [[paper]](https://arxiv.org/pdf/1911.09969.pdf)[[code]](https://github.com/jd-aig/nlp_baai) - ***BAAI-JDAI-BERT***
- UER: An Open-Source Toolkit for Pre-training Models (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-3041/)[[code]](https://github.com/dbiir/UER-py) - ***UER***
- ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (ICLR 2020) [[paper]](https://openreview.net/forum?id=r1xMH1BtvB) - ***ELECTRA***
- StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding (ICLR 2020) [[paper]](https://arxiv.org/abs/1908.04577) - ***StructBERT***
- FreeLB: Enhanced Adversarial Training for Language Understanding (ICLR 2020) [[paper]](https://arxiv.org/abs/1909.11764)[[code]](https://github.com/zhuchen03/FreeLB) - ***FreeLB***
- HUBERT Untangles BERT to Improve Transfer across NLP Tasks (CoRR 2019) [[paper]](https://arxiv.org/pdf/1910.12647.pdf) - ***HUBERT***
- CodeBERT: A Pre-Trained Model for Programming and Natural Languages (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.08155) - ***CodeBERT***
- ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training (CoRR 2020) [[paper]](https://arxiv.org/abs/2001.04063) - ***ProphetNet***
- ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation (CoRR 2020) [[paper]](https://arxiv.org/abs/2001.11314)[[code]](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gen) - ***ERNIE-GEN***
- Efficient Training of BERT by Progressively Stacking (ICML 2019) [[paper]](http://proceedings.mlr.press/v97/gong19a.html)[[code]](https://github.com/gonglinyuan/StackingBERT) - ***StackingBERT***
- PoWER-BERT: Accelerating BERT Inference via Progressive Word-vector Elimination (CoRR 2020) [[paper]](https://arxiv.org/abs/2001.08950)[[code]](https://github.com/IBM/PoWER-BERT)
- Towards a Human-like Open-Domain Chatbot (CoRR 2020) [[paper]](https://arxiv.org/abs/2001.09977) - ***Meena***
- UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.12804)[[code]](https://github.com/microsoft/unilm) - ***UNILMv2***
- Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.04092)[[code]](https://github.com/ChunyuanLI/Optimus) - ***Optimus***
- **SegaBERT: Pre-training of Segment-aware BERT for Language Understanding**. *He Bai, Peng Shi, Jimmy Lin, Luchen Tan, Kun Xiong, Wen Gao, Ming Li*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.14996)
- **Adversarial Training for Large Neural Language Models**. *Xiaodong Liu, Hao Cheng, Pengcheng He, Weizhu Chen, Yu Wang, Hoifung Poon, Jianfeng Gao*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.08994)[[code]](https://github.com/namisan/mt-dnn) - ***ALUM***
- MPNet: Masked and Permuted Pre-training for Language Understanding (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.09297)[[code]](https://github.com/microsoft/MPNet) - ***MPNet***
- Language Models are Few-Shot Learners (CoRR 2020) [[paper]](https://arxiv.org/abs/2005.14165)[[code]](https://github.com/openai/gpt-3) - ***GPT-3***
- SPECTER: Document-level Representation Learning using Citation-informed Transformers (ACL 2020) [[paper]](https://arxiv.org/abs/2004.07180) - ***SPECTER***
- Recipes for building an open-domain chatbot (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.13637)[[post]](https://ai.facebook.com/blog/state-of-the-art-open-source-chatbot/)[[code]](https://github.com/facebookresearch/ParlAI) - ***Blender***
- **Revisiting Pre-Trained Models for Chinese Natural Language Processing**. *Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Shijin Wang, Guoping Hu*. (Findings of EMNLP 2020) [[paper]](https://arxiv.org/abs/2004.13922)[[code]](https://github.com/ymcui/MacBERT)[[blog]](https://mp.weixin.qq.com/s/J7SRrhOe--kHuAI1V8oLRQ) - ***MacBERT***
- PLATO-2: Towards Building an Open-Domain Chatbot via Curriculum Learning (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.16779)[[code]](https://github.com/PaddlePaddle/Knover/tree/master/plato-2) - ***PLATO-2***
- DeBERTa: Decoding-enhanced BERT with Disentangled Attention (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.03654)[[code]](https://github.com/microsoft/DeBERTa) - ***DeBERTa***
- **ConvBERT: Improving BERT with Span-based Dynamic Convolution**. *Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2008.02496)[[code]](https://github.com/yitu-opensource/ConvBert)
- **AMBERT: A Pre-trained Language Model with Multi-Grained Tokenization**. *Xinsong Zhang, Hang Li*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2008.11869)
- **CharBERT: Character-aware Pre-trained Language Model**. *Wentao Ma, Yiming Cui, Chenglei Si, Ting Liu, Shijin Wang, Guoping Hu*. (COLING 2020) [[paper]](https://arxiv.org/abs/2011.01513)[[code]](https://github.com/wtma/CharBERT)[[blog]](https://mp.weixin.qq.com/s/jY2IE-uxn38aFvjPNiwWGw)
- **MVP-BERT: Redesigning Vocabularies for Chinese BERT and Multi-Vocab Pretraining**. *Wei Zhu*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2011.08539)

### Multi-Modal
- **VideoBERT: A Joint Model for Video and Language Representation Learning**. *Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, Cordelia Schmid*. (ICCV 2019) [[paper]](https://arxiv.org/abs/1904.01766)
- **Learning Video Representations using Contrastive Bidirectional Transformer**. *Chen Sun, Fabien Baradel, Kevin Murphy, Cordelia Schmid*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1906.05743) - ***CBT***
- **ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks**. *Jiasen Lu, Dhruv Batra, Devi Parikh, Stefan Lee*. (NeurIPS 2019) [[paper]](https://arxiv.org/abs/1908.02265)[[code]](https://github.com/jiasenlu/vilbert_beta)
- **VisualBERT: A Simple and Performant Baseline for Vision and Language**. *Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1908.03557)[[code]](https://github.com/uclanlp/visualbert)
- **Fusion of Detected Objects in Text for Visual Question Answering**. *Chris Alberti, Jeffrey Ling, Michael Collins, David Reitter*. (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1219/)[[code]](https://github.com/google-research/language/tree/master/language/question_answering/b2t2) - ***B2T2***
- **LXMERT: Learning Cross-Modality Encoder Representations from Transformers**. *Hao Tan, Mohit Bansal*. (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1514/)[[code]](https://github.com/airsplay/lxmert)
- **Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training**. *Gen Li, Nan Duan, Yuejian Fang, Ming Gong, Daxin Jiang, Ming Zhou*. (AAAI 2020) [[paper]](https://arxiv.org/abs/1908.06066)
- **FashionBERT: Text and Image Matching with Adaptive Loss for Cross-modal Retrieval**. *Dehong Gao, Linbo Jin, Ben Chen, Minghui Qiu, Peng Li, Yi Wei, Yi Hu, Hao Wang*. （SIGIR 2020) [[paper]](https://arxiv.org/abs/2005.09801)
- **UNITER: Learning UNiversal Image-TExt Representations**. *Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, Jingjing Liu*. (ECCV 2020) [[paper]](https://arxiv.org/abs/1909.11740)[[code]](https://github.com/ChenRocks/UNITER)
- **Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks**. *Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, Yejin Choi, Jianfeng Gao*. (ECCV 2020) [[paper]](https://arxiv.org/abs/2004.06165)[[code]](https://github.com/microsoft/Oscar)
- **VD-BERT: A Unified Vision and Dialog Transformer with BERT**. *VD-BERT: A Unified Vision and Dialog Transformer with BERT*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2004.13278)[[code]](https://github.com/salesforce/VD-BERT)
- **CodeBERT:A Pre-Trained Model for Programming and Natural Languages**. *Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, Ming Zhou*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2002.08155)[[code]](https://github.com/microsoft/CodeBERT)
- **VL-BERT: Pre-training of Generic Visual-Linguistic Representatio**. *Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, Jifeng Dai*. (ICLR 2020) [[paper]](https://arxiv.org/abs/1908.08530)[[code]](https://github.com/jackroos/VL-BERT)
- **ERNIE-ViL: Knowledge Enhanced Vision-Language Representations Through Scene Graph**. *Fei Yu, Jiji Tang, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.16934)
- **Large-Scale Adversarial Training for Vision-and-Language Representation Learning**. *Zhe Gan, Yen-Chun Chen, Linjie Li, Chen Zhu, Yu Cheng, Jingjing Liu*. (NeurIPS 2020) [[paper]](https://arxiv.org/abs/2006.06195) - ***VILLA***

### Multilingual
- **Cross-lingual Language Model Pretraining**. *Guillaume Lample, Alexis Conneau*. (NeuIPS 2019) [[paper]](https://arxiv.org/abs/1901.07291)[[code]](https://github.com/facebookresearch/XLM) - ***XLM***
- **Unsupervised Cross-lingual Representation Learning at Scale**. *Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov*. (ACL 2020) [[paper]](https://arxiv.org/abs/1911.02116) - ***XLM-R***
- **Multilingual Denoising Pre-training for Neural Machine Translation**. *Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2001.08210) - ***mBART***
- **MultiFiT: Efficient Multi-lingual Language Model Fine-tuning**. *Julian Eisenschlos, Sebastian Ruder, Piotr Czapla, Marcin Kadras, Sylvain Gugger, Jeremy Howard*. (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1572/)[[code]](https://github.com/n-waves/multifit)
- **XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization**. *Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, Melvin Johnson*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2003.11080)[[code]](https://github.com/google-research/xtreme)
- **Pre-training via Paraphrasing**. *Mike Lewis, Marjan Ghazvininejad, Gargi Ghosh, Armen Aghajanyan, Sida Wang, Luke Zettlemoyer*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.15020) - ***MARGE***
- **WikiBERT Models: Deep Transfer Learning for Many Languages**. *Sampo Pyysalo, Jenna Kanerva, Antti Virtanen, Filip Ginter*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.01538)[[code]](https://github.com/turkunlp/wikibert)
- **Language-agnostic BERT Sentence Embedding**. *Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, Wei Wang*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2007.01852) - ***LaBSE***
- **Pre-training Multilingual Neural Machine Translation by Leveraging Alignment Information**. *Zehui Lin, Xiao Pan, Mingxuan Wang, Xipeng Qiu, Jiangtao Feng, Hao Zhou, Lei Li*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2010.03142)[[code]](https://github.com/linzehui/mRASP) - ***mRASP***
- **mT5: A massively multilingual pre-trained text-to-text transformer**. *Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2010.11934)[[code]](https://github.com/google-research/multilingual-t5)
- **InfoXLM: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training**. *Zewen Chi, Li Dong, Furu Wei, Nan Yang, Saksham Singhal, Wenhui Wang, Xia Song, Xian-Ling Mao, Heyan Huang, Ming Zhou* (CoRR 2020) [[paper]](https://arxiv.org/abs/2007.07834)[[code]](https://github.com/microsoft/unilm/tree/master/infoxlm)

### Knowledge
- **ERNIE: Enhanced Language Representation with Informative Entities**. *Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun, Qun Liu*. (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1139)[[code]](https://github.com/thunlp/ERNIE)
- **ERNIE: Enhanced Representation through Knowledge Integration**. *Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, Hua Wu*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1904.09223)
- **Knowledge Enhanced Contextual Word Representations**. *Matthew E. Peters, Mark Neumann, Robert Logan, Roy Schwartz, Vidur Joshi, Sameer Singh, Noah A. Smith*. (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1005/) - ***KnowBert***
- **K-BERT: Enabling Language Representation with Knowledge Graph**. *Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng, Ping Wang*. (AAAI 2020) [[paper]](https://arxiv.org/abs/1909.07606)[[code]](https://github.com/autoliuweijie/K-BERT)
- **KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation**. *Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhengyan Zhang, Zhiyuan Liu, Juanzi Li, Jian Tang*. (TACL 2020) [[paper]](https://arxiv.org/abs/1911.06136)[[code]](https://github.com/THU-KEG/KEPLER)
- **CoLAKE: Contextualized Language and Knowledge Embedding**. *Tianxiang Sun, Yunfan Shao, Xipeng Qiu, Qipeng Guo, Yaru Hu, Xuanjing Huang, Zheng Zhang*. (COLING 2020) [[paper]](https://arxiv.org/abs/2010.00309)[[code]](https://github.com/txsun1997/CoLAKE)
- **Exploiting Structured Knowledge in Text via Graph-Guided Representation Learning**. *Tao Shen, Yi Mao, Pengcheng He, Guodong Long, Adam Trischler, Weizhu Chen*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2004.14224)
- **K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters**. *Ruize Wang, Duyu Tang, Nan Duan, Zhongyu Wei, Xuanjing Huang, Jianshu ji, Guihong Cao, Daxin Jiang, Ming Zhou*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.01808)
- **BERT-MK: Integrating Graph Contextualized Knowledge into Pre-trained Language Models**. *Bin He, Di Zhou, Jinghui Xiao, Xin Jiang, Qun Liu, Nicholas Jing Yuan, Tong Xu*. (EMNLP 2020) [[paper]](https://www.aclweb.org/anthology/2020.findings-emnlp.207/)
- **JAKET: Joint Pre-training of Knowledge Graph and Language Understanding**. *Donghan Yu, Chenguang Zhu, Yiming Yang, Michael Zeng*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2010.00796)

### Compression & Accelerating
- **Distilling Task-Specific Knowledge from BERT into Simple Neural Networks**. *Raphael Tang, Yao Lu, Linqing Liu, Lili Mou, Olga Vechtomova, Jimmy Lin*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1903.12136)
- **Model Compression with Multi-Task Knowledge Distillation for Web-scale Question Answering System**. *Ze Yang, Linjun Shou, Ming Gong, Wutao Lin, Daxin Jiang*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1904.09636) - ***MKDM***
- **Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding**. *Xiaodong Liu, Pengcheng He, Weizhu Chen, Jianfeng Gao*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1904.09482)
- **Well-Read Students Learn Better: On the Importance of Pre-training Compact Models**. *Iulia Turc, Ming-Wei Chang, Kenton Lee, Kristina Toutanova*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1908.08962)
- **Small and Practical BERT Models for Sequence Labeling**. *Henry Tsai, Jason Riesa, Melvin Johnson, Naveen Arivazhagan, Xin Li, Amelia Archer*. (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1374/)
- **Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT**. *Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W. Mahoney, Kurt Keutzer*. (AAAI 2020) [[paper]](https://arxiv.org/abs/1909.05840)
- **Patient Knowledge Distillation for BERT Model Compression**. *Siqi Sun, Yu Cheng, Zhe Gan, Jingjing Liu*. (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1441/) - ***BERT-PKD***
- **Extreme Language Model Compression with Optimal Subwords and Shared Projections**. *Sanqiang Zhao, Raghav Gupta, Yang Song, Denny Zhou*. (ICLR 2019) [[paper]](https://arxiv.org/abs/1909.11687)
- **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**. *Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf*. [[paper]](https://arxiv.org/pdf/1910.01108.pdf)[[code]](https://github.com/huggingface/transformers/tree/master/examples/distillation)
- **TinyBERT: Distilling BERT for Natural Language Understanding**. *Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu*. (ICLR 2019) [[paper]](https://arxiv.org/pdf/1909.10351.pdf)[[code]](https://github.com/huawei-noah/Pretrained-Language-Model)
- **Q8BERT: Quantized 8Bit BERT**. *Ofir Zafrir, Guy Boudoukh, Peter Izsak, Moshe Wasserblat*. (NeurIPS 2019 Workshop) [[paper]](https://arxiv.org/abs/1910.06188)
- **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**. *Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut*. (ICLR 2020) [[paper]](https://arxiv.org/abs/1909.11942)[[code]](https://github.com/google-research/ALBERT)
- **Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning**. *Mitchell A. Gordon, Kevin Duh, Nicholas Andrews*. (ICLR 2020) [[paper]](https://openreview.net/forum?id=SJlPOCEKvH)[[PyTorch code]](https://github.com/lonePatient/albert_pytorch)
- **Reducing Transformer Depth on Demand with Structured Dropout**. *Angela Fan, Edouard Grave, Armand Joulin*. (ICLR 2020) [[paper]](https://arxiv.org/abs/1909.11556) - ***LayerDrop***
- Multilingual Alignment of Contextual Word Representations (ICLR 2020) [[paper]](https://arxiv.org/abs/2002.03518)
- **AdaBERT: Task-Adaptive BERT Compression with Differentiable Neural Architecture Search**. *Daoyuan Chen, Yaliang Li, Minghui Qiu, Zhen Wang, Bofang Li, Bolin Ding, Hongbo Deng, Jun Huang, Wei Lin, Jingren Zhou*. (IJCAI 2020) [[paper]](https://arxiv.org/pdf/2001.04246.pdf) - ***AdaBERT***
- **BERT-of-Theseus: Compressing BERT by Progressive Module Replacing**. *Canwen Xu, Wangchunshu Zhou, Tao Ge, Furu Wei, Ming Zhou*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.02925)[[pt code]](https://github.com/JetRunner/BERT-of-Theseus)[[tf code]](https://github.com/qiufengyuyi/bert-of-theseus-tf)[[keras code]](https://github.com/bojone/bert-of-theseus)
- **MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers**. *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.10957)[[code]](https://github.com/microsoft/unilm/tree/master/minilm)
- **FastBERT: a Self-distilling BERT with Adaptive Inference Time**. *Weijie Liu, Peng Zhou, Zhiruo Wang, Zhe Zhao, Haotang Deng, Qi Ju*. (ACL 2020) [[paper]](https://arxiv.org/abs/2004.02178)[[code]](https://github.com/autoliuweijie/FastBERT)
- **MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices**. *Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, Denny Zhou*. (ACL 2020) [[paper]](https://arxiv.org/abs/2004.02984)[[code]](https://github.com/google-research/google-research/tree/master/mobilebert)
- **Towards Non-task-specific Distillation of BERT via Sentence Representation Approximation**. *Bowen Wu, Huan Zhang, Mengyuan Li, Zongsheng Wang, Qihang Feng, Junhong Huang, Baoxun Wang*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.03097) - ***BiLSTM-SRA & LTD-BERT***
- **Poor Man's BERT: Smaller and Faster Transformer Models**. *Hassan Sajjad, Fahim Dalvi, Nadir Durrani, Preslav Nakov*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.03844)
- **DynaBERT: Dynamic BERT with Adaptive Width and Depth**. *Lu Hou, Lifeng Shang, Xin Jiang, Qun Liu*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.04037)
- **SqueezeBERT: What can computer vision teach NLP about efficient neural networks?**. *Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, Kurt W. Keutzer*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.11316)
- **Optimal Subarchitecture Extraction For BERT**. *Adrian de Wynter, Daniel J. Perry*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2010.10499)[[code]](https://github.com/alexa/bort/) - ***Bort***
- **TernaryBERT: Distillation-aware Ultra-low Bit BERT**. *Wei Zhang, Lu Hou, Yichun Yin, Lifeng Shang, Xiao Chen, Xin Jiang, Qun Liu*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2009.12812)[[code]]()[[blog]](https://mp.weixin.qq.com/s/jHfUAe1Pr6HpARzRw2Y0hw)
- **BERT-EMD: Many-to-Many Layer Mapping for BERT Compression with Earth Mover's Distance**. *Jianquan Li, Xiaokang Liu, Honghong Zhao, Ruifeng Xu, Min Yang, Yaohong Jin*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2010.06133)[[code]](https://github.com/lxk00/BERT-EMD)[[blog]](https://mp.weixin.qq.com/s/w1sT126jS_lZ_Q3cRi6fGQ)

### Application
- BERT for Joint Intent Classification and Slot Filling (CoRR 2019) [[paper]](https://arxiv.org/abs/1902.10909)
- GPT-based Generation for Classical Chinese Poetry (CoRR 2019) [[paper]](https://arxiv.org/abs/1907.00151)
- **BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer**. *Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, Peng Jiang*. (CIKM 2019) [[paper]](https://arxiv.org/abs/1904.06690)[[code]](https://github.com/FeiSun/BERT4Rec)
- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (EMNLP 2019) [[paper]](https://arxiv.org/abs/1908.10084)[[code]](https://github.com/UKPLab/sentence-transformers)
- Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring (ICLR 2020) [[paper]](https://arxiv.org/abs/1905.01969)
- Pre-training Tasks for Embedding-based Large-scale Retrieval (ICLR 2020) [[paper]](https://arxiv.org/abs/2002.03932)
- Keyword-Attentive Deep Semantic Matching (CoRR 2020) [[paper & code]](https://github.com/DataTerminatorX/Keyword-BERT) [[post]](https://zhuanlan.zhihu.com/p/112562420) - ***Keyword BERT***
- Unified Multi-Criteria Chinese Word Segmentation with BERT (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.05808)
- ToD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogues (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.06871)[[code]](https://github.com/jasonwu0731/ToD-BERT)
- Spelling Error Correction with Soft-Masked BERT (ACL 2020) [[paper]](https://arxiv.org/abs/2005.07421) - ***Soft-Masked BERT***
- DeFormer: Decomposing Pre-trained Transformers for Faster Question Answering (ACL 2020) [[paper]](https://arxiv.org/abs/2005.00697)[[code]](https://github.com/StonyBrookNLP/deformer) - ***DeFormer***
- BLEURT: Learning Robust Metrics for Text Generation (ACL 2020) [[paper]](https://arxiv.org/abs/2004.04696)[[code]](https://github.com/google-research/bleurt) - ***BLEURT***
- Context-Aware Document Term Weighting for Ad-Hoc Search (WWW 2020) [[paper]](https://dl.acm.org/doi/pdf/10.1145/3366423.3380258)[[code]](https://github.com/AdeDZY/DeepCT) - ***HDCT***
- **E-BERT: A Phrase and Product Knowledge Enhanced Language Model for E-commerce**. *Denghui Zhang, Zixuan Yuan, Yanchi Liu, Zuohui Fu, Fuzhen Zhuang, Pengyang Wang, Haifeng Chen, Hui Xiong*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2009.02835)
- **Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks**. *Nandan Thakur, Nils Reimers, Johannes Daxenberger, Iryna Gurevych*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2010.08240)
- **CogLTX: Applying BERT to Long Texts**. *Ming Ding, Chang Zhou, Hongxia Yang, Jie Tang*. (NeurIPS 2020) [[paper]](http://keg.cs.tsinghua.edu.cn/jietang/publications/NIPS20-Ding-et-al-CogLTX.pdf)[[code]](https://github.com/Sleepychord/CogLTX)

### Analysis & Tools
- Probing Neural Network Comprehension of Natural Language Arguments (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1459.pdf)[[code]](https://github.com/IKMLab/arct2)
- Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1334/) [[code]](https://github.com/tommccoy1/hans)
- To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks (RepL4NLP@ACL 2019) [[paper]](https://www.aclweb.org/anthology/W19-4302/)
- **BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model**. *Alex Wang, Kyunghyun Cho*. (NeuralGen 2019) [[paper]](https://arxiv.org/abs/1902.04094)
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
- **A Primer in BERTology: What we know about how BERT works**. *Anna Rogers, Olga Kovaleva, Anna Rumshisky*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.12327)
- BERT Can See Out of the Box: On the Cross-modal Transferability of Text Representations (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.10832)
- Contextual Embeddings: When Are They Worth It? (ACL 2020) [[paper]](https://arxiv.org/abs/2005.09117)
- Weight Poisoning Attacks on Pre-trained Models (ACL 2020) [[paper]](https://arxiv.org/abs/2004.06660)[[code]](https://github.com/neulab/RIPPLe) - ***RIPPLe***
- Roles and Utilization of Attention Heads in Transformer-based Neural Language Models (ACL 2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.311/)[[code]](https://github.com/heartcored98/transformer_anatomy) - ***Transformer Anatomy***
- Adversarial Training for Large Neural Language Models (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.08994)[[code]](https://github.com/namisan/mt-dnn/tree/master/alum)
- Cross-Lingual Ability of Multilingual BERT: An Empirical Study (ICLR 2020) [[paper]](https://arxiv.org/abs/1912.07840)[[code]](https://github.com/ZihanWangKi/mbert-study)
- DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference (ACL 2020) [[paper]](https://arxiv.org/abs/2004.12993)[[code]](https://github.com/castorini/DeeBERT)[[huggingface implementation]](https://github.com/huggingface/transformers/tree/master/examples/deebert)
- **Beyond Accuracy: Behavioral Testing of NLP models with CheckList**. *Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, Sameer Singh*. (ACL 2020 Best Paper) [[paper]](https://arxiv.org/abs/2005.04118)[[code]](https://github.com/marcotcr/checklist)
- **Don't Stop Pretraining: Adapt Language Models to Domains and Tasks**. *Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, Noah A. Smith*. (ACL 2020) [[paper]](https://arxiv.org/abs/2004.10964)[[code]](https://github.com/allenai/dont-stop-pretraining)
- **TextBrewer: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing**. *Ziqing Yang, Yiming Cui, Zhipeng Chen, Wanxiang Che, Ting Liu, Shijin Wang, Guoping Hu*. (ACL 2020) [[paper]](https://arxiv.org/abs/2002.12620)[[code]](https://github.com/airaria/TextBrewer)
- **Perturbed Masking: Parameter-free Probing for Analyzing and Interpreting BERT**. *Zhiyong Wu, Yun Chen, Ben Kao, Qun Liu*. (ACL 2020) [[paper]](https://arxiv.org/abs/2004.14786)[[pt code]](https://github.com/LividWo/Perturbed-Masking)[[keras code]](https://github.com/bojone/perturbed_masking)
- **Rethinking Positional Encoding in Language Pre-training**. *Guolin Ke, Di He, Tie-Yan Liu*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.15595)[[code]](https://github.com/guolinke/TUPE) - ***TUPE***
- **Variance-reduced Language Pretraining via a Mask Proposal Network**. *Liang Chen*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2008.05333)
- **Does BERT Solve Commonsense Task via Commonsense Knowledge?**. *Leyang Cui, Sijie Cheng, Yu Wu, Yue Zhang*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2008.03945)
- **Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference**. *Timo Schick, Hinrich Schütze*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2001.07676)[[code]](https://github.com/timoschick/pet) - ***PET***
- **It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners**. *Timo Schick, Hinrich Schütze*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2009.07118)[[code]](https://github.com/timoschick/pet)
- **Automatically Identifying Words That Can Serve as Labels for Few-Shot Text Classification**. *Timo Schick, Helmut Schmid, Hinrich Schütze*. (COLING 2020) [[paper]](https://arxiv.org/abs/2010.13641)[[code]](https://github.com/timoschick/pet)
- **InfoBERT: Improving Robustness of Language Models from An Information Theoretic Perspective**. *Boxin Wang, Shuohang Wang, Yu Cheng, Zhe Gan, Ruoxi Jia, Bo Li, Jingjing Liu*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2010.02329)
- **Self-training Improves Pre-training for Natural Language Understanding**. *Jingfei Du, Edouard Grave, Beliz Gunel, Vishrav Chaudhary, Onur Celebi, Michael Auli, Ves Stoyanov, Alexis Conneau*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2010.02194)[[related blog]](https://mp.weixin.qq.com/s/rCMFy08viSnLbqg-Yt91Sg)
- **Commonsense knowledge adversarial dataset that challenges ELECTRA**. *Gongqi Lin, Yuan Miao, Xiaoyong Yang, Wenwu Ou, Lizhen Cui, Wei Guo, Chunyan Miao*. (ICARCV 2020) [[paper]](https://arxiv.org/abs/2010.13049)
- **Neural Semi-supervised Learning for Text Classification Under Large-Scale Pretraining**. *Zijun Sun, Chun Fan, Xiaofei Sun, Yuxian Meng, Fei Wu, Jiwei Li*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2011.08626)[[code]](https://github.com/ShannonAI/Neural-Semi-Supervised-Learning-for-Text-Classification)[[blog]](https://mp.weixin.qq.com/s/t7a_1cf1EFuoTYnm2gAYSw)
- **Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting**. *Sanyuan Chen, Yutai Hou, Yiming Cui, Wanxiang Che, Ting Liu, Xiangzhan Yu*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2004.12651)[[code]](https://github.com/Sanyuan-Chen/RecAdam)[[blog]](https://mp.weixin.qq.com/s/M89mqFxa7_iK3lzlEgLzAQ)
- **To BERT or Not to BERT: Comparing Task-specific and Task-agnostic Semi-Supervised Approaches for Sequence Tagging**. *Kasturi Bhattacharjee, Miguel Ballesteros, Rishita Anubhai, Smaranda Muresan, Jie Ma, Faisal Ladhak, Yaser Al-Onaizan*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2010.14042)
- **Investigating Novel Verb Learning in BERT: Selectional Preference Classes and Alternation-Based Syntactic Generalization**. *Tristan Thrush, Ethan Wilcox, Roger Levy*. (BlackboxNLP 2020) [[paper]](https://arxiv.org/abs/2011.02417)[[code]](https://github.com/TristanThrush/few-shot-lm-learning)
- **On the Sentence Embeddings from Pre-trained Language Models**. *Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, Lei Li*. (EMNLP 2020) [[paper]](https://arxiv.org/pdf/2011.05864.pdf)[[code]](https://github.com/bohanli/BERT-flow)

### Tutorial & Survey
- **Transfer Learning in Natural Language Processing**. *Sebastian Ruder, Matthew E. Peters, Swabha Swayamdipta, Thomas Wolf*. (NAACL 2019) [[paper]](https://www.aclweb.org/anthology/N19-5004/)
- **Evolution of Transfer Learning in Natural Language Processing**. *Aditya Malte, Pratik Ratadiya*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1910.07370)
- **Transferring NLP Models Across Languages and Domains**. *Barbara Plank*. (DeepLo 2019) [[slides]](https://www.dropbox.com/s/myle46vl64nasg8/Deeplo-talk-2019.pdf?dl=0)
- **Recent Breakthroughs in Natural Language Processing**. *Christopher Manning* (BAAI 2019) [[slides]](https://www.jianguoyun.com/p/DVRRLHUQq7ftBxjG3Y8C)
- **Pre-trained Models for Natural Language Processing: A Survey**. *Xipeng Qiu, Tianxiang Sun, Yige Xu, Yunfan Shao, Ning Dai, Xuanjing Huang*. (Invited Review of Science China Technological Sciences 2020) [[paper]](https://arxiv.org/pdf/2003.08271.pdf)
- **Embeddings in Natural Language Processing**. *Mohammad Taher Pilehvar, Jose Camacho-Collados*. (2020) [[book]](http://josecamachocollados.com/book_embNLP_draft.pdf)

### Repository
- [bojone / albert_zh](https://github.com/bojone/albert_zh) - 转换brightmart版的albert权重到Google版格式
- [bojone / bert4keras](https://github.com/bojone/bert4keras) - bojone's (苏神) BERT Keras implementation
- [brightmart / albert_zh](https://github.com/brightmart/albert_zh) - 海量中文预训练ALBERT模型
- [brightmart / roberta_zh](https://github.com/brightmart/roberta_zh) - RoBERTa中文预训练模型
- [CyberZHG / keras-bert](https://github.com/CyberZHG/keras-bert) - CyberZHG's BERT Keras implementation
- [Ethan-yt / GuwenBERT](https://github.com/Ethan-yt/guwenbert) - A Pre-trained Language Model for Classical Chinese (Literary Chinese)
- [graykode / gpt-2-Pytorch](https://github.com/graykode/gpt-2-Pytorch) - Simple Text-Generator with OpenAI gpt-2 Pytorch Implementation
- [hanxiao / bert-as-service](https://github.com/hanxiao/bert-as-service) - Using BERT model as a sentence encoding service
- [heartcored98 / Transformer_Anatomy](https://github.com/heartcored98/transformer_anatomy) - Toolkit for finding and analyzing important attention heads in transformer-based models
- [Hironsan / bertsearch](https://github.com/Hironsan/bertsearch) - Elasticsearch with BERT for advanced document search
- [CLUEbenchmark / CLUE](https://github.com/CLUEbenchmark/CLUE) - Chinese Language Understanding Evaluation Benchmark
- [jessevig / bertviz](https://github.com/jessevig/bertviz) - BERT Visualization Tool
- [Jiakui / awesome-bert](https://github.com/Jiakui/awesome-bert) - Collect BERT related resources
- [Morizeyao / GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese) - Chinese version of GPT2 training code, using BERT tokenizer
- [Separius / BERT-keras](https://github.com/Separius/BERT-keras) - Separius' BERT Keras implementation
- [policeme / roberta-wwm-base-distill](https://github.com/policeme/roberta-wwm-base-distill) - A chinese Roberta wwm distillation model which was distilled from roberta-ext-wwm-large
- [terrifyzhao / bert-utils](https://github.com/terrifyzhao/bert-utils) - One line generate BERT's sent2vec for classification or matching task
- [Tencent / TurboTransformers](https://github.com/Tencent/TurboTransformers) - A fast and user-friendly runtime for transformer inference on CPU and GPU
- [THUNLP / OpenCLaP](https://github.com/thunlp/OpenCLaP) - Open Chinese Language Pre-trained Model Zoo
- [THUNLP / PLMpapers](https://github.com/thunlp/PLMpapers) - Must-read Papers on pre-trained language models.
- [THUNLP-AIPoet / BERT-CCPoem](https://github.com/THUNLP-AIPoet/BERT-CCPoem) - A BERT-based pre-trained model particularly for Chinese classical poetry
- [tomohideshibata / BERT-related-papers](https://github.com/tomohideshibata/BERT-related-papers) - This is a list of BERT-related papers.
- [TsinghuaAI / CPM-Generate](https://github.com/TsinghuaAI/CPM-Generate) - Chinese Pre-Trained Language Models (CPM-LM) Version-I
- [valuesimplex / FinBERT](https://github.com/valuesimplex/FinBERT)
- [ymcui / Chinese-ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) - Pre-trained Chinese ELECTRA（中文ELECTRA预训练模型）
- [ymcui / Chinese-XLNet](https://github.com/ymcui/Chinese-XLNet) - Pre-Trained Chinese XLNet（中文XLNet预训练模型）
- [ZhuiyiTechnology / pretrained-models](https://github.com/ZhuiyiTechnology/pretrained-models) - Open Language Pre-trained Model Zoo
- [ZhuiyiTechnology / SimBERT](https://github.com/ZhuiyiTechnology/simbert) - A bert for retrieval and generation
- [ZhuiyiTechnology / WoBERT](https://github.com/ZhuiyiTechnology/WoBERT)

### Chinese Blog
- [Andy Yang / BERT 瘦身之路：Distillation，Quantization，Pruning](https://zhuanlan.zhihu.com/p/86900556)
- [RUC AI Box / BERT meet Knowledge Graph：预训练模型与知识图谱相结合的研究进展](https://mp.weixin.qq.com/s/3hAm2_Lxgf0o7BjRd7UxIA)
- [阿里 / BERT蒸馏在垃圾舆情识别中的探索](https://mp.weixin.qq.com/s/ljYPSK20ce9EoPbfGlaCrw)
- [科学空间 / 从语言模型到Seq2Seq：Transformer如戏，全靠Mask](https://kexue.fm/archives/6933)
- [科学空间 / 抛开约束，增强模型：一行代码提升albert表现](https://spaces.ac.cn/archives/7187)
- [科学空间 / BERT-of-Theseus：基于模块替换的模型压缩方法](https://kexue.fm/archives/7575)
- [科学空间 / 提速不掉点：基于词颗粒度的中文WoBERT](https://kexue.fm/archives/7758)
- [科学空间 / 必须要GPT3吗？不，BERT的MLM模型也能小样本学习](https://spaces.ac.cn/archives/7764)
- [科学空间 / 用ALBERT和ELECTRA之前，请确认你真的了解它们](https://kexue.fm/archives/7846)
- [李理 / XLNet原理](http://fancyerii.github.io/2019/06/30/xlnet-theory/)
- [李如 / ELECTRA: 超越BERT, 19年最佳NLP预训练模型](https://zhuanlan.zhihu.com/p/89763176)
- [李如 / 谈谈我对ELECTRA源码放出的看法](https://zhuanlan.zhihu.com/p/112813856)
- [李如 / BERT蒸馏完全指南｜原理/技巧/代码](https://mp.weixin.qq.com/s/3dy1OmXIpqVrfggsF0RPXg)
- [李如 / BERT-flow｜CMUx字节推出简单易用的文本表示新SOTA！](https://mp.weixin.qq.com/s/MavPYIEO2xxXPjvDl104Fw)
- [刘群 / 预训练语言模型研究进展和趋势展望](https://mp.weixin.qq.com/s/yQV8WK2ZfWt4FGEZ1ICCCQ)
- [美团 / 结合业务场景案例实践分析，倾囊相授美团BERT的探索经验](https://mp.weixin.qq.com/s/AU2_UtbcWBsY0zKudATpPw) [[video]](https://www.bilibili.com/video/BV1vC4y147px)
- [Microsoft / 8篇论文梳理BERT相关模型进展与反思](https://www.msra.cn/zh-cn/news/features/bert)
- [NLP有品 / 关于BERT：你不知道的事](https://mp.weixin.qq.com/s/OujaLboNbf7CAHnHQdXHYA)
- [sliderSun / Transformer和Bert相关知识解答](https://zhuanlan.zhihu.com/p/149634836)
- [Tobias Lee / BERT 的演进和应用](https://mp.weixin.qq.com/s/u4k-A3dSb2-6PDodWPePhA)
- [腾讯 / 内存用量1/20，速度加快80倍，腾讯QQ提出全新BERT蒸馏框架，未来将开源](https://mp.weixin.qq.com/s/W668zeWuNsBKV23cVR0zZQ)
- [小米 / BERT适应业务遇难题？这是小米NLP的实战探索](https://mp.weixin.qq.com/s/XnKfqm-bj9tbPqf2lZJc0A)
- [邱震宇 / 模型压缩实践收尾篇——模型蒸馏以及其他一些技巧实践小结](https://zhuanlan.zhihu.com/p/124215760)
- [夕小瑶的卖萌屋 / 超一流 | 从XLNet的多流机制看最新预训练模型的研究进展](https://mp.weixin.qq.com/s/VfytCWa-h8CmUZW1RWAdnQ)
- [夕小瑶的卖萌屋 / 如何优雅地编码文本中的位置信息？三种positioanl encoding方法简述](https://mp.weixin.qq.com/s/ENpXBYQ4hfdTLSXBIoF00Q)
- [许维 / 深度长文：NLP的巨人肩膀（上）](https://mp.weixin.qq.com/s/Rd3-ypRYiJObi-e2JDeOjQ)
- [许维 / 深度长文：NLP的巨人肩膀（下）](https://mp.weixin.qq.com/s/7imMQ3GkD52xP7N4fqNPog)
- [张俊林 / 从Word Embedding到BERT模型—自然语言处理中的预训练技术发展史](https://zhuanlan.zhihu.com/p/49271699)
- [张俊林 / BERT时代的创新（应用篇）：Bert在NLP各领域的应用进展](https://zhuanlan.zhihu.com/p/68446772)
- [张俊林 / BERT时代的创新：BERT应用模式比较及其它](https://zhuanlan.zhihu.com/p/65470719)
- [张俊林 / BERT和Transformer到底学到了什么 | AI ProCon 2019](https://zhuanlan.zhihu.com/p/90167937)
- [张俊林 / 效果惊人的GPT 2.0模型：它告诉了我们什么](https://zhuanlan.zhihu.com/p/56865533)
- [张俊林 / XLNet:运行机制及和Bert的异同比较](https://zhuanlan.zhihu.com/p/70257427)
- [张俊林 / 乘风破浪的PTM：两年来预训练模型的技术进展](https://zhuanlan.zhihu.com/p/254821426)
- [张正 / 从基础到前沿看迁移学习在NLP中的演化](https://mp.weixin.qq.com/s/Xsh3VNLYCxqh5TH_mK1uXQ)
- [知乎问答 / BERT为何使用学习的position embedding而非正弦position encoding?](https://www.zhihu.com/question/307293465/answer/1039311514)

### English Blog
- [A Fair Comparison Study of XLNet and BERT with Large Models](https://medium.com/@xlnet.team/a-fair-comparison-study-of-xlnet-and-bert-with-large-models-5a4257f59dc0)
- [All The Ways You Can Compress BERT](http://mitchgordon.me/machine/learning/2019/11/18/all-the-ways-to-compress-BERT.html)
- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)
- [Rasa / Compressing BERT for faster prediction](https://blog.rasa.com/compressing-bert-for-faster-prediction-2/)
