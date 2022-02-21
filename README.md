# Transformer-101-Syllabus
About Transformer Syllabus 101 Chapters

NLP on Transformers 101NLP on Transformers 101

(基于Transformer的NLP智能对话机器人实战课程)

One Architecture， One Course，One Universe

星空智能对话机器人的Gavin认为Transformer是拥抱数据不确定性的艺术。
Transformer的架构、训练及推理等都是在Bayesian神经网络不确定性数学思维下来完成的。Encoder-Decoder架构、Multi-head注意力机制、Dropout和残差网络等都是Bayesian神经网络的具体实现；基于Transformer各种模型变种及实践也都是基于Bayesian思想指导下来应对数据的不确定性；混合使用各种类型的Embeddings来提供更好Prior信息其实是应用Bayesian思想来集成处理信息表达的不确定性、各种现代NLP比赛中高分的作品也大多是通过集成RoBERTa、GPT、ELECTRA、XLNET等Transformer模型等来尽力从最大程度来对抗模型信息表示和推理的不确定性。
从数学原理的角度来说，传统Machine Learning及Deep learning算法训练的目标函数一般是基于Naive Bayes数学原理下的最大似然估计MLE和最大后验概率MAP来实现，其核心是寻找出最佳的模型参数；而Bayesian的核心是通过计算后验概率Posterior的predictive distribution，其通过提供模型的不确定来更好的表达信息及应对不确定性。对于Bayesian架构而言，多视角的先验概率Prior知识是基础，在只有小数据甚至没有数据的时候是主要依赖模型Prior概率分布(例如经典的高斯分布)来进行模型推理，随着数据的增加，多个模型会不断更新每个模型的参数来更加趋近真实数据的模型概率分布；与此同时，由于（理论上）集成所有的模型参数来进行Inference，所以Bayesian神经网络能够基于概率对结果的提供基于置信度Confidence的分布区间，从而在各种推理任务中更好的掌握数据的不确定性。
当然，由于Bayesian模型因为昂贵的CPU、Memory及Network的使用，在实际工程实践中计算Bayesian神经网络中所有概率模型分布P(B)是棘手的甚至是Intractable的几乎不能实现事情，所以在工程落地的时候会采用Sampling技术例如MCMC的Collapsed Gibbs Sampling、Metropolis Hastings、Rejection Sampling及Variational Inference的Mean Field及Stochastic等方法来降低训练和推理的成本。Transformer落地Bayesian思想的时候权衡多种因素而实现最大程度的近似估计Approximation，例如使用了计算上相对CNN、RNN等具有更高CPU和内存使用性价比的Multi-head self-attention机制来完成更多视角信息集成的表达，在Decoder端训练时候一般也会使用多维度的Prior信息完成更快的训练速度及更高质量的模型训练，在正常的工程落地中Transformer一般也会集成不同来源的Embeddings，例如星空智能对话机器人的Transformer实现中就把One-hot encoding、Word2vec、fastText、GRU、BERT等encoding集成来更多层级和更多视角的表达信息。
拥抱数据不确定性的Transformer基于Bayesian下共轭先验分布conjugate prior distribution等特性形成了能够整合各种Prior知识及多元化进行信息表达、及廉价训练和推理的理想架构。理论上讲Transformer能够更好的处理一切以 “set of units” 存在的数据，而计算机视觉、语音、自然语言处理等属于这种类型的数据，所以理论上讲Transformer会在接下来数十年对这些领域形成主导性的统治力。

*****************************************************************************

贝叶斯神经网络（Bayesian Neural Network）通过提供不确定来回答“Why Should I Trust You？”这个问题。实现上讲，贝叶斯通过集成深度学习参数矩阵中参数的Uncertainty来驾驭数据的不确定性，提供给具体Task具有置信空间Confidence的推理结构。

一般的神经网络我们称为Point estimation neural networks，通过MLE最大似然估计的方式建立训练的目标函数，为神经网络中的每个参数寻找一个optimal最优值；而贝叶斯深度学习一种把概率分布作为权重的神经网络，通过真实数据来优化参数的概率分布，在训练的过程中会使用MAP最大后验概率集成众多的模型参数的概率分布来拟合各种不确定的情况，提供处理数据不确定性的信息表达框架。

Transformer是一个符合Bayesian深度学习网络的AI架构，尤其是其经典的multi-head self-attention机制，该机制其实采用模型集成的思想来从工程角度落地贝叶斯深度学习网络；基于Prior先验信息的正则化效果，multi-head机制所表达的信息多元化及不确定性能够提供具有高置信度区间的回答 “Why Should I Trust You？” 这一问题

**********************************************************************************************************

“How many heads are enough in Transformer？” 是持续困扰NLP研究人员和从业者的核心问题。人们对multi-head attention内部机制理解的匮乏导致使用Transformer时候产生无处安放的焦虑：一方面，以multi-head attention为核心的Transformer已经成为了NLP全新一代的技术引擎；另一方面，人们因为缺乏对其内部真相的理解而产生日益增长的紧张。

Multi-head attention机制目的是通过“capture different attentive information”来提升AI模型信息表达能力；从实现上讲，Transformer通过联合使用multi-head矩阵的subspace来从不同的视角perspective更多维度的表达input data。从贝叶斯神经网络的角度来说，Multi-head attention机制其实是一种Sampling技术，每个head其实是一个sample。更多的有区分度的sample会使得整个Transformer能够提供更好的后验概率分布posterior distribution对数据进行Approximation。那么，在不考虑overfitting并抛开CPU、Memory、Network等硬件限制的情况下，是不是要把head的数量设置到无限大呢？毕竟，从数学的角度讲，无限多sample意味我们可以通过微积分来完美的表达贝叶斯中的P(B)；从工程实践的角度来说，更多的head意味着更多的Sample，而更多高质量的sample可以帮助模型提供更加精确的approximations。

要回答这个问题，需要思考Transformer模型训练的目标：使得以multi-head attention为代表的Encoder-Decoder模型无限逼近target posterior distribution。而这正是问题的来源，因为head作为sample不是完全从数据中产生，而是通过具有Trainable参数的linear transformation基于input data而来，并且head是采用vector的方式进行离散discrete信息表达而非continuous类型，所以就导致了head为代表的sample并不是真正来自target distribution，并且不能够表达连续性概率分布。这些gap或者discrepancy导致了更多的head意味着更多误差的积累，从而导致模型的performance deterioration。因此，在head的数量和模型质量上产生trade-off，也就是说，更多的head有可能会导致更大的模型误差。



本课程以Transformer架构为基石、萃取NLP中最具有使用价值的内容、围绕手动实现工业级智能业务对话机器人所需要的全生命周期知识点展开，学习完成后不仅能够从算法、源码、实战等方面融汇贯通NLP领域NLU、NLI、NLG等所有核心环节，同时会具备独自开发业界领先智能业务对话机器人的知识体系、工具方法、及参考源码，成为具备NLP硬实力的业界Top 1%人才。

课程特色：
  101章围绕Transformer而诞生的NLP实用课程
  5137个围绕Transformers的NLP细分知识点
  大小近1200个代码案例落地所有课程内容
  10000+行纯手工实现工业级智能业务对话机器人
  在具体架构场景和项目案例中习得AI相关数学知识
  以贝叶斯深度学习下Attention机制为基石架构整个课程
  五大NLP大赛全生命周期讲解并包含比赛的完整代码实现

<details>
<summary>第1章: 贝叶斯理论下的Transformer揭秘</summary>
<br>
<pre>
1，基于Bayesian Theory，融Hard Attention、Soft Attention、Self-Attention、Multi-head Attention于一身的Transformer架构
2，为什么说抛弃了传统模型（例如RNN、 LSTM、CNN等）的Transformer拉开了非序列化模型时代的序幕？
3，为什么说Transformer是预训练领域底层通用引擎？
4，Transformer的Input-Encoder-Decoder-Output模型组建逐一剖析
5，Transformer中Encoder-Decoder模型进行Training时候处理Data的全生命周期七大步骤揭秘
6，Transformer中Encoder-Decoder模型进行Inference时候处理Data的全生命周期六大步骤详解
7，Teacher Forcing数学原理及在Transformer中的应用
8，穷根溯源：为何Scaled Dot-Product Attention是有效的？
9，透视Scaled Dot-Product Attention数据流全生命周期
10，穷根溯源：Queries、Keys、Values背后的Trainable矩阵揭秘
11，当Transformer架构遇到Bayesian理论：multi-head attention
12，End-to-end Multi-head attention的三种不同实现方式分析
13，透视Multi-head attention全生命周期数据流
14，Transformer的Feed-Forward Networks的两种实现方式：Linear Transformations和Convolutions
15，Embeddings和Softmax参数共享剖析
16，Positional Encoding及Positional Embedding解析
17，Sequence Masking和Padding Masking解析
18，Normal distribution、Layer Normalization和Batch Normalization解析
19，Transformer的Optimization Algorithms数学原理、运行流程和最佳实践
20，Learning rate剖析及最佳实践
21，从Bayesian视角剖析Transformer中的Dropout及最佳实践
22，Label Smoothing数学原理和工程实践解析
23，Transformer背后的驱动力探讨
</pre>
</details>

<details>
<summary>第2章: 通过30+个细分模块完整实现Transformer论文源码及项目调试</summary>
<br>
<pre>
1，Transformer源码训练及预测整体效果展示
	2，模型训练model_training.py代码完整实现
	3，数据预处理data_preprocess.py代码完整实现
	4，Input端Embeddings源码完整实现
	5，Attention机制attention.py代码完整实现
	6，Multi-head Attention机制multi_head_attention.py代码完整实现
	7，Position-wise Feed-forward源码完整实现
	8，Masking 在Encoder和Decoder端的源码完整实现0
	9，SublayerConnection源码完整实现
	10，Encoder Layer源码完整实现
	11，LayerNormalization源码完整实现
	12，DecoderLayer源码完整实现
	13，Encoder Stack源码完整实现
	14，Decoder Stack源码完整实现
	15，由Memory链接起来的EncoderDecoder Module源码完整实现
	16，Batch操作完整源码实现
	16，Optimization源码完整实现
	17，Loss计算数学原理及完整源码实现
	18，Output端Generator源码完整实现
	19，Transformer模型初始化源码及内幕揭秘
	20， Label Smoothing源码完整实现
	21，Training源码完整实现
22，Greedy Decoding源码及内幕解析
	23，Tokenizer源码及调试
	24，Multi-GPU训练完整源码
27，使用自己实现的Transformer完成分类任务及调试
	28，Transformer翻译任务代码完整实现及调试
	29，BPE解析及源码实现
	30，Shared Embeddings解析及源码实现
	31，Beam Search解析及源码实现
	32，可视化Attention源码实现及剖析
</pre>
</details>

<details>
<summary>第3章: 细说Language Model内幕及Transformer XL源码实现</summary>
<br>
<pre>
	1，人工智能中最重要的公式之一MLE数学本质剖析及代码实战
	2，Language Model的数学原理、Chain Rule剖析及Sparsity问题
	3，Markov Assumption：first order、second order、third order剖析
	4，Language Model：unigram及其问题剖析、bigram及依赖顺序、n-gram
	5，使用Unigram训练一个Language Model剖析及实践
	6，使用Bigram训练一个Language Model剖析及实践
	7，使用N-gram训练一个Language Model剖析及实践
	8，拼写纠错案例实战：基于简化后的Naive Bayes的纠错算法详解及源码实现
	9，使用基于Average Log Likelihood的PPL(Perplexity)来评估Language Model
	10，Laplace Smoothing剖析及基于PPL挑选最优化K的具体方法分析
	11，Interpolation Smoothing实现解析：加权平均不同的N-gram概率
	12，Good-Turning Smoothing算法解析
	13，Vallina Transformer language model处理长文本架构解析
	14， Vallina Transformer Training Losses：Multiple Postions Loss、Intermediate Layer Losses、Multiple Targets Losses
	15，Vallina Transformer的三大核心问题：Segment上下文断裂、位置难以区分、预测效率低下
	16，Transformer XL：Attentive Language Models Beyond a Fixed-Length Context
	17，Segment-level Recurrence with State Reuse数学原理及实现分析
	18，Relative Positional Encoding算法解析
	19，Transformer XL 中降低矩阵运算复杂度的Trick解析
	20，缓存机制在语言模型中的使用思考
	21，Transformer XL之数据预处理完整源码实现及调试
	22，Transformer XL之MemoryTransformerLM完整源码实现及调试
	23，Transformer XL之PartialLearnableMultiHeadAttention源码实现及调试
	24，Transformer XL之PartialLearnableDecoderLayer源码实现及调试
	25，Transformer XL之AdaptiveEmbedding源码实现及调试
	26，Transformer XL之相对位置编码PositionalEncoding源码实现及调试
	27，Transformer XL之Adaptive Softmax解析及源码完整实现
	28，Transformer XL之Training完整源码实现及调试
	29，Transformer XL之Memory更新、读取、维护揭秘
	30，Transformer XL之Unit单元测试
	31，Transformer XL案例调试及可视化
</pre>
</details>

<details>
<summary> 第4章: Autoregressive Language Models之GPT-1、2、3解析及GPT源码实现 </summary>
<br>
<pre>
	1，Task-aware的人工智能Language model + Pre-training + Fine-tuning时代
	2，Decoder-Only Stack数学原理及架构解析
	3，训练材料标注：neutral、contradiction、entailment、multi-label、QA等
	4，NLP(Natural Language Understanding)：Semantic similarity、document classification、textual entailment等
	5，大规模Unsupervised pre-training贝叶斯数学原理及架构剖析
	6，Task-specific Supervised fine-tuning的Softmax及Loss详解
	7，针对Classification、Entailment、Similarity、Mutiple Choice特定任务的Input数据预处理解析及矩阵纬度变化处理
	8，GPT2架构解析：Language Models for unsupervised multitask learners
	9，GPT 2把Layer Norm前置的数据原理剖析
	10，GPT 2 Self-Attention剖析
	11，GPT 2 Training数据流动全生命周期解析
	12，GPT 2 Inference数据流动全生命周期解析
	13，GPT 3 架构剖析：Language Models are Few-Shot Learners
	14，由GPT 3引发的NLP12大规律总结
	15，GPT数据预处理源码完整实现及调试
	16，GPT的BPE实现源码及调试
	17，GPT的TextEncoder源码实现及调试
	18，GPT的Attention完整源码实现及调试
	19，GPT的Layer Normalization完整源码实现及调试
	20，GPT的Feed Foward神经网络通过Convolutions源码实现
	21，GPT的Block源码完整实现及调试
	22，GPT的TransformerModel源码完整实现及调试
	23，GPT的输入LMHead源码完整实现及调试
	24，GPT的MultipleChoiceHead源码完整实现及调试
	25，GPT的语言模型及特定Task的DoubleHeadModel源码完整实现
	26，GPT的OpenAIAdam优化器源码及调试
	27，GPT的LanguageModel loss源码及调试
	28，GPT的MultipleChoiceLoss源码及调试
	29，OpenAI GPT的Pretrained Model的加载使用
	30，GPT模型Task-specific训练完整源码及调试
	31，GPT进行Inference完整源码实现及代码调试
</pre>
</details>

<details>
<summary> 第5章: Autoencoding Language Models数学原理及模型架构解析 </summary>
<br>
<pre>
1，Auto-encoding Language Models通用数学原理详解
2，为何要放弃采用Feature-Based语言模型ELMo而使用Fine-tuning模型？
3，双向语言模型：both left-to-right and right-to-left不同实现及数学原理解析
4，深度双向语言模型背后的数学原理及物理机制
5，Unsupervised Fine-tuning训练模型架构及数学原理解析
6，Transfer Learning数学原理及工程实现详解
7，MLM(Masked Language Models)数学原理及工程架构解析
8，MLM问题解析及解决方案分析
9，Pre-training + Fine-tuning的BERT分层架构体系及组件解析
10，BERT的三层复合Embeddings解析
11，BERT不同模块的参数复杂度分析
12，BERT在进行Masking操作中采用10%随机选取词库的内容进行替换masked位置的内容的数学原理剖析
13，BERT在进行Masking操作中采用10%的内容维持不变的数学原理揭秘
14，BERT的Masking机制五大缺陷及其解决方案分析
15，BERT的Masking机制在Data Enchancement方面的妙用
16，BERT的Masking机制在处理智能对话系统中不规范用语甚至是错误语法及用词的妙用
17，BERT的NSP(Next Sentence Prediction)机制及其实现
18，BERT的NSP三大问题及解决方案剖析
19，BERT的CLS剖析及工程实现
20，BERT的CLS三个核心问题及解决方案
21，Knowledge Distillation for BERT数学原理贝叶斯及KL散度解析及案例实战
22，使用BERT进行Classification架构及案例实战
23，使用BERT进行NER(Named Entity Recognition)架构及案例实战
24，使用BERT实现文本Similarity任务的架构及案例实战
25，使用BERT实现Question-Answering任务的架构及案例实战
26，ALBERT模型架构解析
27，RoBERTa模型架构解析
28，SpanBERT模型架构解析
29，TinyBERT模型架构解析
30，Sentence-BERT模型架构解析
31，FiBERT模型架构解析
32，K-BERT模型架构解析
33，KG-BERT模型架构解析
</pre>
</details>

<details>
<summary> 第6章: BERT Pre-training模型源码完整实现、测试、调试及可视化分析 </summary>
<br>
<pre>
	1，词典Vocabulary库构建多层级源码实现及测试
	2，Dataset加载及数据处理源码完整实现及测试和调试
	3，Next Sentence Prediction机制源码完整实现及测试
	4，Masked Language Model机制中80%词汇Masking源码实现
	5，Masked Language Model机制中10%词汇随机替换和10%词汇保持不变源码实现
	6，Masked Language Model机制下的Output Label操作源码实现
	7，加入CLS、SEP 等Tokens
	8，Segment Embeddings源码实现
	9，Padding源码实现及测试
	10，使用DataLoader实现Batch加载
	11，BERT的初始化init及forward方法源码实现
	12，PositionalEmbeddings源码实现详解
	13，TokenEmbeddings源码
	14，SegmentEmbeddings源码
	15，BERTEmbeddings层源码实现及调试
	16，基于Embeddings之多Linear Transformation操作
	17，Queries、Keys、Values操作源码
	18，Attention机制源码实现
	19，Multi-head Attention源码实现
	20，Layer Normalization数学原理及源码实现
	21，Sublayer Connection源码实现
	22，Position-wise Feedforward层源码实现
	23，Dropout数学机制及源码实现
	24，基于Embeddings之上的Linear Transformation及其不同源码实现方式
	25，TransformerBlock源码完整实现及测试
	26，BERT模型训练时候多二分类和多分类别任务数学原理和实现机制
	26，BERT Training Task之MLM源码完整实现及测试
	27，BERT Training Task之NSP源码完整实现及测试
	28，Negative Sampling数学原理及实现源码
	29，MLM和NSP的Loss计算源码实现
	30，BERT模型的训练源码实现及测试
	31，使用小文本训练BERT模型源码、测试和调试
	32，使用特定领域的(例如医疗、金融等)来对BERT进行Pre-training最佳实践
	33，BERT加速训练技巧：动态调整Attention的Token能够Attending的长度
	34，BERT可视化分析
</pre>
</details>

<details>
<summary> 第7章: BERT Fine-tuning源码完整实现、调试及案例实战 </summary>
<br>
<pre>
	1，数据预处理训练集、测试集源码
	2，文本中的Token、Mask、Padding的预处理源码
	3，数据的Batch处理实现源码及测试
	4，加载Pre-training模型的BertModel及BertTokenizer
	5，模型Config配置
	6，Model源码实现、测试、调试
	7，BERT Model微调的数学原理及工程实践
	8，BERT Model参数Frozen数学原理及工程实践
	9，BertAdam数学原理及源码剖析
	10，训练train方法源码详解
	11，fully-connected neural network层源码详解及调试
	12，采用Cross-Entropy Loss Function数学原理及代码实现
	13，Evaluation 指标解析及源码实现
	14，Classification任务下的Token设置及计算技巧
	15，适配特定任务的Tokenization解析
	16，BERT + ESIM(Enhanced Sequential Inference Model)强化BERT模型
	17，使用BERT + LSTM整合强化BERT 模型
	18，基于Movie数据的BERT Fine-tuning案例完整代码实现、测试及调试
</pre>
</details>

<details>
<summary> 第8章: 轻量级ALBERT模型剖析及BERT变种中常见模型优化方式详解 </summary>
<br>
<pre>
	1，从数学原理和工程实践的角度阐述BERT中应该设置Hidden Layer的维度高于(甚至是高几个数量级)Word Embeddings的维度背后的原因
	2，从数学的角度剖析Neural Networks参数共享的内幕机制及物理意义
	3，从数学的角度剖析Neural Networks进行Factorization的机制及物理意义
	4，使用Inter-sentence coherence任务进行模型训练的的数学原理剖析
	5，上下文相关的Hidden Layer Embeddings
	6，上下午无关或不完全相关的Word Embeddings
	7，ALBERT中的Factorized embedding parameterization剖析
	8，ALBERT中的Cross-Layer parameter sharing机制：只共享Attention参数
	9，ALBERT中的Cross-Layer parameter sharing机制：只共享FFN参数
	10，ALBERT中的Cross-Layer parameter sharing机制：共享所有的参数
	11，ALBERT不同Layers的Input和Output相似度分析
	12，训练Task的复杂度：分离主题预测和连贯性预测的数学原因及工程实践
	13，ALBERT中的不同于BERT的 Sentence Negative Sampling
	14，句子关系预测的有效行分析及问题的底层根源
	15，ALBERT的SOP(Sentence Order Prediction)实现分析及工程实践
	16，ALBERT采用比BERT更长的注意力长度进行实际的训练
	17，N-gram Masking LM数学原理和ALERT对其实现分析
	18，采用Quantization优化技术的Q8BERT模型架构解析
	19，采用Truncation优化技术的“Are Sixteen Heads Really Better than One?”模型架构解析
	20，采用Knowledge Distillation优化技术的distillBERT模型架构解析
	21，采用多层Loss计算+知识蒸馏技术的TinyBERT模型架构解析
	22，由轻量级BERT带来的关于Transformer网络架构及实现的7点启示
</pre>
</details>

<details>
<summary> 第9章: ALBERT Pre-training模型及Fine-tuning源码完整实现、案例及调试 </summary>
<br>
<pre>
1，Corpus数据分析
	2，Pre-training参数设置分析
	3，BasicTokenizer源码实现
	4，WordpieceTokenizer源码实现
	5，ALBERT的Tokenization完整实现源码
	6，加入特殊Tokens CLS和SEP
	7，采用N-gram的Masking机制源码完整实现及测试
	8，Padding操作源码
	9，Sentence-Pair数据预处理源码实现
	10，动态Token Length实现源码
	11，SOP正负样本源码实现
	12，采用了Factorization的Embeddings源码实现
	13，共享参数Attention源码实现
	14，共享参数Multi-head Attention源码实现
	15，LayerNorm源码实现
	16，共享参数Position-wise FFN源码实现
	17，采用GELU作为激活函数分析
	18，Transformer源码完整实现
	19，Output端Classification和N-gram Masking机制的Loss计算源码
	20，使用Adam进行优化源码实现
	21，训练器Trainer完整源码实现及调试
	22，Fine-tuning参数设置、模型加载
	23，基于IMDB影视数据的预处理源码
	24，Fine-tuning阶段Input Embeddings实现源码
	25，ALBERT Sequence Classification参数结构总结
	26，Fine-tuning 训练代码完整实现及调试
	27，Evaluation代码实现
	28，对Movie数据的分类测试及调试
</pre>
</details>

<details>
<summary> 第10章: 明星级轻量级高效Transformer模型ELECTRA: 采用Generator-Discriminator的Text Encoders解析及ELECTRA模型源码完整实现 </summary>
<br>
<pre>
	1，GAN：Generative Model和Discriminative Model架构解析
	2，为什么说ELECTRA是NLP领域轻量级训练模型明星级别的Model？
	3，使用replaced token detection机制规避BERT中的MLM的众多问题解析
	4，以Generator-Discriminator实现的ELECTRA预训练架构解析
	5，ELECTRTA和GAN的在数据处理、梯度传播等五大区别
	6，ELECTRA数据训练全生命周期数据流
	7，以Discriminator实现Fine-tuning架构解析
	8，ELECTRA的Generator数学机制及内部实现详解
	9，Generator的Loss数学机制及实现详解
	10，Discriminator的Loss数学机制及实现详解
	11，Generator和Discriminator共享Embeddings数据原理解析
	12，Discriminator网络要大于Generator网络数学原理及工程架构
	13，Two-Stage Training和GAN-style Training实验及效果比较
	14，ELECTRA数据预处理源码实现及测试
	15，Tokenization源码完整实现及测试
	16，Embeddings源码实现
	17，Attention源码实现
	18，借助Bert Model实现Transformer通用部分源码完整实现
	19，ELECTRA Generator源码实现
	20，ELECTRA Discriminator源码实现
	21，Generator和Discriminator相结合源码实现及测试
	22，pre-training训练过程源码完整实现
	23，pre-training数据全流程调试分析
	24，聚集于Discriminator的ELECTRA的fine-tuning源码完整实现
	25，fine-tuning数据流调试解析
	26，ELECTRA引发Streaming Computations在Transformer中的应用思考
</pre>
</details>

<details>
<summary> 第11章: 挑战BERT地位的Autoregressive语言模型XLNet剖析及源码完整实现 </summary>
<br>
<pre>
	1，作为Autoregressive语言模型的XLNet何以能够在发布时在20个语言任务上都能够正面挑战作为Autoencoding与训练领域霸主地位的BERT？
	2，XLNet背后Permutation LM及Two-stream self-attention数学原理解析
	3，Autoregressive LM和Autoencoding LM数学原理及架构对比
	4，Denoising autoencoding机制的数学原理及架构设计
	5，对Permutation进行Sampling来高性价比的提供双向信息数学原理
	6，XLNet的Permutation实现架构和运行流程：content stream、query stream
	7，XLNet中的缓存Memory记录前面Segment的信息
	8，XLNet中content stream attention计算
	9，XLNet中query stream attention计算
	10，使用Mask Matrices来实现Two-stream Self-attention
	11，借助Transformer-XL 来编码relative positional 信息
	12，XLNet源码实现之数据分析及预处理
	13，XLNet源码实现之参数设定
	14，Embeddings源码实现
	15，使用Mask实现causal attention
	16，Relative shift数学原理剖析及源码实现
	17，XLNet Relative attention源码完整实现
	18，content stream源码完整实现
	19，queery stream源码完整实现
	20，Masked Two-stream attention源码完整实现
	21，处理长文件的Fixed Segment with No Grad和New Segment
	22，使用einsum进行矩阵操作
	23，XLNetLayer源码实现
	24，Cached Memory设置
	25，Head masking源码
	26，Relative-position encoding源码实现
	27，Permutation实现完整源码
	28，XLNet FFN源码完整实现
	29，XLNet源码实现之Loss操作详解
	30，XLNet源码实现之training过程详解
	31，从特定的checkpoint对XLNet进行re-training操作
	32，Fine-tuning源码完整实现
	33，Training Evaluation分析
	34，使用XLNet进行Movies情感分类案例源码、测试及调试
</pre>
</details>

<details>
<summary> 第12章：NLP比赛的明星模型RoBERTa架构剖析及完整源码实现 </summary>
<br>
<pre>
1，为什么说BERT模型本身的训练是不充分甚至是不科学的？
2，RoBERTa去掉NSP任务的数学原理分析
3，抛弃了token_type_ids的RoBERTa
	4，更大的mini-batches在面对海量的数据训练时是有效的数学原理解析
	5，为何更大的Learning rates在大规模数据上会更有效？
	6，由RoBERTa对hyperparameters调优的数学依据
	7，RoBERTa下的byte-level BPE数学原理及工程实践
	6，RobertaTokenizer源码完整实现详解
	7，RoBERTa的Embeddings源码完整实现
	8，RoBERTa的Attention源码完整实现
	9，RoBERTa的Self-Attention源码完整实现
	10，RoBERTa的Intermediate源码完整实现
	11，RobertLayer源码完整实现
	12，RobertEncoder源码完整实现
	13，RoBERTa的Pooling机制源码完整实现
	14，RoBERTa的Output层源码完整实现
	15，RoBERTa Pre-trained model源码完整实现
	16，RobertaModel源码完整实现详解
	17，实现Causal LM完整源码讲解
	18，RoBERTa中实现Masked LM完整源码详解
	19，RobertLMHead源码完整实现
	20，RoBERTa实现Sequence Classification完整源码详解
	21，RoBERTa实现Token Classification完整源码详解
	22，RoBERTa实现Multiple Choice完整源码详解
	23，RoBERTa实现Question Answering完整源码详解
</pre>
</details>

<details>
<summary> 第13章：DistilBERT：smaller, faster, cheaper and lighter的轻量级BERT架构剖析及完整源码实现 </summary>
<br>
<pre>
	1，基于pretraining阶段的Knowledge distillation
	2，Distillation loss数学原理详解
	3，综合使用MLM loss、distillation loss、cosine embedding loss
	4，BERT Student architecture解析及工程实践
	5，抛弃了BERT的token_type_ids的DistilBERT
	6，Embeddings源码完整实现
	7，Multi-head Self Attention源码完整实现
	8，Feedforward Networks源码完整实现
	9，TransformerBlock源码完整实现
	10，Transformer源码完整实现
	11，继承PreTrainedModel的DistilBertPreTrainedModel源码完整实现
	13，DistilBERT Model源码完整实现
	14，DistilBertForMaskedLM源码完整实现
	15，DistilBert对Sequence Classification源码完整实现
</pre>
</details>

<details>
<summary> 第14章: Transformers动手案例系列</summary>
<br>
<pre>
	1，动手案例之使用Transformers实现情感分析案例代码、测试及调试
	2，动手案例之使用Transformers实现NER代码、测试及调试
	3，动手案例之使用Transformers实现闲聊系统代码、测试及调试
	4，动手案例之使用Transformers实现Summarization代码、测试及调试
	5，动手案例之使用Transformers实现Answer Span Extraction代码、测试及调试
	6，动手案例之使用Transformers实现Toxic Language Detection Multi-label Classification代码、测试及调试
	7，动手案例之使用Transformers实现Zero-shot learning代码、测试及调试
	8，动手案例之使用Transformers实现Text Clustering代码、测试及调试
	9，动手案例之使用Transformers实现semantics search代码、测试及调试
	10，动手案例之使用Transformers实现IMDB分析代码、测试及调试
	11，动手案例之使用Transformers实现cross-lingual text similarity代码、测试及调试

</pre>
</details>

第12课 基于Transformer的多轮对话系统四要素解密
1，Intent预测与管理
2，对话管理State Tracking
3，对话行为的预测，根据现在和之前的对话预测接下来的情况
4，Response Selection策略

第13课 如何使用Transformer构建具有抗干扰能力的面向任务的对话系统？
1，使用Dialogue Stacks和LSTM来处理subdialogue的潜在问题分析
2， Transformer实现扛干扰的多轮对话系统架构剖析
3， Transformer实现扛干扰的多轮对话关键技术剖析

第14课 使用Transformer构建具有抗干扰能力的对话系统Experiments深度剖析
1，针对Sub-dailogues进行抗干扰处理实验分析
2，与LSTM进行对话处理试验对比
3，通过Modular Training进行试验分析

第15课 基于多任务Transformer架构的Intent和NER算法实现
1，对话机器人中的Modular Approach的架构剖析
2，经典的处理Intent和NER multi-task Transformer架构分析
3，多任务架构思考与总结

第16课 基于Transformer的轻量级多任务NLU系统解密
1，Transformer处理Intent和NER的Input Emeddings架构解析
2，CLS和MASK的特殊实现解密
3，LOSS计算背后的数学原理详解

第17课 轻量级多任务Transformer语言理解框架DIET试验分析
1，意图识别和NER的联合训练
2，多种Embeddings模型的整合
3，与BERT的对比分析

第18课 基于Transformer端到端的任务对话系统解密
1，Task-Oriented Dialogue与用户交互过程解析
2，SimpleTOD模型架构详解
3，SimpleTOD端到端的任务对话系统训练函数剖析

第19课 基于Transformer的端到端对话系统SimpleTOD试验分析
1，SimpleTOD端到端任务对话系统运行流程回顾
2，Special Tokens设置及其重大影响
3，SimpleTOD在多场景下的试验分析

第20课 基于Transformer的Scalable对话状态管理模型BERT-DST详解 
1，Scalable 对话状态管理系统剖析
2，BERT-DST算法解析
3，BERT-DST试验分析

第21课 细粒度Retrieval-Base对话系统解密
1，Fine-grainded post-training架构解析
2，实现对话内部更细粒度信息提取
3，实现更精细的Target目标训练

第22课 细粒度Retrieval-Base对话算法详解
1，细粒度Related Work解析
2，Problem Formulation
3，算法内部过程详解

第23课 BERT-FP两大训练任务内幕及Experiment解析
1，Short Context-response Pair Training 解析
2，Utterance Relevance Classification解析
3，Experiments数据集、Beaseline Models、及训练结果分析

第24课Retrieval-Based对话系统BERT-FP的Further Analysis及星空对话机器人内幕实现解密 
1，对BERT-FP的Further Analysis解析
2，星空对话机器人在Data Augmentation等的处理秘密 
3，对话系统数据处理思考与总结 

第25课 基于Transformer轻量级高效精确的Conversational Representation对话系统ConveRT解密
1，为何Gavin认为ConverRT是在超过3000篇NLP论文中排名前五的论文？
2，ConveRT不同于BERT的训练目标及其Compact网络
3，Single-Context ConveRT及Multi-Context ConveRT

第26课 惊才绝艳的基于Transformer的ConveRT算法内幕逐句解密
1，Vocabulary构建及Iinput和Response的Representation算法内幕 
2，Input and Response Encoder Networks算法内幕
3，Input-Response Interaction算法内幕

第27课 基于Transformer的ConveRT算法及试验设置解密
1，ConveRT下的Quantization内幕机制详解
2，Multi-Context ConveRT架构师实现解析
3，ConveRT进行试验的数据及Baselines分析

第28课 基于Transformer的ConveRT的Experiments、Results及Discussion
1，ConveRT中的Transfer Learning
2，low-data settings分析及最佳实践
3，low data发展方向探索

第29课 基于Transformer的Poly-Encoder架构体系解密
1，基于Transformer的Bi-encoder解析
2，基于Transformer的Cross-encoder解析
3，基于Transformer的Poly-encoder解析

第30课 基于Transformer的Poly-Encoder的Tasks和Methods内幕详解
1，Poly-Encoder下的Tasks详解
2，Bi-Encoder及Cross-Encoder的算法详解
3，Poly-Encoder算法实现详解

第31课 基于Transformer的Poly-Encoder实验详解
1，Bi-encoders and Cross-encoders实验详解
2，Poly-encoders实验详解
3，Domain-specific Pre-training 实验详解

第32课 基于Transformer的Recipes for building an open-domain chatbot论文解析
1，论文Abstract详解
2，论文Discussion详解
3，为何Toxic Language及Gender Bias很难解决？

第33课 基于Transformer的Recipes for building an open-domain chatbot架构及策略分析
1，通过两幅图解密Recipes for building an open-domain chatbot架构精髓
2，Blending Skills解析
3，Generation Strategies解析

第34课 基于Transformer的Recipes for building an open-domain chatbot的Generator、Retriever及Objectives
1，Generator、Dialogue Retrieval及Knowledge Retrieval详解
2，Ranking for Retrieval及Likelihood Training for Generation详解
3，α-blending for Retrieve and Refine详解

第35课 基于Transformer的Recipes for building an open-domain chatbot的Decoding、Training及Safety Characteristics分析
1， Unlikehood training及Decoding详解
2，Training Details和Training Data关键点解析
3，Safety Characteristics深度结项

第36课 基于Transformer的Rasa Internals解密之Retrieval Model剖析
1，什么是One Graph to Rule them All
2，为什么工业级对话机器人都是Stateful Computations？
3，Rasa引入Retrieval Model内幕解密及问题解析

第37课 基于Transformer的Rasa Internals解密之去掉对话系统的Intent内幕剖析
1，从inform intent的角度解析为何要去掉intent
2，从Retrieval Intent的角度说明为何要去掉intent
3，从Multi intents的角度说明为何要去掉intent
4，为何有些intent是无法定义的？

第38课 基于Transformer的Rasa Internals解密之去掉对话系统的End2End Learning内幕剖析
1，How end-to-end learning in Rasa works
2，Contextual NLU解析
3，Fully end-to-end assistants

第39课 基于Transformer的Rasa Internals解密之全新一代可伸缩DAG图架构内幕
1，传统的NLU/Policies架构问题剖析
2，面向业务对话机器人的DAG图架构
3，DAGs with Caches解密
4，Example及Migration注意点

第40课 基于Transformer的Rasa Internals解密之定制Graph NLU及Policies组件内幕
1，基于Rasa定制Graph Component的四大要求分析
2，Graph Components解析
3，Graph Components源代码示范

第41课 基于Transformer的Rasa Internals解密之自定义GraphComponent内幕
1，从Python角度分析GraphComponent接口
2，自定义模型的create和load内幕详解
3，自定义模型的languages及Packages支持

第42课 基于Transformer的Rasa Internals解密之自定义组件Persistence源码解析
1，自定义对话机器人组件代码示例分析
2，Rasa中Resource源码逐行解析
3，Rasa中ModelStorage、ModelMetadata等逐行解析

第43课 基于Transformer的Rasa Internals解密之自定义组件Registering源码解析
1，采用Decorator进行Graph Component注册内幕源码分析
2，不同NLU和Policies组件Registering源码解析
3，手工实现类似于Rasa注册机制的Python Decorator全流程实现

第44课 基于Transformer的Rasa Internals解密之自定义组件及常见组件源码解析
1，自定义Dense Message Featurizer和Sparse Message Featurizer源码解析
2，Rasa的Tokenizer及WhitespaceTokenizer源码解析
3，CountVectorsFeaturizer及SpacyFeaturizer源码解析

第45课 基于Transformer的Rasa Internals解密之框架核心graph.py源码完整解析及测试 
1，GraphNode源码逐行解析及Testing分析
2，GraphModelConfiguration、ExecutionContext、GraphNodeHook源码解析
3，GraphComponent源码回顾及其应用源码

第46课 基于Transformer的Rasa Internals解密之框架DIETClassifier及TED
1，作为GraphComponent的DIETClassifier和TED实现了All-in-one的Rasa架构
2，DIETClassifier内部工作机制解析及源码注解分析
3，TED内部工作机制解析及源码注解分析

第47课 基于Transformer的Rasa 3.x Internals解密之DIET近1825行源码剖析
1，DIETClassifier代码解析
2，EntityExtractorMixin代码解析
3，DIET代码解析

第48课 基于Transformer的Rasa 3.x Internals解密之TED Policy近2130行源码剖析
1，TEDPolicy父类Policy代码解析
2，TEDPolicy完整解析
3，继承自TransformerRasaModel的TED代码解析

第49课 基于Transformer的Rasa 3.x 内核解密之UnexpecTEDIntentPolicy架构及实践
1，UnexpecTEDIntentPolicy设计的背后机制
2，UnexpecTEDIntentPolicy与TEDPolicy源码分析
3，UnexpecTEDIntentPolicy与人工服务自定义功能实现

第50课 BERT架构、pretraining预训练、Fine Tuning下游任务微调全生命周期内幕解密
1，BERT架构内幕核心解密
2，BERT Pretraining预训练剖析
3，BERT Fine-tuning解析

第51课 BERT预训练Pre-training源码完整实现
1，构建Dictionary和Data Preprocessing源码
2，BERT神经网络代码实现
3，BERT Language Model代码实现

第52课 BERT Fine-tuning数学原理及案例源码解析
1，Fine-tuning背后数学原理详解
2，Fine-tuning中数据Input处理代码实现
3，Fine-tuning案例代码实现

第53课 UnexpecTEDIntentPolicy源码研读
1，UnexpecTEDIntentPolicy导入包和类分析
2，UnexpecTEDIntentPolicy和TEDPolicy关系分析
3，UnexpecTEDIntentPolicy源码剖析

第54课 UnexpecTEDIntentPolicy算法源码及IntentTED详解
1，UnexpecTEDIntentPolicy算法源码
2，Graph Architecture
3，IntentTED算法及源码

第55课 Rasa Memoization对话策略及源码解析
1，Memoization Policy及Augmented Memoization Policy对话策略分析
2，MemoizationPolicy完整源码解析
3，AugmentedMemoizationPolicy完整源码解析

第56课 Rasa Rule-based Policies架构设计与源码解析
1，Rule Policy内部机制解析
2，InvalidRule源码详解
3，RulePolicy与MemoizationPolicy关系源码详解

第57课 Rasa RulePolicy完整源码详解
1，RulePolicy初始化代码详解
2，RulePolicy训练源码详解
3，RulePolicy Prediction源码详解

第58课 Rasa对话策略Policy完整源码详解
1，Policy与GraphComponent
2，SupportedData完整源码详解
3，PolicyPrediction完整源码详解

第59课 Rasa Policy完整源码详解
1，Policy的初始化及和子类关系源码剖析
2，Policy训练源码详解
3，Policy预测源码详解

第60课 Rasa对话策略Ensemble完整源码剖析
1，Ensemble架构及其在Rasa中的应用解密
2，PolicyPredictionEnsemble源码逐行解析
3，DefaultPolicyPredictionEnsemble源码逐行解析

第61课 Rasa Fallback Classifier处理对话失败情况三大处理方式内幕及代码实战
1，Rasa Fallback Classifier在具体对话机器人开发中的重大价值分析
2，Simple Fallback及Single-stage Fallback处理及代码实现
3，Two-stage Fallback流程分析及代码实现

第62课 Rasa Fallback and Human Handoff全解
1，Out-of-scope消息的处理
2，NLU Fallback的处理
3，Rasa Core Low Action Confidence的处理

第63课 Rasa FallbackClassifier源码逐行剖析
1，FallbackClassifier使用的包及初始化源码解析
2，核心方法process源码逐行解析
3，FallbackClassifier与GraphComponent、IntentClassifier关系源码解析

第64课 Rasa对话机器人业务逻辑Action Servers架构设计与核心运行流程解密
1，Rasa Server与Action Servers交互关系解析
2，请求执行custom action的RESTful中JSON内容详解及示例
3，Action Servers返回的events及responses详解及示例

第65课 Rasa Events剖析及源码详解
1，Event接口分析
2，14大Event剖析及源码详解
3，Loop相关Event分析及源码详解

第66课 Rasa微服务Action自定义及Slot Validation详解
1，Rasa Action剖析及代码示例
2，ValidationAction剖析及代码示例
3，FormValidationAction剖析

第67课 Form全生命周期解析及Default Actions剖析
1，Form全生命周期运行内幕
2，Form的高级用法
3，Default Actions详解

第68课 Rasa微服务四大组件全解
1，Rasa Actions和Tracker详解
2，Rasa Dispatcher及Event详解
3，关于Metadata的使用及Action Server启动参数详解

第69课 Rasa Knowledge Base案例解析、工作机制及自定义详解
1，ActionQueryKnowledgeBase分析及案例解析
2，Knowledge Base Actions工作机制解密
3，Knowledge Base Actions自定义详解

第70课 Rasa Core action.py源码剖析之常见类、工具方法及微服务通信类
1，三大常见类Action、ActionBotResponse、ActionListent源码逐行剖析
2，action.py中工具方法源码详解
3，微服务请求核心RemoteAction源码逐行剖析及AIOHTTP使用详解

第71课 Rasa系统内置Action源码逐行解析
1，ActionSessionStart、ActionRestart、ActionBack源码逐行解析
2，ActionEndToEndResponse、ActionDefaultFallback、ActionRevertFallbackEvents源码逐行解析
3，ActionDeactivateLoop、ActionUnlikelyIntent、ActionExecutionRejection源码逐行解析
4，ActionDefaultAskAffirmation、ActionDefaultAskRephrase、ActionExtractSlots源码逐行解析
5，extract_slot_value_from_predefined_mapping源码逐行解析

第72课 Rasa ActiveLoop、LoopAction及TwoStageFallbackAction源码逐行剖析
1，ActiveLoop源码逐行剖析
2，Rasa LoopAction源码逐行剖析
3，TwoStageFallbackAction源码逐行剖析

第73课 654行Rasa LoopAction类型的FormAction源码逐行剖析
1，LoopAction类型的FormAction运行机制和业务开发意义分析
2，Slots状态的管理、校验、和维护源码解析
3，do方法和is_done方法深度分析

第74课 代理模式下的Rasa微服务Form共1288行源码架构设计及源码逐行解析
1，Action类型的FormAction和LoopAction类型的FormAction区别与联系分析
2，Rasa微服务接口interfaces.py共370行源码逐行解析
3，Rasa SDK中的forms.py共918行源文件逐行解析

第75课 Rasa Interactive Learning运行原理、运行流程及案例实战
1，为什么说Rasa Interactive Learning是解决Rasa对话机器人Bug最容易的途径？
2，Rasa Interactive与Rasa Visualize的联合使用：Stories、Rules、NLU、Policies
3，项目案例Microservices源码逐行解析
4，使用Rasa Interactive Learning逐行调试nlu及prediction案例的三大用例场景
5，使用Rasa Interactive Learning生产数据示例实战

第76课 通过Rasa Interactive Learning发现及解决对话机器人的Bugs案例实战
1，动态的Rasa Visualization http://localhost:5006/visualization.html
2，Rasa Interactive Learning定位Slot的Bug及解决方案现场实战
3，Rasa Interactive Learning定位微服务Bug及其分析

第77课 基于ElasticSearch的Knowledge Base与Rasa对话机器人的整合在对话机器人开发中巨大价值分析
1，通过Rasa Visualize分析Pizza项目的三大运行流程
2，Pizza项目的NLU、Stories及Rules内容详解
3，项目的微服务代码详解
4，通过Rasa Interactive Learning测试Pizza form的运行及validation运行机制
5，通过Rasa Interactive Learning实战围绕Pizza form的错误对话路径及改造方式
6，通过Rasa Interactive Learning生成新的Pizza form训练数据及其训练

第78课 基于ElasticSearch的Rasa项目实战之Movie及Book Knowledge Base整合
1，基于ElasticSearch的Knowledge Base与Rasa对话机器人的整合在对话机器人开发中巨大价值分析
2，基于ElasticSearch的Rasa项目核心运行流程分析：Movies及Books操作功能详情
3，打通Rasa、微服务及ElasticSearch功能演示及运行机制分析
4，通过Rasa Shell演示项目案例的核心功能
5，通过Rasa Interactive Learning演示项目案例的内幕运行机制及流程深度剖析

第79课 Rasa与ElasticSearch整合项目案例数据及配置作机制、最佳实践、及源码剖析
1，domain.yml中的config及session_config工作机制、最佳实践、内幕自定义源码剖析
2，项目的entities及slots、Responses和actions的关系解析
4，config.yml中Pipeline及Policies详解及其背后的Rasa Graph Architecture剖析
5，NLU及Policies训练数据详解
6，通过Rasa Interactive动手实战演示join movie and rating的功能

第80课 基于ElasticSearch的Rasa项目实战之微服务源码逐行解析
1，Rasa微服务和ElasticSearch整合中代码架构分析
2，KnowledgeBase源码解析
3，MovieDocumentType、BookDocumentType、RatingDocumentType源码解析
4，ElasticsearchKnowledgeBase源码解析
5，ActionElasticsearchKnowledgeBase源码解析

第81课 通过Rasa Interactive对Rasa对话机器人项目实战之ConcertBot源码、流程及对话过程内幕解密
1，通过Rasa Visualize从全局分析ConcertBot执行流程
2，ConcertBot中的Data剖析
3，定制Slot的Mapping的三种方式剖析及具体实现
4，Rasa Interactive全程解密ConcertBot内部机制
5，自定义的Slot Mapping的Action行为透视

第82课 Rasa项目实战之Helpdesk Assistant运行流程、交互过程及源码剖析
1，通过Rasa shell演示Helpdesk Assistant的项目功能
2，现场解决DucklingEntityExtractor在Docker中使用问题
3，通过Rasa Visualize透视Helpdesk Assistant核心运行流程
4，action_check_incident_status源码解析及Slot操作深度剖析

第83课：Rasa项目实战之Helpdesk Assistant中Bug调试过程全程再现及各类现象内幕解密
1，通过Rasa Shell交互式命令复现案例中的Bug问题
2，逐词阅读Bug信息定位错误来源
3，关于payload中KeyError内幕剖析
4，配置文件分析及源码解析
5，使用rasa data validate进行数据校验
6，使用Debug模式透视问题内幕
7，Helpdesk Assistant中Bug的解决及过程总结

第84课：Rasa项目实战之Helpdesk Assistant中Domain、Action逐行解密及Rasa Interactive运行过程剖析
1，对Helpdesk Assistant中的Domain内容逐行解密
2，Helpdesk Assistant中的Action微服务代码逐行解密
3，通过Rasa Interactive纠正Helpdesk Assistant中的NLU错误全程演示
4，通过Rasa Interactive纠正Helpdesk Assistant中的Prediction错误全程演示
5，通过Rasa Interactive纠正Helpdesk Assistant中的两大核心场景全程交互解密

第85课：Rasa项目实战之电商零售Customer Service智能业务对话机器人运行流程及项目Bug调试全程演示
1，电商零售Customer Service智能业务对话机器人功能分析
2，电商零售Customer Service智能业务对话机器人运行流程
3，使用Rase shell --debug模式测试电商零售Customer Service项目及问题Bug思考
4，使用Rasa Interactive来尝试解决项目Bug
5，调整rule文件效果测试及问题分析
6，调整slot配置测试及问题解决方案剖析
7，电商零售Customer Service智能业务对话机器人调试全流程及解决方案总结

第86课：Rasa项目实战之电商零售Customer Service智能业务对话机器人微服务代码逐行解密及基于Rasa Interactive的对话试验
1，Customer Service案例使用的SQLite3数据库中数据分析
2，增加了数据库的内容但在测试的时候却没有起作用原因及解决方案
3，action_order_status代码逐行解析及Rasa Interactive试验解密
4，action_cancel_order代码逐行解析及Rasa Interactive试验解密
5，action_return代码逐行解析及Rasa Interactive试验解密
6，chitchat和faq背后的ResponseSelector解密

第87课：Rasa项目实战之电商零售Customer Service智能业务对话机器人系统行为分析及项目总结
1，电商零售Customer Service的config内容逐行分析
2，Rasa 3.x Graph Architecture剖析
3，项目实战之电商零售Customer Service的Domain内容逐行分析
4，项目实战之电商零售Customer Service的rules内容逐行分析
5，项目实战之电商零售Customer Service的数据操作代码逐行分析
6，chitchat及faq在Rasa Interactive下的测试及行为分析
7，项目实战之电商零售Customer Service项目总结

第88课：Rasa项目实战之银行金融Financial Bot智能业务对话机器人架构、流程及通过Rasa Interactive实验现象解密
1，使用Rasa Visualize对Financial Bot智能业务对话机器人架构进行解析
2，逐行剖析Rasa Interactive启动内幕及Config文件剖析
3，Rasa 3.X Graph Architecture在Financial Bot智能业务对话机器人中的应用解密
4，使用Rasa Interactive实验Financial Bot进行账户余额查询及现象解密
5，使用Rasa Interactive实验Financial Bot进行transactions消费查询及现象解密
6，action_transaction_search微服务代码解析及SlotSet事件行为分析

第89课：通过Debugging模式贯通Rasa项目实战之银行金融Financial Bot智能业务对话机器人系统启动、语言理解、对话决策、状态管理、微服务调用全生命周期流程
1，使用Rasa shell --debug模式启动银行金融Financial Bot分析
2，Financial Bot的Rasa Server启动、模型加载Debugging内容逐行解密
3，从Rasa 3.X的Graph Architecture的视角分析Financial Bot启动步骤内幕
4，用户输入Message在NLU处理中的各大组件process方法解析
5，基于State而进行的并行话policies预测过程解密
6，不同阶段State的出发机制及具体内容剖析
7，使用Financial Bot进行transfer money操作出发form循环分析
8，Rasa Server中的action及Rasa微服务中的action区别和联系源码剖析
9，Slots状态分析和状态管理
10，Financial Bot全生命周期调试总结及进一下的探索思考

第90课：Rasa项目实战之银行金融Financial Bot多种状态转换及Rasa Interactive行为分析
1，使用Rasa Interactive分析Financial Bot从money transfer状态到search recipients状态
2，使用Rasa Interactive分析Financial Bot从money transfer状态到search transactions状态
3，使用Rasa Interactive分析Financial Bot从credit card payment状态到check balance状态
4，使用Rasa Interactive分析Financial Bot从credit card payment整个证明周期流程
5，对于多状态Rasa对话机器人状态切换问题、解决方案及最佳实践分析

第91课：Rasa对话机器人项目实战之银行金融Financial Bot微服务代码逐行解密及工业级对话机器人高级代码最佳实践
1，Financial Bot微服务中使用SlotSet, Restarted,FollowupAction,UserUtteranceReverted等Event解密
2，Financial Bot微服务中对SQLite数据库的使用解析
3，Financial Bot微服务中对自定义Form Validation类CustomFormValidationAction代码逐行剖析
4，Financial Bot微服务中Payment Form Action源码及Validation代码逐行剖析
5，Financial Bot微服务中Money Transfer源码及Validation代码逐行剖析
6，Financial Bot微服务中Transaction Search源码及Validation代码逐行剖析
7，Financial Bot微服务中Explain function源码及触发代码逐行剖析
8，Financial Bot微服务中ActionSessionStart及ActionRestart自定义代码逐行剖析
9，Financial Bot微服务中ActionSwitchForms中的Ask、Deny、Affirm等行为代码逐行剖析
10，Financial Bot微服务中ActionSwitchBackAsk代码逐行剖析
11，Financial Bot微服务中代码总结及工业级Rasa对话机器人代码最佳实践分析


第92课：图解Rasa对话机器人项目实战之银行金融Financial Bot架构视角下的Training及Reference全生命周期、功能实现、及产品的二次开发
1，Rasa 3.X中Graph Architecture解析及其在银行金融Financial Bot中的落地实现
2，Rasa Architecture中的Agent、Channels、NLU Pipeline、Dialogue Policies、Tracker Store等解密
3，Rasa Architecture中的Agent和Action Server的RESTful架构通信内幕解析
4，Rasa Component Training Lifecycle组件实例化、训练及持久化解密
5，Rasa中使用Rule的通用原则及三大经典最佳实践及其在Financial Bot具体的应用
6，Rasa中多任务切换系统stories文件的设计及最佳实践及其在Financial Bot具体应用
7，Financial Bot架构视角下的Training及Reference全生命周期总结及产品的二次开发实践指导

第93课：Rasa对话机器人项目实战之保险行业Insurance Bot架构设计、流程分析、状态管理及基于Rasa Interactive的智能对话实验剖析
1，通过Rasa Visualize可视化工具详解保险行业Insurance Bot功能及架构设计
2，Rasa 3.X架构中的Agent、NLU Pipelines、Dialogue Policies、Action Server、Tracker Store等详解
3，保险行业Insurance Bot案例对Rasa 3.X各组件的应用示例
4，Insurance Bot对Graph Architecture的具体落地应用
5，逐行解密Rasa Interactive启动过程内幕
6，剖析Rasa Interactive中NLU对Insurance Bot输入的Message的处理：Intents、Entities、Slots
7，剖析Rasa Interactive中Policies触发Insurance Bot Form表单的过程内幕
8，剖析Rasa Interactive中Form运行流程及背后的密码
9，解密Insurance Bot表单提交执行微服务action全生命周期流程及Slots状态管理

第94课：Rasa对话机器人项目实战之保险行业Insurance Bot微服务代码逐行解析及现场实验剖析
1，ValidateQuoteForm三大Slot校验源码详解
2，ValidateQuoteForm三大Slot实验分析
3，ActionStopQuote代码解析及实验分析
4，ActionGetQuote源码逐行解析
5，ActionGetQuote实验分析
6，Rasa Custom Action Server Required Endpoint进程调用数据传输协议及内容剖析
7，extract slot function解密及其妙用分析
8，Address操作相关微服务代码逐行剖析
9，Claim操作相关微服务代码逐行剖析
10，Card操作相关微服务代码逐行剖析
11，Payment 操作相关微服务代码逐行剖析
12，Insurance Bot微服务源码总结及状态操作最佳实践

第95课：Rasa对话机器人项目实战之保险行业Insurance Bot的NLU及Policies数据内幕解密、源码解析及最佳实践
1，为什么有了DIETClassifier及预训练模型Duckling、spaCy等来协同完成意图识别和实体提取却还需要RegexFeaturizer、RegexEntityExtractor及EntitySynonymMapper？
2，RegexFeaturizer配置、原理、示例及文档剖析
3，RegexEntityExtractor配置、原理、示例及文档剖析
4，使用RegexFeaturizer及RegexEntityExtractor的三大最佳实践及其背后的原因剖析
5，EntitySynonymMapper配置、原理、示例及文档剖析
6，EntitySynonymMapper源码实现逐行剖析
7，Rules文件最佳实践剖析及三大经典应用
8，Stories文件最简实践解析及能够使用Stories完成不同任务上下文状态切换的背后Transformer原理解密
9，贝叶斯思想下的NLU及Policies数据最佳实践解密

第96课：Rasa对话机器人项目实战之保险行业Insurance Bot调试Debugging全程实战及背后架构、源码及本质解密
1，Rasa 3.X架构中的Agent、NLU Pipelines、Dialogue Policies、Action Server、Tracker Store等交互关系解析
2，解密Rasa shell –debug启动Insurance Bot中基于Sanic的Agent启动内幕
3，解密Rasa shell –debug启动Insurance Bot中基于Tracker Store启动内幕及最佳实践
4，解密Rasa shell –debug启动Insurance Bot中基于NLU Pipelines各大组件启动内幕
5，解密Rasa shell –debug启动Insurance Bot中基于Dialogue Policies各大组件启动内幕
6，解密Insurance Bot Debugging处理用户输入信息message的语言理解NLU全生命周期内幕
7，解密Insurance Bot Debugging处理用户输入信息message的Policies全生命周期内幕
8，解密Insurance Bot Debugging状态管理全生命周期内幕
9，解密Insurance Bot Debugging中Agent与Action Server交互的全生命周期内幕
10，解密Insurance Bot Debugging中form表单处理的全生命周期及微服务调用内幕

第97课：Rasa对话机器人项目实战之保险行业Insurance Bot调试、interactive learning解密及项目总结
1，使用Debugging模式解密Insurance Bot中的Check Claim Status全生命周期
2，使用Debugging模式解密Insurance Bot中的Pay Claim 全生命周期
3，Rasa Core中action具体请求远程微服务端endpoint数据封装、Aiohttp调用等源码剖析
4，Rasa Core中action具体收到远程微服务端endpoint的响应后进行数据处理以Channel调用等源码剖析
5，使用Rasa Interactive Learning启动Insurance Bot过程详解
6，使用Rasa Interactive Learning解密Insurance Bot的order a new card的全生命周期
7，使用Rasa Interactive Learning解密Insurance Bot的file a claim的全生命周期
8，使用Rasa Interactive Learning纠正Insurance Bot的NLU行为实战
9，使用Rasa Interactive Learning纠正Insurance Bot的Policies Prediction行为实战
10，基于使用Rasa Interactive Learning生成的新增数据分析及对话机器人训练
11，Rasa对话机器人项目实战之保险行业Insurance Bot项目总结
