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
</details>

<details>
<summary>第2章: 通过30+个细分模块完整实现Transformer论文源码及项目调试</summary>
<br>
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
</details>
