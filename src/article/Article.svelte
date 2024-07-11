<script>
	import HyperparameterView from '../detail-view/Hyperparameterview.svelte';
  import Youtube from './Youtube.svelte';

	let softmaxEquation = `$$\\text{Softmax}(x_{i}) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}$$`;
	let reluEquation = `$$\\text{ReLU}(x) = \\max(0,x)$$`;

  let currentPlayer;
</script>

<style>
	#description {
    margin-bottom: 60px;
    margin-left: auto;
    margin-right: auto;
    max-width: 78ch;
  }

  #description h2 {
    color: #444;
    font-size: 40px;
    font-weight: 450;
    margin-bottom: 12px;
    margin-top: 60px;
  }

  #description h4 {
    color: #444;
    font-size: 32px;
    font-weight: 450;
    margin-bottom: 8px;
    margin-top: 44px;
  }

  #description h6 {
    color: #444;
    font-size: 24px;
    font-weight: 450;
    margin-bottom: 8px;
    margin-top: 44px;
  }

  #description p {
    margin: 16px 0;
  }

  #description p img {
    vertical-align: middle;
  }

  #description .figure-caption {
    font-size: 13px;
    margin-top: 5px;
  }

  #description ol {
    margin-left: 40px;
  }

  #description p, 
  #description div,
  #description li {
    color: #555;
    font-size: 17px;
    line-height: 1.6;
  }

  #description small {
    font-size: 12px;
  }

  #description ol li img {
    vertical-align: middle;
  }

  #description .video-link {
    color: #3273DC;
    cursor: pointer;
    font-weight: normal;
    text-decoration: none;
  }

  #description ul {
      list-style-type: disc;
      margin-top: -10px;
      margin-left: 40px;
      margin-bottom: 15px;
  }
    
  #description a:hover, 
  #description .video-link:hover {
    text-decoration: underline;
  }

  .figure, .video {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
</style>

<body>
  <div id="description">
    <h2>什么是卷积神经网络？</h2>
    <p>
      在机器学习中，分类器会将一个数据点分配给一个类标签。例如，一个<em>图像分类器</em>会生成一个类标签（如鸟、飞机）来表示图像中存在的对象。<em>卷积神经网络</em>，简称 CNN，是一种特别擅长解决这个问题的分类器。
	 </p>
  	<p>
      CNN 是一种神经网络：一种用于识别数据中模式的算法。神经网络通常由一组组织在层中的神经元组成，每个神经元都有自己的可学习权重和偏置。我们来看看 CNN 的基本构成部分。
  	</p>
  	<ol>
      <li><strong>张量</strong>可以理解为一个 n 维矩阵。在 CNN 中，除了输出层以外，张量通常是 3 维的。</li>
      <li><strong>神经元</strong>可以看作一个函数，它接收多个输入并产生一个输出。神经元的输出在上面表示为<span style="color:#FF7577;">红色</span> &rarr; <span style="color:#60A7D7;">蓝色</span> <strong>激活图</strong>。</li>
      <li><strong>层</strong>是一组执行相同操作的神经元，包括相同的超参数。</li>
      <li><strong>核权重和偏置</strong>，虽然每个神经元的权重和偏置都是独特的，但它们在训练过程中被调整，使分类器能够适应具体的问题和数据集。在可视化中，它们以<span style="color:#BC8435;">黄色</span> &rarr; <span style="color:#39988F;">绿色</span>的色标表示。具体的值可以通过点击神经元或在<em>卷积弹性解释视图</em>中悬停在核/偏置上查看。</li>
      <li>CNN 传达了一个<strong>可微分的评分函数</strong>，在输出层的可视化中表示为<strong>类别得分</strong>。</li>
  	</ol> 
  	<p>
      如果你之前研究过神经网络，这些术语可能对你来说很熟悉。那么，CNN 有什么不同呢？CNN 使用一种叫做卷积层的特殊层，使它们非常擅长从图像及类似图像的数据中学习。对于图像数据，CNN 可以用于多种计算机视觉任务，例如<a href="http://ijcsit.com/docs/Volume%207/vol7issue5/ijcsit20160705014.pdf" title="CNN Applications">图像处理、分类、分割和目标检测</a>。
  	</p>  
  	<p>
      在 CNN Explainer 中，你可以看到一个简单的 CNN 是如何进行图像分类的。尽管由于网络的简化，其性能并不是最优的，但这并不要紧！CNN Explainer 中使用的网络架构<a href="http://cs231n.stanford.edu/" title="Tiny VGG Net presented by Stanford's CS231n">Tiny VGG</a>包含了当今最先进的 CNN 所使用的许多层和操作，只是规模更小。因此，更容易让初学者理解和掌握。 
    </p>     

      <h2>每层网络的作用是什么？</h2>
      <p>
        让我们来逐步了解网络中的每一层。在阅读的同时，可以通过点击和悬停上方的可视化部分进行互动。
      </p>
      <h4 id='article-input'>输入层</h4>
      <p>
        输入层（最左侧的层）代表输入到 CNN 的图像。由于我们使用 RGB 图像作为输入，输入层有三个通道，分别对应红色、绿色和蓝色通道。在该层中展示。当你点击上方的 <img class="is-rounded" width="12%" height="12%" src="PUBLIC_URL/assets/figures/network_details.png" alt="网络详情图标"/> 图标时，可以使用色彩比例来显示详细信息（包括该层及其他层）。
      </p>
      <h4 id='article-convolution'>卷积层</h4>
      <p>
        卷积层是 CNN 的基础，因为它们包含了已学习的核（权重），这些核提取出区分不同图像的特征——这正是我们分类所需要的！当你与卷积层互动时，你会注意到前一层和卷积层之间的链接。每个链接代表一个独特的核，用于卷积操作以产生当前卷积神经元的输出或激活图。
      </p>
  	<p>
      卷积神经元使用一个独特的核与前一层对应神经元的输出进行逐元素点积。这将产生与独特核数量相等的中间结果。卷积神经元是所有中间结果与学习到的偏置之和。
  	</p>
  	<p>
      例如，让我们看看上方 Tiny VGG 架构中的第一个卷积层。注意这一层有 10 个神经元，而前一层只有 3 个神经元。在 Tiny VGG 架构中，卷积层是全连接的，意味着每个神经元与前一层的每个其他神经元相连。聚焦在第一个卷积层中最顶部的卷积神经元的输出上，当我们悬停在激活图上时，可以看到有 3 个独特的核。
  	</p>
    <div class="figure">
      <img src="PUBLIC_URL/assets/figures/convlayer_overview_demo.gif" alt="clicking on topmost first conv. layer activation map" width=60% height=60% align="middle"/>
      <div class="figure-caption">
  		  图 1.  当你把鼠标移动到第一个卷积层顶部节点的激活图上时，你会发现有三个内核被用来生成这个激活图。点击这个激活图后，你可以看到每个内核的卷积操作过程。

  	  </div>
    </div>

  	<p>
      这些核的大小是由网络架构设计师设定的一种超参数。为了得到卷积神经元的输出（也就是激活图），我们需要将上一层的输出与网络所学习的独特核进行元素级点积操作。在 TinyVGG 网络中，点积操作的步幅为 1，意味着每进行一次点积操作，核就移动 1 个像素。不过，这个步幅是一个可以根据数据集需要进行调整的超参数。我们必须对所有 3 个核进行这种操作，最终得到 3 个中间结果。
  	</p>
    <div class="figure">
      <img src="PUBLIC_URL/assets/figures/convlayer_detailedview_demo.gif" alt="clicking on topmost first conv. layer activation map" />
      <div class="figure-caption">
        图 2. 核函数被应用于生成讨论中的激活图的最高中间结果。
      </div>
    </div>
  	<p>
      然后，网络会对所有 3 个中间结果及其学习到的偏差进行逐元素相加。这样，结果是一个二维张量，可以在界面上查看，这是第一卷积层中最顶层神经元的激活图。每个神经元的激活图都是通过相同的方法得到的。
  	</p>
  	<p>
      通过简单的数学运算，我们可以知道，在第一卷积层中有 3 x 10 = 30 个独特的核，每个核的大小为 3x3。卷积层与前一层的连接方式是设计网络架构时的一个重要决定，它会影响每个卷积层的核数量。通过点击可视化界面，你可以更好地理解卷积层的运作。试试能否跟随上面的例子进行理解！
    </p>
    <h6>理解超参数</h6>
    <p>
    	<HyperparameterView/>
    </p>
    <ol>
      <li><strong>Padding</strong>（填充）在卷积核超出激活图时常常是必要的。填充可以保留激活图边界的数据，从而提高性能，并且可以帮助<a href="https://arxiv.org/pdf/1603.07285.pdf" title="See page 13">保持输入的空间尺寸</a>，这使得架构设计师能够构建更深、更高性能的网络。存在<a href="https://arxiv.org/pdf/1811.11718.pdf" title="Outlines major padding techniques">许多填充技术</a>，但最常用的方法是零填充，因为其性能、简单性和计算效率。这种技术涉及在输入的边缘对称地添加零。这种方法被许多高性能的卷积神经网络（CNN）如<a href="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf" title="AlexNet">AlexNet</a>采用。</li>
      <li><strong>Kernel size</strong>（核大小），通常也称为滤波器大小，指的是在输入上滑动窗口的尺寸。选择这个超参数对图像分类任务有巨大的影响。例如，小核大小能够从输入中提取包含高度局部特征的更大量的信息。如上图所示，较小的核大小也会导致层维度的较小减少，从而允许构建更深的架构。相反，大核大小提取较少的信息，这会导致层维度的更快减少，通常会导致性能下降。大核更适合提取较大的特征。最终，选择适当的核大小将取决于你的任务和数据集，但一般来说，较小的核大小对图像分类任务的性能更好，因为架构设计师能够堆叠<a href="https://arxiv.org/pdf/1409.1556.pdf" title="Learn why deeper networks perform better!">更多层以学习更复杂的特征</a>！</li>
      <li><strong>Stride</strong>（步幅）表示每次应将卷积核移动多少像素。例如，在上面的卷积层示例中，Tiny VGG对其卷积层使用步幅为1，这意味着在输入的3x3窗口上执行点积以得出输出值，然后在每次后续操作时向右移动一个像素。步幅对CNN的影响类似于核大小。随着步幅的减小，学习到的特征更多，因为提取的数据更多，这也导致输出层更大。相反，随着步幅的增加，这会导致特征提取更有限，输出层维度更小。架构设计师的一项职责是在实现CNN时确保卷积核对称地滑过输入。使用上面的超参数可视化工具，在各种输入/核尺寸上改变步幅以理解这个限制！</li>
    </ol>
    <h4>激活函数</h4>
    <h6 id='article-relu'>ReLU</h6>
    <p>
      神经网络在现代技术中非常普及——这是因为它们极其准确！如今，性能最好的卷积神经网络（CNN）由大量层组成，这些层能够学习更多的特征。这些突破性的 CNN 之所以能达到如此<a href="https://arxiv.org/pdf/1512.03385.pdf" title="ResNet">惊人的准确性</a>，部分原因在于它们的非线性特性。ReLU 函数将必需的非线性引入模型中，这种非线性对于形成非线性决策边界是必要的，从而使输出无法通过输入的线性组合来表示。如果没有非线性激活函数，深层的 CNN 架构将简化为单一的等效卷积层，其性能将大大下降。ReLU 激活函数被特别选作非线性激活函数，而不是其他非线性函数如<em>Sigmoid</em>，因为已有<a href="https://arxiv.org/pdf/1906.01975.pdf" title="See page 29">实验证据</a>表明，使用 ReLU 的 CNN 比使用其他激活函数的 CNN 训练速度更快。
    </p>
    <p>
      ReLU激活函数是一种逐元素的数学运算: {reluEquation}
    </p>
    <div class="figure">
    <img src="PUBLIC_URL/assets/figures/relu_graph.png" alt="relu graph" width="30%" height="30%"/>
      <div class="figure-caption">
        图 3. ReLU 激活函数绘制的图形，忽略了所有负值数据。
      </div>
    </div>
    <p>
      这个激活函数会逐个应用于输入张量的每个值。例如，当 ReLU 应用于值 2.24 时，结果为 2.24，因为 2.24 大于 0。你可以通过点击上方网络中的 ReLU 神经元，来观察这个激活函数是如何工作的。在上述网络架构中，每个卷积层之后都会执行修正线性激活函数 (ReLU)。注意，这一层对网络中不同神经元的激活图有显著影响！

    </p>
    <h6 id='article-softmax'>Softmax</h6>
    <p>
    	{softmaxEquation}
    	softmax 操作有一个重要的作用：确保 CNN（卷积神经网络）的输出总和为 1。因此，softmax 操作非常适合将模型的输出转换为概率。点击网络的最后一层，你可以看到 softmax 操作。注意，展平后的 logits（逻辑回归输出值）没有被缩放到 0 到 1 之间。为了直观地显示每个 logit（未缩放的标量值）的影响，它们被编码为 <span style="color:#FFC385;">浅橙色</span> &rarr; <span style="color:#C44103;">深橙色</span> 的颜色刻度。通过 softmax 函数处理后，每个类别现在都有一个相应的概率！

    </p>
    <p>
    	你可能会好奇标准归一化和 softmax 之间有什么区别&mdash;毕竟，它们都将 logits 缩放到 0 到 1 之间。要记住，反向传播是训练神经网络的关键&mdash;我们希望正确答案有最大的“信号”。通过使用 softmax，我们实际上在获得可微分性的同时“近似”了 argmax。重新缩放不会显著提高最大 logits 的权重，而 softmax 会。简单来说，softmax 就是一个“更柔和”的 argmax&mdash;明白我们的意思了吗？

    </p>
    <div class="figure">
    <img src="PUBLIC_URL/assets/figures/softmax_animation.gif" alt="softmax interactive formula view"/>
      <div class="figure-caption">
        图 4. <em>Softmax Interactive Formula View</em> 让用户可以与颜色编码的 logit 和公式进行互动，帮助理解在展平层之后，预测分数是如何归一化为分类分数的。

      </div>
    </div>
    <h4 id='article-pooling'>Pooling Layers</h4>
    <p>
      在不同的 CNN（卷积神经网络）架构中，有很多种池化层，但它们的作用都是逐步减小网络的空间范围，从而减少网络的参数和计算量。在上文提到的 Tiny VGG 架构中，使用的是最大池化（Max-Pooling）。
    </p>
    <p>
      最大池化操作需要在设计架构时选择一个内核大小和步长长度。一旦确定，操作就会以指定的步长在输入数据上滑动内核，同时在每个内核切片中只选择最大的值作为输出值。可以通过点击上方网络中的池化神经元查看这一过程。
    </p>
    <p>
      在上文提到的 Tiny VGG 架构中，池化层使用的是 2x2 的内核和步长为 2 的操作。这种规格的操作会丢弃 75% 的激活值。通过丢弃这么多值，Tiny VGG 的计算效率更高，同时避免了过拟合。
    </p>
    
    <h4 id='article-flatten'>Flatten Layer</h4>
    <p>      
      这个层将网络中的三维数据转换为一维向量，以适应全连接层的分类输入。例如，一个 5x5x2 的张量会被转换成大小为 50 的向量。网络之前的卷积层从输入图像中提取了特征，而现在需要对这些特征进行分类。我们使用 softmax 函数来对这些特征进行分类，它需要一个一维输入。这就是为什么需要 flatten 层。可以通过点击任何输出类别来查看这个层。
    </p>

    <h2>交互功能</h2>
    <ol>
      <li><strong>上传你的图片</strong>，选择 <img class="icon is-rounded" src="PUBLIC_URL/assets/figures/upload_image_icon.png" alt="upload image icon"/> 来查看你的图片如何被分类到 10 个类别中。通过分析整个网络的神经元，你可以了解激活图和提取的特征。</li>
      <li><strong>调整激活图的颜色比例</strong>，通过调整 <img class="is-rounded" width="12%" height="12%" src="PUBLIC_URL/assets/figures/heatmap_scale.png" alt="heatmap"/> 来更好地理解不同抽象层次上的激活效果。</li>
      <li><strong>查看网络细节</strong>，比如层的维度和颜色比例，通过点击 <img class="is-rounded" width="12%" height="12%" src="PUBLIC_URL/assets/figures/network_details.png" alt="network details icon"/> 图标。</li>
      <li><strong>模拟网络运行</strong>，通过点击 <img class="icon is-rounded" src="PUBLIC_URL/assets/figures/play_button.png" alt="play icon"/> 按钮，或在 <em>交互式公式视图</em> 中悬停在输入或输出部分来与层片段互动，以了解映射和底层操作。</li>
      <li><strong>学习各层功能</strong>，通过点击 <img class="icon is-rounded" src="PUBLIC_URL/assets/figures/info_button.png" alt="info icon"/>，在 <em>交互式公式视图</em> 中阅读文章中的层细节。</li>
      
    </ol> 

    <h2>Video Tutorial</h2>
    <ul>
      <li class="video-link" on:click={currentPlayer.play(0)}>
        CNN Explainer Introduction
        <small>(0:00-0:22)</small>
      </li>
      <li class="video-link" on:click={currentPlayer.play(27)}>
        <em>Overview</em>
        <small>(0:27-0:37)</small>
      </li>
      <li class="video-link" on:click={currentPlayer.play(37)}>
        Convolutional <em>Elastic Explanation View</em>
        <small>(0:37-0:46)</small>
      </li>
      <li class="video-link" on:click={currentPlayer.play(46)}>
        Convolutional, ReLU, and Pooling <em>Interactive Formula Views</em>
        <small>(0:46-1:21)</small>
      </li>
      <li class="video-link" on:click={currentPlayer.play(82)}>
        Flatten <em>Elastic Explanation View</em>
        <small>(1:22-1:41)</small>
      </li>
      <li class="video-link" on:click={currentPlayer.play(101)}>
        Softmax <em>Interactive Formula View</em>
        <small>(1:41-2:02)</small>
      </li>
      <li class="video-link" on:click={currentPlayer.play(126)}>
        Engaging Learning Experience: Understanding Classification
        <small>(2:06-2:28)</small>
      </li>
      <li class="video-link" on:click={currentPlayer.play(149)}>
        Interactive Tutorial Article
        <small>(2:29-2:54)</small>
      </li>
    </ul>
    <div class="video">
      <Youtube videoId="HnWIHWFbuUQ" playerId="demo_video" bind:this={currentPlayer}/>
    </div>

    <h2> CNN Explainer 是如何实现的？</h2>
    <p>
      CNN Explainer 通过使用 <a href="https://js.tensorflow.org/"><em>TensorFlow.js</em></a>，这是一种可以在浏览器中运行的 GPU 加速深度学习库，来加载预训练模型进行可视化。整个互动系统是用 Javascript 编写的，采用 <a href="https://svelte.dev/"><em>Svelte</em></a> 作为框架，并用 <a href="https://d3js.org/"><em>D3.js</em></a> 进行数据可视化。你只需要一个网络浏览器，就可以开始学习 CNNs 了！
    </p>
    

    <h2>谁开发了 CNN Explainer？</h2>
    <p>
      CNN Explainer 由
      <a href="https://zijie.wang/">Jay Wang</a>,
      <a href="https://www.linkedin.com/in/robert-turko/">Robert Turko</a>, 
      <a href="http://oshaikh.com/">Omar Shaikh</a>,
      <a href="https://haekyu.com/">Haekyu Park</a>,
      <a href="http://nilakshdas.com/">Nilaksh Das</a>,
      <a href="https://fredhohman.com/">Fred Hohman</a>,
      <a href="http://minsuk.com">Minsuk Kahng</a> 和
      <a href="https://www.cc.gatech.edu/~dchau/">Polo Chau</a> 创建，这是 Georgia Tech 和 Oregon State 研究合作的成果。我们感谢 Anmol Chhabria、Kaan Sancak、Kantwon Rogers 和 Georgia Tech Visualization Lab 的支持和建设性反馈。本项目部分由 NSF（IIS-1563816，CNS-1704701）、NASA NSTRF、DARPA GARD、Intel、NVIDIA、Google 和 Amazon 的捐赠支持。
    </p>
  </div>
</body>
