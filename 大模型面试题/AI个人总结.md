### 1、大模型微调

#### 1.1、大模型项目pipeline

```
1、确定要解决的问题
2、选择模型
3、模型调整（提示词工程->微调->对其人类反馈->评估）
4、应用
```

#### 1.2、微调：在已经预训练好的模型基础上，利用特定领域的数据集进行二次训练的过程

```txt
1、预训练 pre-train:无标注，海量语料
2、监督式微调SFT：有标注，少量语料
构造数据集：
	训练模型认识标注资料中的标识符

3、强化学习RL【算法：PPO，RLHF，DPO】：训练语料（1、chosen效果好，2、rejected效果差）
DPO算法让大模型回答问题的时候，倾向于chosen这种回答（偏好对齐）
RLHF算法

搭建环境：云端资源->连接环境->下载预训练模型（hugging face）
```

#### 1.3、数据收集：

```
思路1：人类生成，标注（高质量高成本）
思路2：已存在的数据，转换结构化数据（instructed data）
思路3：AI生成，人类过滤
思路4：人类生成少量，指导AI生成（评价数据的手段）
```

#### 1.4、LORA、QLoRA、AdaLoRA

```
LORA篇
核心思想（前提：模型不改变）：将LLM的参数矩阵分解为低秩近似，来减少模型的复杂度和计算资源的需求。（矩阵的秩，rank，降rank，Rank R=8最优）
方法：矩阵分解技术，如奇异值分解（SVD）、特征值分解
效果：加速大模型推理过程，减小模型存储需求，提高模型效率，可以最大限度的保持模型性能

QLoRA篇
在LORA的基础上，额外过将参数进⾏量化，即将浮点参数转换为固定位数的整数或简单表示，从而显著减少模型的存储需求
效果：，QLoRA通过量化和低秩适应的⽅法，可以在减少存储需求和计算复杂度的同时，保持模型的关键
特征和性能。它具有⾼效、通⽤和可扩展的特点，适⽤于各种⼤型语⾔模型的优化。

AdaLoRA篇
个人认为就是迭代LORA，一次LORA之后进行评估，若符合要求，则作为最终模型，若不符合要求，则在上一次的基础上继续LORA

合并LoRA权重到原模型的过程：通常涉及将低秩矩阵重新组合成原始模型的参数矩阵。这可以通过矩阵
乘法等操作来实现
注意：合并LoRA权重到原模型时，可能会有⼀些微⼩的性能损失。这是因为低秩适应过程中
对参数进⾏了量化和近似处理，可能会损失⼀些细节信息。然⽽，通过合适的低秩适应⽅法和参数设
置，可以最⼩化这种性能损失，同时获得较⾼的效率和较低的资源开销。

LoRA微调具有以下⼏个优点：
1、保留原模型知识
2、减少资源开销
3、提高模型的泛化能力，能够适应不同领域的任务和要求（对原模型进⾏了正则化）
4、可扩展性和灵活性：LoRA微调⽅法的设计可以根据具体任务和资源限制进⾏调整和优化。可以通过
调整低秩适应的程度、迭代次数和参数设置等来平衡性能和效率。这种灵活性使得LoRA微调适⽤于
不同规模和需求的语⾔模型，具有较⾼的可扩展性

LoRA微调⽅法能加速训练的原因：减少了参数量，降低了计算复杂度，加速收敛速度，提高了计算效率
```

微调参数与W的关系：
放大了某方面的能力![image-20250309114847257](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250309114847257.png)

#### 1.5、模型蒸馏

```
1、原理：将待压缩的模型作为教师模型，将体积更小的模型作为学生模型，让学生模型在教师模型的监督下进行优化，将学生模型学习到教师模型的概率分布，通过kl散度进行控制

2、方法：
	2.1、黑盒知识蒸馏：使用大模型生成知识数据，通过生成的数据进行模型微调。
		优缺点：实现简单，蒸馏效率低
	2.2、白盒知识蒸馏：获取学生模型和教师模型的输出概率分布，通过kl散度将学生模型的概率向教		   师模型对齐。
		（前向kl散度，反向kl散度，偏向前kl散度，偏向反kl散度）
	2.3、代码实现白盒知识蒸馏
		

```

#### 1.6、SFT

```
指令微调数据构建
1、收集原始数据
2、标注数据
3、划分数据集（训练集、验证集，测试集）
4、数据预处理
5、格式转换
6、模型微调
7、模型评估

SFT（有监督微调）阶段主要通过任务特定的标签预测、上下⽂理解和语⾔模式、特征提取和表示学
习以及任务相关的优化来进⾏学习。通过这些学习，模型可以适应特定的任务，并在该任务上表现出良
好的性能。
```



### 2、AI-在线助手项目

#### 离线部分：

数据采集->结构化数据->RNN判断是否是合理的领域的名词->存入数据库
数据采集->非结构化数据->NER模型提取名词->RNN判断是否是合理的名词->存入数据库

#### 命名实体审核任务：

对于处理过的非结构化数据进行合法性校验，检验过程无需上下文，只关注于字符本身的组合方式
模型选取：短文本选择RNN。bert-chinese预训练模型来获取中文的向量表示
数据集：1表示正标签（通过人工审核）0表示负样本（正样本逆序排列）
RNN模型：两层结构input2hidden，inpout2output

#### 命名实体识别任务：

BIO，识别专有名词，
模型选取：BiLSTM+CRF，BiLSTM输出相应标签的打分矩阵，CRF将这些分值作为输入，CRF层能从训练数据中获取约束性的规则，降低非法数据

#### 句子主题相关：BERT

构建全连接微调模型：两个全连接层

### 2、RAG大模型知识库项目：

模型：ChatGLM-6B 模型文件大小10G左右
硬件：64核cpu，1T内存，40G显存

#### 文档处理角度：

读取：多模态文档处理流水线，支持多种文件格式解析
->切分：1、达摩院模型nlp_bert进行语义分割，2、文本分割成句子列表，3、根据一系列分隔符递归地分割文本，并返回分割后的文本块列表，4、判断文本是否可能是标题
->转换为向量存入向量数据库（embedding模型和对应的向量数据库）文本列表转换为归一化的嵌入向量列表
->问句向量化 
->在文本向量中匹配出与问句向量最相似的 top k个
->知识库内容+句子上下文
->传给llm生成prompt
->LLM生成回答

#### 文档处理模块：

1、加载器（pdf，csv，图片（ocr识别））
2、文档分割（1、使用模型分割（达摩院开源nlp_bert_document-segmentation_chinese-base）2、换行符、标点符号分割3、文章标题判断）
3、文本向量化

#### 数据库模块：

1、存放的模型：聊天记录模型、知识库模型、知识文件模型、文件-向量库文档模型、chunk-summary模型，用于存储file_doc中每个doc_id的chunk片段、聊天记录模型
2、封装数据库CURD方法
3、向量数据库：FAISS

#### 知识模块：

1、文档分段总结任务：使用llm完成文本总结（1、对每个文档进行处理，得到每个文档的摘要 2、对每个文档的摘要进行合并，得到最终的摘要）
2、向量数据库保存到本地磁盘，从本地磁盘加载到内存中
3、本地知识库服务封装

#### 用户对话模块：

1、保存对话信息到数据库，可以设置从数据库读取历史对话消息
2、用户上传的文件处理：文档处理模块处理之后，存入向量数据库
3、在文本向量中匹配出与问句向量最相似的 top k个 -> 匹配出的文本作为上下文和问题一起添加到prompt中 -> 提交给LLM生成回答
4、记录聊天记录信息：评分，理由等

```
RAG高级
1、优化原始提问

2、路由和高级查询
	路由：查找更适合提示词模板/更合适的文档
	查询构建：将问题转换为Metadata数据
3、丰富索引结构
4、重排序
```

#### RAG问题

先将prompt和context经过一个小的LLM模型，要求它生成搜索词，然后发给rag数据库进行搜索，再将结果和context给一个最终模型。

### 3、LangGraph ChatBot 开发

#### 1、构建基础聊天机器人

##### 1.1、定义状态图 StateGraph

​       定义状态类型-》创建状态图对象
​       graph_builder = StateGraph(State)

##### 1.2、添加聊天节点（节点表示一个计算单元，通常是常规的python函数）

​       节点函数接收当前State，返回一个更新后的列表字典
​       graph_builder.add_node()

##### 1.3、定义对话流程（状态图的起终点）

​        graph_builder.add_edge(START, "chatbot")
​        graph_builder.add_edge("chatbot", END)

##### 1.4、编译图

​        graph = graph_builder.compile()

##### 1.5、运行图

​        graph.stream({"messages": ("user", user_input)})

#### 2、为聊天机器人添加工具

##### 2.1、Tavily

##### 2.2、定义工具        

```python
定义 Tavily 搜索工具，最大搜索结果数设置为 2
tool = TavilySearchResults(max_results=2)
tools = [tool]
```

##### 2.3、将工具集成到状态图中

```python
# 初始化 LLM 并绑定搜索工具
chat_model = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = chat_model.bind_tools(tools)

```

##### 2.4、工具调用

​        创建一个函数来运行工具，通过向图中添加一个新节点来实现这一点        

```python
# 将 BasicToolNode 添加到状态图中
tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
```

##### 2.5、添加条件边

​        条件边将控制流从一个节点路由到另一个节点。条件边通常包含 if 语句，以根据当前状态将控制流路由到不同的节点
​        1、route_tools 函数：这是一个路由函数，用来决定机器人在对话流程中的下一步
​        2、add_conditional_edges：这是 LangGraph 中用于条件路由的函数。它允许我们根据 route_tools 函数的返回值决定下一个要执行的节点
​        3、add_edge：当工具节点执行完成后，机器人会返回到 chatbot 节点继续对话

```python
# 添加条件边，判断是否需要调用工具
graph_builder.add_conditional_edges(
    "chatbot",  # 从聊天机器人节点开始
    route_tools,  # 路由函数，决定下一个节点
    {
        "tools": "tools", 
        "__end__": "__end__"
    },  # 定义条件的输出，工具调用走 "tools"，否则走 "__end__"
)
# 当工具调用完成后，返回到聊天机器人节点以继续对话

graph_builder.add_edge("tools", "chatbot")
```

##### 2.6、编译图

#### 3、为聊天机器人添加记忆功能

​     MemorySaver 是 LangGraph 中的一个检查点机制（checkpointing），用于保存和恢复对话或任务执行的状态。通过 MemorySaver，我们可以在每个步骤后保存机器人的状态，并在之后恢复对话，允许机器人具有“记忆”功能，支持多轮对话、错误恢复、时间旅行等功能。

##### 3.1、创建MemorySaver 检查点

​        memory = MemorySaver()

##### 3.2、使用检查点编译图

​        graph = graph_builder.compile(checkpointer=memory)

##### 3.3、执行机器人        

```python
    config = {"configurable": {"thread_id": "1"}}
        events = graph.stream(
    {"messages": [("user", user_input)]},  # 第一个参数传入用户的输入消息，消息格式为 ("user", "输入内容")
    config,  # 第二个参数用于指定线程配置，包含线程 ID
    stream_mode="values"  # stream_mode 设置为 "values"，表示返回流式数据的值）
    # 通过 get_state 函数查看机器人当前的状态
```

#### 4、引入人类审查

​     有时机器人可能需要人类的介入来做出复杂决策，或者某些任务需要人类批准。LangGraph 支持这种“人类参与”的工作流

##### 4.1、编译图时添加中断    

​        通过 interrupt_before 参数，在工具节点执行之前中断对话，让人类有机会审查		
​        

##### 4.2、执行对话并在工具调用前中断

```python
# 编译状态图，指定在工具节点之前进行中断
graph = graph_builder.compile(
    checkpointer=memory,  # 使用 MemorySaver 作为检查点系统
    interrupt_before=["tools"],  # 在进入 "tools" 节点前进行中断
)
```

##### 4.3、人工介入，修改工具执行结果        

```python
# 手动生成一个工具调用的消息，并更新到对话状态中
tool_message = ToolMessage(
    content="LangGraph 是一个用于构建状态化、多参与者应用的库。",  # 工具调用返回的内容
    tool_call_id=snapshot.values["messages"][-1].tool_calls[0]["id"]  # 关联工具调用的 ID
)
# 更新对话状态，加入工具调用的结果

graph.update_state(config, {"messages": [tool_message]})

# 继续执行对话，查看工具调用后的后续处理

events = graph.stream(None, config, stream_mode="values")
```



#### 5、查看 ChatBot 历史对话

​     使用 graph.get_state_history() 来获取对话的所有历史状态

### 4、LangGraph 多智能体协作

#### 工作流程概述

1、定义工具：为每个智能体提供专用工具，用于执行特定的任务
2、定义辅助函数：agent_node：将每个智能体与对应任务进行关联，定义图中的智能体节点，使其能够处理特定任务。
3、定义辅助函数：create_agent：为每个任务创建独立的智能体，例如研究智能体、图表生成器智能体等。每个智能体使用独立的语言模型和工具。
4、定义研究智能体及节点: Researcher: 研究智能体使用 Tavily 搜索工具，回应用户提问。
5、定义图表生成器智能体及节点: Chart_Generator: 根据提供的数据，在沙盒环境执行 Python 代码生成图表。
6、导入预构建的工具节点: ToolNode: 将2中定义的 Tavily 搜索工具和 Python REPL 工具作为一个工具节点，这样可以方便地在工作流中使用这些工具。
7、建立智能体节点间通信: AgentState：通过 LangGraph 实现智能体间通信，智能体能够共享状态并相互协作完成复杂任务。
8、定义工作流（状态图)：创建状态图以管理多智能体协作的流程，包含任务路由和边逻辑，确保正确的智能体按顺序执行。
9、执行工作流：根据状态图执行多智能体任务，通过工具调用和智能体协作，完成目标任务并生成最终输出。

#### 定义路由函数：是工作流中的一个关键逻辑，用于根据当前的状态和消息内容来决定下一步的操作。

### 5、LangGraph Reflection（反思）机制

Reflection 是一种重要的模型能力，通过让模型观察其过去的步骤和外部环境反馈，评估自身行为的质量，并不断改进输出。在生成与反思的循环中，模型可以逐步优化内容，从而提升生成质量和用户满意度。
Reflection 机制被广泛应用于生成任务中，例如文章写作、内容修改与反馈、以及智能助理等场景。通过引导 LLM 进行自我反思和用户反馈处理，开发者可以让模型在多轮交互中自动调整其生成的内容，达到高效、精准、结构完善的输出。
应用：写作助手智能体
1、定义写作助手智能体
2、定义审阅老师智能体

### LangGraph Reflexion（自我反思）开发

1、定义LLM
2、Actor（具有反思功能）
3、构建工具
4、初始响应器
5、修订
6、创建工具节点
举例：自我反思机器人
1、初始化：
  参数： llm: ChatOpenAI,
	 tools: list[BaseTool],
	 history_len: int,
	 checkpoint: BaseCheckpointSaver,
	 knowledge_base: str = None,
	 top_k: int = None,
	 score_threshold: float = None
#担心用户没有选择任何 tool 而造成 agent 逻辑无效, 为保证效果, 强行追加一个 search_internet 工具, 如开发者不需要可注释此行代码.
        search_internet = get_tool(name="search_internet")
        self.tools = add_tools_if_not_exists(tools_provides=self.tools, 		tools_need_append=[search_internet])
2、工具节点[tool node]
ToolNode 默认只将结果追加到 messages 队列中, 所以需要手动在 history 中追加 ToolMessage 结果

3、图
子图：
function_call_sub_graph_builder = StateGraph(ReflexionState)
function_call_sub_graph_builder.add_node("function_call", self.function_call)
        function_call_sub_graph_builder.add_node("tools", tool_node)
        function_call_sub_graph_builder.add_node("process_func_call_history", self.process_func_call_history)
        function_call_sub_graph_builder.set_entry_point("function_call")
        function_call_sub_graph_builder.add_edge("function_call", "tools")
        function_call_sub_graph_builder.add_edge("tools", "process_func_call_history")
        function_call_sub_graph_builder.add_edge("process_func_call_history", END)
        function_call_sub_graph = function_call_sub_graph_builder.compile(checkpointer=self.checkpoint)
构造图：
        builder = StateGraph(ReflexionState)

        builder.add_node("history_manager", self.async_history_manager)
        builder.add_node("draft", self.initial) # 草稿
        builder.add_node("revise", self.revision) # 修改
        builder.add_node("function_call", function_call_sub_graph)
        builder.add_node("function_call_loop", function_call_sub_graph)

        builder.set_entry_point("history_manager")
        builder.add_edge("history_manager", "function_call")
        builder.add_edge("function_call", "draft")
        # draft -> execute_tools
        builder.add_edge("draft", "function_call_loop")
        # execute_tools -> revise
        builder.add_edge("function_call_loop", "revise")
        # revise -> execute_tools OR end
        builder.add_conditional_edges("revise", self.event_loop)

        graph = builder.compile(checkpointer=self.checkpoint)

蒙特卡洛树搜索



您好！本人Python 基础扎实，擅长使用FastAPI构建高性能服务，并运用Docker/K8s实现容器化部署。熟悉Transformer/GPT架构，能基于PyTorch实现RAG增强和LoRA微调，成功开发过具备工具调用能力的LangChain智能体。掌握从数据清洗、模型训练（Scikit-learn/pytorch）到Dify多智能体落地的全流程，擅长设计高扩展性架构。在GitHub贡献过结合Streamlit与Llama-index的记忆增强型应用，注重代码工程化和业务价值转化。期待用AI工程化能力为贵团队创造实际效益



本人Python 基础扎实，有良好的编码习惯，能够使用numpy，matplotlib，pandas进行数据分析、数据挖掘、数据预测等，熟悉 SciPy、Keras、scikit-learn等算法库，能够运用Tensorflow、Keras、Pytorch等开源框架实现RNN、CNN系列神经网络模型搭建，熟悉常用模型如：Transformer、GPT、 Bert，了解计算机视觉相关算法如：Yolo系列，SSD，RCNN系列等，具备基于⽬标数据集微调参数、实现迁移学习的能力，具有英文文档阅读能力





