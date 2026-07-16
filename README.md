# 电气故障诊断多智能体RAG系统
## 项目简介
基于LangChain搭建分层智能体工作流，使用Chroma向量数据库搭建行业知识库，针对通义千问7B开源大模型完成行业微调、4bit量化轻量化部署，实现建筑电气设备故障智能问答诊断。
## 技术栈
Python3.10、LangChain、Chroma向量库、Qwen-7B、Transformers、bitsandbytes量化、FastAPI
## 项目模块
1. 多智能体工作流：文档解析、向量入库、检索决策、故障推理四层智能体调度
2. 模型微调模块：电气故障私有数据集监督微调脚本
3. 模型量化模块：4/8bit低精度量化，低配设备本地推理
4. 后端接口服务：FastAPI实现线上故障诊断问答
## 配套实战项目
睿抗机器人大赛AI视觉项目，任务完成度94%，独立完成数据集标注、模型训练、迭代调优全流程
## 项目运行指南
1. 安装项目全部依赖库
pip install -r requirements.txt

2. 加载电气故障文档，构建向量知识库
python build_db.py

3. 启动4bit量化Qwen大模型本地推理服务
python run_qwen.py

4. 启动FastAPI后端接口，进行故障智能问答诊断
python app.py
<img width="1919" height="859" alt="fb253db88b88ca5113d0e3d48daa67b2" src="https://github.com/user-attachments/assets/07dd1016-b85d-4c2d-8074-86dca736afdb" />
<img width="1448" height="1012" alt="b6eb188d0d8d241d27bb625c1bddbf3d" src="https://github.com/user-attachments/assets/7965b90d-aa17-44b1-93c6-c08249dc0b56" />
