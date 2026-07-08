import gradio as gr
import re
import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

print("==================================================")
print("1. 正在初始化并加载本地故障向量数据库 (Chroma)...")
print("==================================================")
# 初始化向量模型（保持与 build_db.py 一致）
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# 读取之前生成的数据库
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
print("数据库加载成功！")

print("\n==================================================")
print("2. 正在载入 Qwen2-VL 大模型到显存/内存 (约需30-60秒)...")
print("==================================================")
model_id = "qwen/Qwen2-VL-2B-Instruct"
# 自动检测GPU，如果配置不够会自动切换到内存运行
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)
print("大模型点火成功！核心引擎就绪！")


def diagnose_system(input_text, input_image):
    """
    核心诊断逻辑：RAG检索库 + 视觉大模型结合
    """
    if not input_text and not input_image:
        return "请输入故障现象或上传故障图片。", None

    retrieved_knowledge = ""
    # 1. 如果用户输入了文字，先去向量数据库里捞专业案例
    if input_text:
        print(f"\n[用户提问]: {input_text}")
        docs = db.similarity_search(input_text, k=1)
        if docs:
            retrieved_knowledge = docs[0].page_content
            print(f"[匹配本地案例成功]")

    # 2. 组装输入给大模型的话（把本地专业知识当做标准参考喂给大模型）
    prompt = f"你是一个专业的建筑电气专家。\n"
    if retrieved_knowledge:
        prompt += f"请参考以下我们本地系统的专业故障处理手册：\n{retrieved_knowledge}\n\n"
    prompt += f"请结合以上参考手册和用户提供的信息，对以下故障现象给出详细的诊断分析和维修步骤：\n{input_text if input_text else '分析这张图片中的电气故障'}"

    # 3. 判断是否需要处理图片
    image_path_to_show = None
    messages = [{"role": "user", "content": []}]

    if input_image:
        # 用户直接在网页上传了图片
        messages[0]["content"].append({"type": "image", "image": input_image})
        image_path_to_show = input_image
    else:
        # 如果用户没传图，但是检索到的 TXT 文档里自带图片索引标记 [图名:xxx.jpg]
        img_match = re.search(r'\[图名:(.*?)\]', retrieved_knowledge)
        if img_match:
            img_name = img_match.group(1).strip()
            # 尝试在你的各专业子目录下寻找这张图
            possible_paths = [
                os.path.join("knowledge_base", "knowledge_bas2e配电系统", img_name),
                os.path.join("knowledge_base", "knowledge_base1楼宇自控", img_name),
                os.path.join("knowledge_base", "knowledge_base3弱电安防", img_name),
                os.path.join("knowledge_base", "knowledge_base4PLC", img_name),
                os.path.join("images", img_name),
                img_name
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    image_path_to_show = p
                    messages[0]["content"].append({"type": "image", "image": p})
                    print(f"[多模态联动] 成功拦截图片索引，匹配到本地参考图：{p}")
                    break

    # 补充文字到模型输入
    messages[0]["content"].append({"type": "text", "text": prompt})

    # 4. 运行大模型推理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    # 如果有参考手册，在前端顺便把参考手册原件打印出来，让毕业论文工作量看起来更扎实
    if retrieved_knowledge:
        output_text = f"### 🤖 AI 智能诊断建议：\n{output_text}\n\n---\n### 📚 知识库原始标准手册参考：\n{retrieved_knowledge}"

    return output_text, image_path_to_show


print("\n==================================================")
print("3. 正在启动网页交互界面 (Gradio)...")
print("==================================================")

# 构建精美的毕业设计前端展示界面
with gr.Blocks(title="建筑电气故障诊断系统") as demo:
    gr.Markdown("# 🏢 基于开源大模型的建筑电气设备智能故障诊断系统")
    gr.Markdown("毕业设计系统演示端 —— 集成本地专业故障向量数据库（RAG）与多模态视觉大模型（Qwen2-VL）")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(label="请输入故障现象描述 (例如: 接触器跳动、视频卡顿)", lines=3)
            input_image = gr.Image(label="可直接上传现场故障图片（可选）", type="filepath")
            btn = gr.Button("🚀 开始智能诊断", variant="primary")
        
        with gr.Column(scale=1):
            output_result = gr.Markdown(label="系统诊断与检修方案")
            output_img = gr.Image(label="系统关联调取的故障参考图/原理图")

    btn.click(
        fn=diagnose_system,
        inputs=[input_text, input_image],
        outputs=[output_result, output_img]
    )

if __name__ == "__main__":
    # 启动本地服务器
    demo.launch(inbrowser=True)
