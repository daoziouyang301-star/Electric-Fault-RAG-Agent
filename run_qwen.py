from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

model_id = 'qwen/Qwen2-VL-2B-Instruct'

print('正在加载模型，请稍候...')
model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype='auto', device_map='auto')
processor = AutoProcessor.from_pretrained(model_id)

print('模型加载成功！')
print(f'CUDA是否可用: {torch.cuda.is_available()}')