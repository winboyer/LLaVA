import torch
import requests
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from llava.model.builder import load_pretrained_model
from llava.eval.run_llava import eval_model

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.constants import LOGDIR

import hashlib
from transformers import TextStreamer

import json
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import base64
import time
import re
import cv2
import matplotlib.font_manager

import spacy
nlp_zh = spacy.load("zh_core_web_sm")
nlp = spacy.load("en_core_web_sm")

from functools import partial

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--english", action='store_true', help='only output English')
parser.add_argument("--version", type=str, default="chat", help='version to interact with')
parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
parser.add_argument("--no_prompt", action='store_true', help='Sometimes there is no prompt in stage 1')
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true")
parser.add_argument("--server_port", type=int, default=8080, help='the gradio server port')
args = parser.parse_args()


# model_path = "/root/jinyfeng/models/LLaVa/llava-v1.5-13b"
model_path = '/root/jinyfeng/models/LLaVa/llava-v1.6-34b'

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "load_8bit": args.load_8bit,
    "load_4bit": args.load_4bit,
    "conv_mode": None,
    "sep": ",",
    # "temperature": 0,
    "temperature": 0.2,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "server_port": args.server_port
})()

from llava.utils import (build_logger, server_error_msg, violates_moderation, moderation_msg)
from llava.utils import disable_torch_init
disable_torch_init()
from enum import auto, Enum
class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

# # check device for weights if u want to
# for n, p in model.named_parameters():
#     print(f"{n}: {p.device}")

def process_video(file_path, interval_time=1.0):
    videoCapture = cv2.VideoCapture(file_path)
    success, frame = videoCapture.read()
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS)+0.5)
    # timestamp = videoCapture.get(cv2.CAP_PROP_POS_MSEC)
    # print('fps, timestamp========',fps, timestamp)
    frame_cnt = 0
    # 视频中，1s包含25帧
    ret_list = []
    while success :
        frame_cnt = frame_cnt + 1
        # print(fps*interval_time)
        if frame_cnt%(fps*interval_time) != 0: # 隔4s抽一帧
            success, frame = videoCapture.read()
            continue
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # pil_img = pil_img.convert('RGB')
        timestamp = videoCapture.get(cv2.CAP_PROP_POS_MSEC)
        seconds = timestamp//1000
        milliseconds = timestamp%1000
        minutes = 0
        hours = 0
        if seconds >= 60:
            minutes = seconds//60
            seconds = seconds % 60

        if minutes >= 60:
            hours = minutes//60
            minutes = minutes % 60

        print('timestamp,seconds,minutes,hours========',timestamp,seconds,minutes,hours)
        ret_list.append((pil_img,seconds,minutes,hours))
        # if len(ret_list)>5:
        #     break
        success, frame = videoCapture.read()
    print('frame_cnt, len(ret_list)=========', frame_cnt, len(ret_list))
    videoCapture.release()

    return ret_list


default_chatbox = [("", "Hi, What do you want to know about from llava?")]
def http_post(tokenizer, model, image_processor,
        file_prompt,
        input_text,
        temperature,
        top_p,
        token_len,
        video_sampler_seconds=1
        ):

    inp = input_text
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], inp)
    # image = None
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print('prompt=================', prompt)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    start_str = '<|startoftext|>'
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # print('stop_str===========', stop_str)

    result_text = []

    # filepath_extname = os.path.splitext(file_prompt.name)[-1]
    filepath_extname = os.path.splitext(file_prompt)[-1]
    print('filepath_extname, file_prompt============', filepath_extname, file_prompt)
    video_flag = filepath_extname in ('.mp4', 'avi', '.ts', '.mpg', '.mpeg', '.rm', '.rmvb', '.mov', '.wmv')
    try:
        if video_flag:
            print('web_video_chat')
            image_list = process_video(file_prompt, video_sampler_seconds)
            result_text.append((input_text, None))
            for image, clk_sec, clk_min, clk_hour in image_list:
                image_size = image.size
                time_str = '时间: '+str(clk_hour)+':'+str(int(clk_min))+':'+str(int(clk_sec))
                print(time_str)
                # print('image_size====', image_size)
                # Similar operation in model_worker.py
                image_tensor = process_images([image], image_processor, model.config)
                if type(image_tensor) is list:
                    # print('image_tensor is list')
                    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    # print('image_tensor is not list')
                    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=[image_size],
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=token_len,
                        streamer=streamer,
                        use_cache=True)

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                # print('len(outputs)===========', len(outputs))
                outputs = outputs[0].strip()
                if outputs.startswith(start_str):
                    outputs = outputs[len(start_str):]
                if outputs.endswith(stop_str):
                    outputs = outputs[: -len(stop_str)]
                
                savename = file_prompt.split('.')[0]+'_'+time_str+'.jpg'
                image.save(savename)
                # result_text.append((None, savename))
                result_text.append((None, (savename,)))
                result_text.append((None, outputs))

                yield result_text

            # image_path_grounding = './results/output.png'
            # show_image(cache_image, image_path_grounding)
            # result_text.append((None, (image_path_grounding,)))

        else:
            print('web_pic_chat')
            image = load_image(file_prompt)
            image_size = image.size
            # Similar operation in model_worker.py
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=token_len,
                    streamer=streamer,
                    use_cache=True)

            outputs = tokenizer.decode(output_ids[0]).strip()
            # outputs = outputs.strip()
            if outputs.startswith(start_str):
                outputs = outputs[len(start_str):]
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            result_text.append((input_text, outputs))
            yield result_text
            
    except Exception as e:
        print("error message", e)
        result_text.append((input_text, 'Timeout! Please wait a few minutes and retry.'))
        return result_text

    # return result_text


def show_image(img, output_fn='./results/output.png'):
    img = img.convert('RGB')
    width, height = img.size
    ratio = min(1920 / width, 1080 / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    new_img = img.resize((new_width, new_height), Image.LANCZOS)

    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    img_with_overlay = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    img_with_overlay.save(output_fn)

def video_clear_fn(value):
    return None, "", ""

def clear_fn(value):
    return "", default_chatbox, None, None

def clear_fn2(value):
    return default_chatbox, None


if __name__ == '__main__':
    gr.close_all()
    examples = []

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path, 
        model_base=args.model_base, 
        model_name=model_name,
        # load_8bit=args.load_8bit, 
        # load_4bit=args.load_4bit
        # load_8bit=True, 
        load_4bit=True
    )
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('<h1><center>LLava-1.6-34b Model</center></h1>')
        with gr.Row():
            temperature = gr.Slider(maximum=1, value=0.2, minimum=0, label='Temperature')
            top_p = gr.Slider(maximum=1, value=0.7, minimum=0, label='Top P')
            token_len = gr.Slider(maximum=1024, value=1024, minimum=0, step=64, label='Max output tokens')

        with gr.Tab("Video"):
            input_text_video = gr.Textbox(label='Enter the question you want to know',
                                    value='分析图片中的内容，识别并描述图片的细节，包括人物特征（如数量、外观、年龄、服装、服装颜色、服装品牌、表情和行为、是否戴口罩、是否背包、是否抽烟、是否在玩手机）、车辆特征（车辆外形、车辆颜色、车辆品牌）、场景布局（比如地点、物体和活动）、以及时间和可能的情境背景。请为我提供一份简要的报告，总结图片中的关键元素。',
                                    elem_id='textbox')
            with gr.Row():
                with gr.Column(scale=4):
                    # input_video_file = gr.File(label="Input Video", file_types=[".mp4",".ts",".avi",".mpg",".mpeg",".rm",".rmvb",".mov",".wmv"])
                    input_video_file = gr.Video(label="Input Video")
                    sample_seconds = gr.Slider(maximum=120, value=60, minimum=1, step=1, label='Sampling seconds')
                with gr.Column(scale=6):
                    result_text_video = gr.components.Chatbot(label='Conversation History', 
                                                # label='Multi-round conversation History', 
                                                value=[("", "Hi, What do you want to know about?")], 
                                                height=500)
            # hidden_image_hash = gr.Textbox(visible=False)
            with gr.Row():
                video_submit = gr.Button('Submit')
                video_clear = gr.Button('Clear')
        with gr.Tab("Image"):
            input_text_image = gr.Textbox(label='Enter the question you want to know',
                                    value='分析图片中的内容，识别并描述图片的细节，包括人物特征（如数量、外观、年龄、服装、服装颜色、服装品牌、表情和行为、是否戴口罩、是否背包、是否抽烟、是否在玩手机）、车辆特征（车辆外形、车辆颜色、车辆品牌）、场景布局（比如地点、物体和活动）、以及时间和可能的情境背景。请为我提供一份简要的报告，总结图片中的关键元素。',
                                    elem_id='textbox')
            with gr.Row():
                with gr.Column(scale=4):
                    # input_image_file = gr.Image(type='pil', label='Input Image')
                    input_image_file = gr.Image(type='filepath', label='Input Image')      
                with gr.Column(scale=6):
                    result_text_image = gr.components.Chatbot(label='Conversation History', 
                                                # label='Multi-round conversation History', 
                                                value=[("", "Hi, What do you want to know about?")], 
                                                height=500)
            # hidden_image_hash = gr.Textbox(visible=False)
            with gr.Row():
                image_submit = gr.Button('Submit')
                image_clear = gr.Button('Clear')

        print(gr.__version__)
        video_submit.click(partial(http_post, tokenizer, model, image_processor),
                                [input_video_file, input_text_video, temperature, top_p, token_len, sample_seconds],
                                [result_text_video])
        video_clear.click(fn=video_clear_fn, outputs=[input_video_file, input_text_video, result_text_video])
        image_submit.click(partial(http_post, tokenizer, model, image_processor),
                                [input_image_file, input_text_image, temperature, top_p, token_len],
                                [result_text_image])
        image_clear.click(lambda: [[], '', ''], None,
                                [input_image_file, input_text_image, result_text_image])
        
        # run_button.click(fn=http_post,inputs=[input_text, temperature, top_p, token_len, file_prompt],
        #                     outputs=[input_text, result_text, hidden_image_hash, gallery_prompt])
        # input_text.submit(fn=http_post,inputs=[input_text, temperature, top_p, token_len, file_prompt],
        #                     outputs=[input_text, result_text, hidden_image_hash, gallery_prompt])
        # clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[input_text, result_text, file_prompt, gallery_prompt])
        # file_prompt.upload(fn=clear_fn2, inputs=file_prompt, outputs=[result_text, gallery_prompt])
        # # file_prompt.clear(fn=clear_fn2, inputs=clear_button, outputs=[result_text, gallery_prompt])
        # file_prompt.clear(fn=clear_fn2, inputs=file_prompt, outputs=[result_text, gallery_prompt])


    demo.queue(concurrency_count=10)
    # demo.launch(share=True)
    demo.launch(server_name="0.0.0.0", show_error=True, server_port=args.server_port)
    # demo.launch(server_name="0.0.0.0", server_port=8088-8089)


