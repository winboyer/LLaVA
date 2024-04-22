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


model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=args.model_path, 
    model_base=args.model_base, 
    model_name=model_name,
    load_8bit=args.load_8bit, 
    load_4bit=args.load_4bit
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


# # check device for weights if u want to
# for n, p in model.named_parameters():
#     print(f"{n}: {p.device}")

early_stop = False

def process_video(file_path, interval_time=4.0):
    videoCapture = cv2.VideoCapture(file_path)
    success, frame = videoCapture.read()
    frame_cnt = 0
    # 视频中，1s包含25帧
    ret_list = []
    while success :
        frame_cnt = frame_cnt + 1
        if frame_cnt%(25*interval_time) != 0: # 隔4s抽一帧
            success, frame = videoCapture.read()
            continue
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # pil_img = pil_img.convert('RGB')
        ret_list.append(pil_img)
        success, frame = videoCapture.read()
    print('frame_cnt, len(ret_list)=========', frame_cnt, len(ret_list))
    videoCapture.release()

    return ret_list


default_chatbox = [("", "Hi, What do you want to know about from llava?")]
def http_post(
        input_text,
        temperature,
        top_p,
        token_len,
        file_prompt,
        result_previous,
        hidden_image,
        ):
    result_text = [(ele[0], ele[1]) for ele in result_previous]
    gallery_prompt = []
    for i in range(len(result_text)-1, -1, -1):
        if result_text[i][0] == "" or result_text[i][0] == None:
            del result_text[i]
    print(f"history {result_text}")

    # filepath_extname = os.path.splitext(file_prompt.name)[-1]
    filepath_extname = os.path.splitext(file_prompt)[-1]
    print('filepath_extname, file_prompt.name============', filepath_extname, file_prompt)
    video_flag = filepath_extname in ('.mp4', 'avi', '.ts', '.mpg', '.mpeg', '.rm', '.rmvb', '.mov', '.wmv')
    global early_stop
    early_stop = False
    try:
        if video_flag:
            print('web_video_chat')
            image_list = process_video(file_prompt.name)

            ret_img_list = []
            response_list = []
            ret_cnt=0
            for pil_img in image_list:
                if early_stop:
                    break

                qs = input_text
                image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                if IMAGE_PLACEHOLDER in qs:
                    if model.config.mm_use_im_start_end:
                        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                    else:
                        qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
                else:
                    if model.config.mm_use_im_start_end:
                        qs = image_token_se + "\n" + qs
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                images = [pil_img]
                images_tensor = process_images(
                    images,
                    image_processor,
                    model.config
                ).to(model.device, dtype=torch.float16)

                input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda()
                )

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=images_tensor,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        top_p=top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=token_len,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                    )

                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(
                        f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
                    )
                outputs = tokenizer.batch_decode(
                    output_ids[:, input_token_len:], skip_special_tokens=True
                )[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[: -len(stop_str)]
                outputs = outputs.strip()
                print(outputs)
                if ' no ' in outputs or 'not visible' in outputs:
                    continue
                chinese_flag = False
                for _char in outputs:
                    if '\u4e00' <= _char <= '\u9fa5':
                        print('this is chinese sentence')
                        chinese_flag = True
                        break
                if chinese_flag:
                    doc = nlp_zh(outputs)
                else:
                    doc = nlp(outputs)
                
                noun_phrases = []
                boxes = []
                box_matches = list(re.finditer(r'\[\[([^\]]+)\]\]', outputs))
                box_positions = [match.start() for match in box_matches]
                for match, box_position in zip(box_matches, box_positions):
                    nearest_np_start = max([0] + [chunk.start_char for chunk in doc.noun_chunks if chunk.end_char <= box_position])
                    noun_phrase = outputs[nearest_np_start:box_position].strip()
                    if noun_phrase and noun_phrase[-1] == '?':
                        noun_phrase = outputs[:box_position].strip()
                    box_string = match.group(1)
                    
                    noun_phrases.append(noun_phrase)
                    boxes.append(boxstr_to_boxes(box_string))

                if len(boxes)<1:
                    gallery_prompt.append(pil_img)
                else:
                    print('draw image not implement yet !!!')
                    print('boxes========', boxes)
                    gallery_prompt.append(pil_img)

                # answer = 'image '+str(ret_cnt)+': '+response
                result_text.append((input_text, outputs))
                # print(result_text)

                yield "", result_text, hidden_image, gallery_prompt

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
            
            # qs = input_text
            # image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

            inp = input_text
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

            conv = conv_templates[args.conv_mode].copy()
            # conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[0], inp)
            image = None
            
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            print('prompt=================', prompt)

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
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
            print('input_text=============', input_text)
            print('outputs=============', outputs)
            result_text.append((input_text, outputs))
            yield "", result_text, hidden_image, gallery_prompt

            
    except Exception as e:
        print("error message", e)
        result_text.append((input_text, 'Timeout! Please wait a few minutes and retry.'))
        # return "", result_text, hidden_image, gallery_prompt
        return "", result_text, hidden_image, gallery_prompt

    return "", result_text, hidden_image, gallery_prompt


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


def clear_fn(value):
    global early_stop
    early_stop = True
    return "", default_chatbox, None, None

def clear_fn2(value):
    return default_chatbox, None


def imagefile_upload_fn(file_prompt):
    filepath_extname = os.path.splitext(file_prompt.name)[-1]
    # print('filepath_extname, file_prompt.name============', filepath_extname, file_prompt.name)
    video_flag = filepath_extname in ('.mp4', 'avi', '.ts', '.mpg', '.mpeg', '.rm', '.rmvb', '.mov', '.wmv')
    if not video_flag:
        return default_chatbox, file_prompt.name
    else:
        return default_chatbox, None

def change_gallery(result_text, gallery_prompt):

    return result_text, gallery_prompt

if __name__ == '__main__':
    gr.close_all()
    examples = []

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=7):
                result_text = gr.components.Chatbot(label='Conversation History', 
                                                # label='Multi-round conversation History', 
                                                value=[("", "Hi, What do you want to know about from llava?")], 
                                                height=500)
                hidden_image_hash = gr.Textbox(visible=False)

            with gr.Column(scale=4):
                file_prompt = gr.Image(type="filepath", label="Image Prompt", value=None)
                # video_prompt = gr.Video(type="file", label="Video Prompt", value=None)
                # file_prompt = gr.File(label="file prompt", 
                # # file_types=["image",".mp4",".ts",".avi",".mpg",".mpeg",".rm",".rmvb",".mov",".wmv"], 
                # file_types=[".mp4",".ts",".avi",".mpg",".mpeg",".rm",".rmvb",".mov",".wmv"], 
                # value=None)
                gallery_prompt = gr.Gallery(label='chat image', height=300)

        with gr.Group():
            input_text = gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.')
            with gr.Row():
                run_button = gr.Button('Generate')
                clear_button = gr.Button('Clear')

            with gr.Row():
                temperature = gr.Slider(maximum=1, value=0.2, minimum=0, label='Temperature')
                top_p = gr.Slider(maximum=1, value=0.7, minimum=0, label='Top P')
                token_len = gr.Slider(maximum=1024, value=1024, minimum=0, step=64, label='Max output tokens')
            
        print(gr.__version__)
        run_button.click(fn=http_post,inputs=[input_text, temperature, top_p, token_len, file_prompt, result_text, hidden_image_hash],
                            outputs=[input_text, result_text, hidden_image_hash, gallery_prompt])
        input_text.submit(fn=http_post,inputs=[input_text, temperature, top_p, token_len, file_prompt, result_text, hidden_image_hash],
                            outputs=[input_text, result_text, hidden_image_hash, gallery_prompt])
        clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[input_text, result_text, file_prompt, gallery_prompt])
        file_prompt.upload(fn=clear_fn2, inputs=file_prompt, outputs=[result_text, gallery_prompt])
        # file_prompt.upload(fn=imagefile_upload_fn, inputs=file_prompt, outputs=[result_text, gallery_prompt])
        # file_prompt.clear(fn=clear_fn2, inputs=clear_button, outputs=[result_text, gallery_prompt])
        file_prompt.clear(fn=clear_fn2, inputs=file_prompt, outputs=[result_text, gallery_prompt])

        # gallery_prompt.change(fn=change_gallery, inputs=[result_text, gallery_prompt], outputs=[result_text, gallery_prompt])

        print(gr.__version__)

    demo.queue(concurrency_count=10)
    # demo.launch(share=True)
    demo.launch(server_name="0.0.0.0", show_error=True, server_port=args.server_port)
    # demo.launch(server_name="0.0.0.0", server_port=8088-8089)


