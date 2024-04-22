#python -m llava.serve.controller --host 0.0.0.0 --port 7866 &
#python -m llava.serve.gradio_web_server --controller http://0.0.0.0:7866 --model-list-mode reload &

#CUDA_VISIBLE_DEVICES=2,3 python web_demo_v1.py > web_demo_v1.log
#CUDA_VISIBLE_DEVICES=7 python web_demo_v3.py
#CUDA_VISIBLE_DEVICES=4,5,6,7 python web_demo_v3.py
#CUDA_VISIBLE_DEVICES=0,1,2,3 python web_demo_v4.py

#fp16流式输出版本
#CUDA_VISIBLE_DEVICES=0,1,2,3,4 python web_demo_v4.py
#int4量化流式输出版本
CUDA_VISIBLE_DEVICES=6,7 python web_demo_v4.py

