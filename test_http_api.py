# -*- coding: utf-8 -*-

import requests
import json
import time
import base64

url = "http://10.21.132.6:8080/api/large_model_post"
#url = "http://10.21.132.6:80/api/large_model_stream"

#fileurl = 'https://lins-oss-dev-test.sensoro.com/lsv2/lins-file/11397994308890703547897.jpg?resourceId=1777966085814558722&secret=a372555de9b612bb8dcc74d8979f7818'
fileurl = 'https://vsc-get.xn.sensoro.vip/1year/20240410/capture/DFC-RZ-06680017C76EFCA6/1712746052_21368_RL_YT.jpg?signature=8005fad72ecb849fa1cc7ed5d8c9d776dc511c6c90ed58eb2b7818befe375c5e752033faaaecac368fdb3b1b9002b167329e0c8c81344c6be060df4d0c06692da6833293162e5e0f0b254a57876dae0b98d40ec9fdda479185c901cd78ee8c3d44cf5cf06abc99dd764698c1a6487e61c295200f98245c90a25439042bdc2bd55bb0532e645e78f12a0297e34ef8862b&expires=604799'
fileurl = 'https://vsc-get.xn.sensoro.vip/DFC-RZ-07520017C753FBB8/1712892059_8434098_RT_YT.jpg?signature=8005fad72ecb849fa1cc7ed5d8c9d776dc511c6c90ed58eb2b7818befe375c5e752033faaaecac368fdb3b1b9002b1679658926e8b72eaa705361d3a6f7360295fd792126e9be078cf4b742fc32442e1289d0e0b3e508a96c07e584a4ec74483fea516cab9267e4b4ee1c24050269cc8&expires=604799'
headers = \
    {
        
        "applicationCode": "detection",
        "operationTime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,
        "Content-Type": "application/json;charset=UTF-8"
    }
body = \
    {
        "file_prompt": fileurl,
        #"input_text": "请描述这张图片",
        "input_text": "分析图片中的内容，识别并描述图片的细节，包括人物特征（如数量、外观、年龄、服装、服装颜色、服装品牌、表情和行为）、车辆特征（车辆外形、车辆颜色、车辆品牌）、场景布局（比如地点、物体和活动）、以及时间和可能的情境背景。请为我提供一份简要的报告，总结图片中的关键元素。最后，输出的时候把内容总结成一段话。",
        "streamer_out": 'False'
    }
#r = requests.post(url,headers=headers,json=body)
r = requests.post(url,json=body)
#r = requests.post(url,json=body, stream=True)

#for line in r.iter_lines():
#    if line:
#        print(line.decode())
#print(r.iter_content())
#for chunk in r.iter_content():
    #temp_data += chunk.decode('utf-8')
    #temp_data += chunk.decode('gb2312')
    #data_decode = chunk.decode('utf-8')
    #print('temp_data===', data_decode)
    #print(chunk.decode('ascii').encode('utf-8'))
#print(r.json()['result'])
print(r)
#print(r.json()['ret_code'], r.json()['result'])
response=r.content
print(response)
json_resp=json.loads(response)
print(json_resp)
print(json_resp['code'], json_resp['result'])

