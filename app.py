import gradio as gr
import sys
from starline import process

from utils import load_cn_model, load_cn_config, randomname
from convertor import pil2cv, cv2pil

from sd_model import get_cn_pipeline, get_cn_detector
import cv2
import os
import numpy as np
from PIL import Image
import zipfile
import spaces

path = os.getcwd()
output_dir = f"{path}/output"
input_dir = f"{path}/input"
cn_lineart_dir = f"{path}/controlnet/lineart"

load_cn_model(cn_lineart_dir)
load_cn_config(cn_lineart_dir)
pipe = get_cn_pipeline()


@spaces.GPU(duration=120)
def generate(detectors, prompt, negative_prompt, reference_flg=False, reference_img=None):
    default_pos = ""
    default_neg = ""
    prompt = default_pos + prompt 
    negative_prompt = default_neg + negative_prompt 
    

    if reference_flg==False:
        image = pipe(
                    prompt=prompt,
                    negative_prompt = negative_prompt,
                    image=detectors,
                    num_inference_steps=50,
                    controlnet_conditioning_scale=[1.0, 0.2],
                    ip_adapter_image=None,
                ).images[0]
    else:

        image = pipe(
                    prompt=prompt,
                    negative_prompt = negative_prompt,
                    image=detectors,
                    num_inference_steps=50,
                    controlnet_conditioning_scale=[1.0, 0.2],
                    ip_adapter_image=reference_img,
                ).images[0]
    return image


def zip_png_files(folder_path):
    # Zipファイルの名前を設定（フォルダ名と同じにします）
    zip_path = os.path.join(folder_path, 'output.zip')
    
    # zipfileオブジェクトを作成し、書き込みモードで開く
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # フォルダ内のすべてのファイルをループ処理
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                # PNGファイルのみを対象にする
                if filename.endswith('.png'):
                    # ファイルのフルパスを取得
                    file_path = os.path.join(foldername, filename)
                    # zipファイルに追加
                    zipf.write(file_path, arcname=os.path.relpath(file_path, folder_path))

def resize_image(img, max_size=1024):
    # 画像を開く
    width, height = img.size
    print(f"元の画像サイズ: 幅 {width} x 高さ {height}")
    
    # 縦または横がmax_sizeを超えているかチェック
    if width > max_size or height > max_size:
        # 縦横比を保ちながらリサイズ
        if width > height:
            new_width = max_size
            new_height = int(max_size * height / width)
        else:
            new_height = max_size
            new_width = int(max_size * width / height)
        
        # リサイズ実行
        resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
        print(f"リサイズ後の画像サイズ: 幅 {new_width} x 高さ {new_height}")
        return resized_img
    else:
        return img


class webui:
    def __init__(self):
        self.demo = gr.Blocks()

    def undercoat(self, input_image, pos_prompt, neg_prompt, alpha_th, thickness, reference_flg, reference_img):
        input_image = resize_image(input_image)
        org_line_image = input_image
        image = pil2cv(input_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        index = np.where(image[:, :, 3] == 0)
        image[index] = [255, 255, 255, 255]
        input_image = cv2pil(image)

        detectors = get_cn_detector(input_image.resize((1024, 1024), Image.ANTIALIAS))

        gen_image = generate(detectors, pos_prompt, neg_prompt, reference_flg, reference_img)
        color_img, unfinished = process(gen_image.resize((image.shape[1], image.shape[0]), Image.ANTIALIAS) , org_line_image, alpha_th, thickness)
        #color_img = color_img.resize((image.shape[1], image.shape[0]) , Image.ANTIALIAS)


        output_img = Image.alpha_composite(color_img, org_line_image)
        name = randomname(10)
        if not os.path.exists(f"{output_dir}"):
            os.makedirs(f"{output_dir}")
        os.makedirs(f"{output_dir}/{name}")
        output_img.save(f"{output_dir}/{name}/output_image.png")
        org_line_image.save(f"{output_dir}/{name}/line_image.png")
        color_img.save(f"{output_dir}/{name}/color_image.png")
        unfinished.save(f"{output_dir}/{name}/unfinished_image.png")

        outputs = [output_img, org_line_image, color_img, unfinished]
        zip_png_files(f"{output_dir}/{name}")
        filename = f"{output_dir}/{name}/output.zip"

        return outputs, filename



    def launch(self, share):
        with self.demo:
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", image_mode="RGBA", label="lineart")
                    pos_prompt = gr.Textbox(value="1girl, blue hair, pink shirts, bestquality, 4K", max_lines=1000, label="positive prompt")                    
                    neg_prompt = gr.Textbox(value=" (worst quality, low quality:1.2), (lowres:1.2), (bad anatomy:1.2), (greyscale, monochrome:1.4)", max_lines=1000, label="negative prompt")

                    alpha_th = gr.Slider(maximum = 255, value=100, label = "alpha threshold")
                    thickness = gr.Number(value=5, label="Thickness of correction area (Odd numbers need to be entered)")
                
                    reference_image = gr.Image(type="pil", image_mode="RGB", label="reference_image")
                    reference_flg = gr.Checkbox(value=True, label="reference_flg")
                    #gr.Slider(maximum = 21, value=3, step=2, label = "Thickness of correction area")

                    submit = gr.Button(value="Start")
                with gr.Row():
                    with gr.Column():
                        with gr.Tab("output"):
                            output_0 = gr.Gallery(format="png")
                        output_file = gr.File()
            submit.click(
                self.undercoat, 
                inputs=[input_image, pos_prompt, neg_prompt, alpha_th, thickness, reference_flg, reference_image], 
                outputs=[output_0, output_file]
            )

        self.demo.queue()
        self.demo.launch(share=share)


if __name__ == "__main__":
    ui = webui()
    if len(sys.argv) > 1:
        if sys.argv[1] == "share":
            ui.launch(share=True)
        else:
            ui.launch(share=False)
    else:
        ui.launch(share=False)
