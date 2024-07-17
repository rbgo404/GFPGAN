import os
import cv2
import requests
import base64
from io import BytesIO
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
from PIL import Image

class InferlessPythonModel:
    def initialize(self):
        nfs_volume = os.getenv("NFS_VOLUME")
        
        if os.path.exists(nfs_volume + "GFPGANv1.4.pth") == False :
            os.system(f"wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P {nfs_volume}")
            os.system(f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P {nfs_volume}")
        
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        upsampler = RealESRGANer(scale=4, model_path=f'{nfs_volume}/realesr-general-x4v3.pth', model=model, tile=0, tile_pad=10, pre_pad=0, half=True)
        self.face_enhancer = GFPGANer(model_path=f'{nfs_volume}/GFPGANv1.4.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
    
    def download_img(self,url,filename):
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
            return filename

    def infer(self, inputs):
        img_url = inputs["img_url"]
        scale = inputs["scale"]
        img = self.download_img(img_url,"temp.jpg")

        if scale > 4:
            scale = 4

        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        h, w = img.shape[0:2]

        if h > 3500 or w > 3500:
            return None
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        _, _, output = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        
        if scale != 2:
            interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
            h, w = img.shape[0:2]
            output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
        
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(output)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        base64_image = base64.b64encode(buff.getvalue()).decode('utf-8')

        return {"generated_image":base64_image}
    
    def finalize(self):
       self.face_enhancer = None
