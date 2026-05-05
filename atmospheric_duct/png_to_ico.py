"""将PNG图标转换为多尺寸ICO文件（兼容PyInstaller）"""
import struct
from PIL import Image
from io import BytesIO
import os

def make_bmp_icondata(rgba_img):
    """RGBA图像转为ICO内部BMP格式（BITMAPINFOHEADER + BGRA像素）"""
    iw, ih = rgba_img.size
    header = struct.pack('<IiiHHIIiiII', 40, iw, ih * 2, 1, 32, 0, 0, 0, 0, 0, 0)
    px = rgba_img.load()
    xor_mask = bytearray()
    for yy in range(ih - 1, -1, -1):
        for xx in range(iw):
            r, g, b, a = px[xx, yy]
            xor_mask.extend([b, g, r, a])
    row_bytes = (iw + 31) // 32 * 4
    return bytes(header) + bytes(xor_mask) + bytes(bytearray(row_bytes * ih))

def make_png_icondata(rgba_img):
    """RGBA图像转为PNG格式（用于256x256大尺寸）"""
    buf = BytesIO()
    rgba_img.save(buf, format='PNG')
    return buf.getvalue()

def png_to_ico(png_path, ico_path=None):
    if ico_path is None:
        ico_path = os.path.splitext(png_path)[0] + '.ico'

    img = Image.open(png_path).convert('RGBA')

    # 256用PNG压缩，其余用BMP（Windows标准做法）
    sizes_spec = [(256, 'PNG'), (128, 'BMP'), (64, 'BMP'), (48, 'BMP'), (32, 'BMP'), (16, 'BMP')]
    entries, data_list = [], []
    for size, fmt in sizes_spec:
        resized = img.resize((size, size), Image.LANCZOS)
        d = make_png_icondata(resized) if fmt == 'PNG' else make_bmp_icondata(resized)
        entries.append({'size': size, 'len': len(d)})
        data_list.append(d)

    with open(ico_path, 'wb') as f:
        # ICO头: reserved(2) + type=1(2) + count(2)
        f.write(struct.pack('<HHH', 0, 1, len(sizes_spec)))
        offset = 6 + 16 * len(sizes_spec)
        for e in entries:
            wh = e['size'] if e['size'] < 256 else 0
            f.write(struct.pack('<BBBBHHII', wh, wh, 0, 0, 1, 32, e['len'], offset))
            offset += e['len']
        for d in data_list:
            f.write(d)

    print(f'ICO已生成: {ico_path} ({os.path.getsize(ico_path)} 字节, {len(sizes_spec)} 个尺寸)')

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(dir_path, '电离层反射预备图标.png')
    png_to_ico(png_path)
