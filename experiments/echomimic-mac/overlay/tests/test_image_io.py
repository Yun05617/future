import tempfile
import unittest
from pathlib import Path
from PIL import Image
from mac.image_io import load_bgr_image

class ImageReadTests(unittest.TestCase):
    def test_unicode_path_and_detached_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'照片.png'
            Image.new('RGB',(8,7),(255,0,0)).save(p)
            pixels=load_bgr_image(p)
            p.unlink()
            self.assertEqual(pixels.shape,(7,8,3))
            self.assertEqual(pixels[0,0].tolist(),[0,0,255])
            self.assertTrue(pixels.flags.c_contiguous)
    def test_bad_input_fails_clearly(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'壞照片.png';p.write_bytes(b'not an image')
            with self.assertRaisesRegex(ValueError,'無法讀取照片'):load_bgr_image(p)
