import unittest
import tempfile
from pathlib import Path
from mac.photo_service import check_duration, reserve_job, release_job, new_job_dir

class PhotoTests(unittest.TestCase):
    def test_audio_duration_limits(self):
        for duration in (0, .49, 5.01, float('nan'), float('inf')):
            with self.assertRaises(ValueError): check_duration(duration)
        self.assertEqual(check_duration(1.25), 1.25)
    def test_unique_directories(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertNotEqual(new_job_dir(Path(root)), new_job_dir(Path(root)))
    def test_single_job(self):
        reserve_job()
        try:
            with self.assertRaises(ValueError): reserve_job()
        finally: release_job()
        reserve_job(); release_job()

if __name__ == '__main__': unittest.main()

class InputTests(unittest.TestCase):
    def test_invalid_image_and_long_audio_rejected(self):
        import subprocess
        from mac.photo_service import prepare_inputs
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); img=root/'face.png'; audio=root/'long.wav'
            Image.new('RGB',(32,32)).save(img)
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:duration=6',str(audio)],check=True)
            job=new_job_dir(root/'jobs')
            with self.assertRaises(ValueError): prepare_inputs(str(img),str(audio),job)
    def test_normalizes_valid_input(self):
        import subprocess
        from mac.photo_service import prepare_inputs
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); img=root/'face.png'; audio=root/'short.wav'
            Image.new('RGB',(32,32)).save(img)
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:duration=1',str(audio)],check=True)
            job=new_job_dir(root/'jobs')
            self.assertAlmostEqual(prepare_inputs(str(img),str(audio),job),1,places=2)
            self.assertTrue((job/'portrait.png').exists())
            self.assertTrue((job/'config.yaml').exists())
    def test_cancel_stops_only_registered_process(self):
        import subprocess, sys
        from mac import photo_service as service
        proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
        try:
            service.PROCESS=proc
            service.cancel_job()
            proc.wait(timeout=5)
            self.assertNotEqual(proc.returncode,0)
        finally:
            service.PROCESS=None
            if proc.poll() is None: proc.kill()
