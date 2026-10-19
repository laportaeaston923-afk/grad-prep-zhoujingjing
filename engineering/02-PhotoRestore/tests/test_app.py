import unittest,sys,io,threading,time,tempfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import server

def png(size=(80,50),mode='RGB'):
 b=io.BytesIO();Image.new(mode,size,'red' if mode=='RGB' else (255,0,0,128)).save(b,format='PNG');return b.getvalue()
class AppChecks(unittest.TestCase):
 def setUp(self):
  (ROOT/'runtime').mkdir(exist_ok=True)
  self.temp=tempfile.TemporaryDirectory(dir=ROOT/'runtime')
  self.old_db=server.app.config['AUTH_DB'];server.app.config['AUTH_DB']=str(Path(self.temp.name)/'accounts.sqlite3')
  self.client=server.app.test_client()
  response=self.client.post('/api/auth/register',json={'username':'test_user','password':'Test-password-123'})
  self.user_id=response.json['user']['id']
 def tearDown(self):
  server.app.config['AUTH_DB']=self.old_db;self.temp.cleanup()
 def test_image_resize_and_alpha(self):
  im,size=server.read_image(png((800,400)),192);self.assertEqual(size,(800,400));self.assertEqual(im.size,(192,96))
  im,_=server.read_image(png((80,50),'RGBA'),192);self.assertEqual(im.mode,'RGB');self.assertGreater(im.getpixel((0,0))[1],100)
 def test_exif_orientation(self):
  b=io.BytesIO();im=Image.new('RGB',(80,40));exif=Image.Exif();exif[274]=6;im.save(b,format='JPEG',exif=exif)
  out,size=server.read_image(b.getvalue(),192);self.assertEqual(size,(40,80));self.assertEqual(out.size,(40,80));self.assertFalse(out.getexif())
 def test_reject_bad_images(self):
  for value in [b'not an image',png((8,8))]:
   with self.assertRaises(ValueError):server.read_image(value,192)
 def test_parameter_validation(self):
  for params in [{'seed':'NaN'},{'max_edge':'999'},{'strength':'101'}]:
   response=self.client.post('/api/jobs',data={**params,'image':(io.BytesIO(png()),'test.png')});self.assertEqual(response.status_code,400)
  self.assertEqual(self.client.post('/api/jobs',data={}).status_code,400)
  self.assertEqual(self.client.get('/api/jobs/not-existing/output.png').status_code,404)
 def test_upload_size_limit(self):
  response=self.client.post('/api/jobs',data=b'x'*(10*1024*1024+1),content_type='multipart/form-data');self.assertEqual(response.status_code,413)
 def test_queue_capacity(self):
  with patch.dict(server.jobs,{'a':{'status':'running','updated':time.time()},'b':{'status':'queued','updated':time.time()},'c':{'status':'queued','updated':time.time()}},clear=True):
   response=self.client.post('/api/jobs',data={'image':(io.BytesIO(png()),'test.png')});self.assertEqual(response.status_code,429)
 def test_cancel_request(self):
  with patch.dict(server.jobs,{'test':{'id':'test','status':'running','cancel':False,'user_id':self.user_id}},clear=True):
   response=self.client.delete('/api/jobs/test');self.assertEqual(response.status_code,200);self.assertTrue(server.jobs['test']['cancel'])
 def test_v3_uses_matching_noise_schedule(self):
  from omegaconf import OmegaConf
  from engine import instantiate
  cfg=OmegaConf.load(ROOT/'vendor/resshift/configs/realsr_swinunet_realesrgan256_journal.yaml')
  diffusion=instantiate(cfg.diffusion)
  self.assertEqual(diffusion.num_timesteps,4)
  self.assertAlmostEqual(float(diffusion.sqrt_etas[0]),.1,places=7)
  import json
  names=[r['name'] for r in json.loads((ROOT/'weights/manifest.json').read_text())]
  self.assertIn('resshift_realsrx4_s4_v3.pth',names)
 def test_attention_equivalence(self):
  import torch
  from engine import efficient_attention
  from ldm.modules.diffusionmodules.model import AttnBlock
  torch.manual_seed(42);block=AttnBlock(32).eval();x=torch.randn(1,32,4,5)
  with torch.inference_mode():a=block(x);b=efficient_attention(block,x)
  torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
if __name__=='__main__':unittest.main()
