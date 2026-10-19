import unittest,sys,sqlite3,time,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import server,auth
class AuthChecks(unittest.TestCase):
 def setUp(self):
  (ROOT/'runtime').mkdir(exist_ok=True);self.temp=tempfile.TemporaryDirectory(dir=ROOT/'runtime');self.old=server.app.config['AUTH_DB'];server.app.config['AUTH_DB']=str(Path(self.temp.name)/'accounts.sqlite3');auth.failed.clear();self.client=server.app.test_client()
 def tearDown(self):server.app.config['AUTH_DB']=self.old;self.temp.cleanup()
 def register(self,client=None,name='alice_test',remember=False):return (client or self.client).post('/api/auth/register',json={'username':name,'password':'Test-password-123','remember':remember})
 def test_login_redirect_and_api_guard(self):
  self.assertEqual(self.client.get('/').location,'/login');self.assertEqual(self.client.get('/index.html').status_code,302)
  self.assertEqual(self.client.get('/api/jobs/anything').status_code,401);self.assertEqual(self.client.post('/api/jobs',data={}).status_code,401)
  self.assertFalse(self.client.get('/api/auth/me').json['authenticated']);response=self.client.get('/login');self.assertEqual(response.status_code,200);response.close()
 def test_registration_hash_and_cookie(self):
  r=self.register(remember=True);self.assertEqual(r.status_code,201);self.assertIn('HttpOnly',r.headers['Set-Cookie']);self.assertIn('SameSite=Lax',r.headers['Set-Cookie']);self.assertIn('Max-Age=604800',r.headers['Set-Cookie'])
  with auth.connect(server.app) as db:
   value=db.execute('SELECT password_hash FROM users').fetchone()[0];token=db.execute('SELECT token_hash FROM sessions').fetchone()[0]
  self.assertNotEqual(value,'Test-password-123');self.assertTrue(value.startswith('scrypt:'));self.assertNotEqual(token,self.client.get_cookie(auth.COOKIE).value)
  self.assertEqual(self.client.get('/api/auth/me').json['user']['username'],'alice_test');self.assertEqual(self.client.get('/login').status_code,302)
 def test_validation_and_duplicate(self):
  self.assertEqual(self.client.post('/api/auth/register',json={'username':'ab','password':'short'}).status_code,400)
  self.register();self.assertEqual(self.register(name='ALICE_TEST').status_code,409)
  self.assertEqual(self.client.post('/api/auth/register',data={'username':'abc'}).status_code,400)
 def test_login_logout_revokes_cookie(self):
  self.register();self.client.post('/api/auth/logout');self.assertEqual(self.client.post('/api/auth/login',json={'username':'alice_test','password':'wrong'}).status_code,401)
  self.assertEqual(self.client.post('/api/auth/login',json={'username':'ALICE_TEST','password':'Test-password-123'}).status_code,200)
  copied=self.client.get_cookie(auth.COOKIE).value;self.client.post('/api/auth/logout');other=server.app.test_client();other.set_cookie(auth.COOKIE,copied)
  self.assertFalse(other.get('/api/auth/me').json['authenticated']);self.assertEqual(other.get('/api/jobs/anything').status_code,401)
 def test_expired_session(self):
  self.register()
  with auth.connect(server.app) as db:db.execute('UPDATE sessions SET expires=0')
  self.assertEqual(self.client.get('/').status_code,302);self.assertEqual(self.client.get('/api/jobs/x').status_code,401)
 def test_cross_account_job_isolation(self):
  owner=self.register().json['user']['id'];second=server.app.test_client();self.register(second,'other_user')
  with patch.dict(server.jobs,{'private':{'id':'private','user_id':owner,'status':'done','updated':time.time(),'cancel':False}},clear=True):
   self.assertEqual(self.client.get('/api/jobs/private').status_code,200)
   self.assertNotIn('user_id',self.client.get('/api/jobs/private').json)
   for method,path in [('get','/api/jobs/private'),('delete','/api/jobs/private'),('get','/api/jobs/private/output.png')]:self.assertEqual(getattr(second,method)(path).status_code,404)
 def test_cross_origin_rejected(self):
  self.register();self.assertEqual(self.client.post('/api/auth/logout',json={},headers={'Origin':'http://evil.example'}).status_code,403);self.assertTrue(self.client.get('/api/auth/me').json['authenticated'])
 def test_login_throttle(self):
  self.register()
  for _ in range(8):self.assertEqual(self.client.post('/api/auth/login',json={'username':'alice_test','password':'wrong'}).status_code,401)
  self.assertEqual(self.client.post('/api/auth/login',json={'username':'alice_test','password':'wrong'}).status_code,429)
if __name__=='__main__':unittest.main()
