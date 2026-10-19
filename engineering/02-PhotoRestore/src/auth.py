"""Local SQLite accounts and revocable opaque-cookie sessions."""
import hashlib,re,secrets,sqlite3,time,threading
from pathlib import Path
from contextlib import contextmanager
from urllib.parse import urlsplit
from flask import request,jsonify,g,redirect
from werkzeug.security import generate_password_hash,check_password_hash
COOKIE='photorestore_session'
failed={};rate_lock=threading.Lock()

@contextmanager
def connect(app):
 path=Path(app.config['AUTH_DB']);path.parent.mkdir(parents=True,exist_ok=True)
 db=sqlite3.connect(path,timeout=10);db.row_factory=sqlite3.Row
 db.executescript("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY,username TEXT NOT NULL,normalized TEXT NOT NULL UNIQUE,password_hash TEXT NOT NULL,created REAL NOT NULL);
 CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY,user_id INTEGER NOT NULL,expires REAL NOT NULL);
 CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);""")
 try:
  with db:yield db
 finally:db.close()

def token_hash(token):return hashlib.sha256(token.encode()).hexdigest()
def user_dict(user):return {'id':user['id'],'username':user['username']}

def init_auth(app,root):
 app.config.setdefault('AUTH_DB',str(root/'appdata/accounts.sqlite3'))
 dummy=generate_password_hash(secrets.token_urlsafe(24),method='scrypt')
 def reject_origin():
  origin=request.headers.get('Origin')
  return origin and (origin=='null' or urlsplit(origin).netloc!=request.host or urlsplit(origin).scheme!=request.scheme)
 def get_user():
  token=request.cookies.get(COOKIE)
  if not token or len(token)>128:return None
  with connect(app) as db:
   row=db.execute('SELECT u.id,u.username FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires>?',(token_hash(token),time.time())).fetchone()
   return dict(row) if row else None
 @app.before_request
 def protect():
  g.user=None
  if request.path.startswith('/api/') and request.method in ('POST','DELETE','PUT','PATCH'):
   if reject_origin() or request.headers.get('Sec-Fetch-Site')=='cross-site':return jsonify(error='请从本网站页面提交请求'),403
  if request.path.startswith('/api/jobs') or request.path.startswith('/api/auth') or request.path in ('/','/index.html','/login'):
   g.user=get_user()
  if request.path.startswith('/api/jobs') and not g.user:return jsonify(error='请先登录',login_required=True),401
  if request.path in ('/','/index.html') and not g.user:return redirect('/login')
 @app.get('/login')
 def login_page():return redirect('/') if g.user else app.send_static_file('login.html')
 @app.get('/api/auth/me')
 def me():return jsonify(authenticated=bool(g.user),user=g.user)
 def payload():
  if not request.is_json:return None
  data=request.get_json(silent=True)
  return data if isinstance(data,dict) else None
 def issue(user,remember):
  token=secrets.token_urlsafe(32);age=7*86400 if remember else 12*3600
  with connect(app) as db:
   db.execute('DELETE FROM sessions WHERE expires<=?',(time.time(),))
   old=request.cookies.get(COOKIE)
   if old:db.execute('DELETE FROM sessions WHERE token_hash=?',(token_hash(old),))
   db.execute('INSERT INTO sessions VALUES(?,?,?)',(token_hash(token),user['id'],time.time()+age))
  response=jsonify(user=user_dict(user))
  response.set_cookie(COOKIE,token,max_age=age if remember else None,httponly=True,samesite='Lax',secure=request.is_secure,path='/')
  return response
 @app.post('/api/auth/register')
 def register():
  data=payload()
  if data is None:return jsonify(error='请使用有效的 JSON 请求'),400
  name=data.get('username');password=data.get('password')
  if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_\u4e00-\u9fff]{3,24}',name.strip()):return jsonify(error='用户名需为 3–24 位中文、字母、数字或下划线'),400
  name=name.strip()
  if not isinstance(password,str) or not 8<=len(password)<=128:return jsonify(error='密码长度需为 8–128 位'),400
  hashed=generate_password_hash(password,method='scrypt')
  try:
   with connect(app) as db:
    cur=db.execute('INSERT INTO users(username,normalized,password_hash,created) VALUES(?,?,?,?)',(name,name.casefold(),hashed,time.time()))
    user={'id':cur.lastrowid,'username':name}
  except sqlite3.IntegrityError:return jsonify(error='该用户名已注册，请登录或更换用户名'),409
  return issue(user,bool(data.get('remember'))),201
 @app.post('/api/auth/login')
 def login():
  data=payload()
  if data is None:return jsonify(error='请使用有效的 JSON 请求'),400
  name=data.get('username');password=data.get('password')
  if not isinstance(name,str) or not isinstance(password,str) or len(name)>128 or len(password)>128:return jsonify(error='用户名或密码不正确'),400
  key=(request.remote_addr or '',name.strip().casefold());now=time.time()
  with rate_lock:
   for k,v in list(failed.items()):
    if now-v[1]>=600:del failed[k]
   if failed.get(key,(0,0))[0]>=8:return jsonify(error='尝试次数过多，请 10 分钟后重试'),429
  with connect(app) as db:user=db.execute('SELECT * FROM users WHERE normalized=?',(name.strip().casefold(),)).fetchone()
  valid=check_password_hash(user['password_hash'] if user else dummy,password)
  if not user or not valid:
   with rate_lock:
    if len(failed)>2048:failed.clear()
    failed[key]=(failed.get(key,(0,now))[0]+1,now)
   return jsonify(error='用户名或密码不正确'),401
  with rate_lock:failed.pop(key,None)
  return issue(user,bool(data.get('remember')))
 @app.post('/api/auth/logout')
 def logout():
  token=request.cookies.get(COOKIE)
  if token:
   with connect(app) as db:db.execute('DELETE FROM sessions WHERE token_hash=?',(token_hash(token),))
  response=jsonify(ok=True);response.delete_cookie(COOKIE,path='/',httponly=True,samesite='Lax');return response
