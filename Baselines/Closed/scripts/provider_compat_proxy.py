import base64,json,mimetypes,os,pathlib,stat,urllib.request
from http.server import BaseHTTPRequestHandler,HTTPServer
P=os.environ['PROVIDER']; M=os.environ['MODEL']; SLUG=os.environ['SLUG']; MAXTOK=int(os.environ.get('MAX_OUTPUT_TOKENS','256')); R=(next((p for p in pathlib.Path(__file__).resolve().parents if (p / "README.md").is_file()), pathlib.Path.cwd()))
k={'openai':('openai_api_key',1,5),'google':('google_gemini_api_key',2,10),'anthropic':('anthropic_api_key',2,10)}[P]; RATE_IN=float(os.environ.get('COST_INPUT_PER_MILLION',k[1])); RATE_OUT=float(os.environ.get('COST_OUTPUT_PER_MILLION',k[2])); MAX_SPEND=float(os.environ.get('MAX_PROVIDER_SPEND_USD','inf')); K=R.parent/'.config/multimodality'/k[0]
if stat.S_IMODE(K.stat().st_mode)&63: raise SystemExit('key mode must be 600')
KEY=K.read_text().strip(); J=R/'Baselines'/'Closed'/SLUG/'runs'/'ch3ef-full1416'/'provider_usage.jsonl'; J.parent.mkdir(parents=True,exist_ok=True); SPENT=sum(json.loads(line).get('estimated_cost_usd',0) for line in J.read_text().splitlines() if line.strip()) if J.exists() else 0.0
def post(u,b,h):
 q=urllib.request.Request(u,data=json.dumps(b).encode(),headers={**h,'Content-Type':'application/json'},method='POST')
 with urllib.request.urlopen(q,timeout=180) as x:return json.loads(x.read()),dict(x.headers)
def invoke(prompt,imgs):
 global SPENT
 if SPENT>=MAX_SPEND: raise RuntimeError('Configured provider spend cap reached before request')
 if P=='openai':
  z=[{'type':'input_text','text':prompt}]
  for b in imgs:
   raw=base64.b64decode(b); mt='image/png' if raw[:4]==b'\x89PNG' else 'image/jpeg'; z.append({'type':'input_image','image_url':'data:'+mt+';base64,'+b,'detail':'high'})
  body={'model':M,'input':[{'role':'user','content':z}],'max_output_tokens':MAXTOK}; effort=os.environ.get('REASONING_EFFORT')
  if effort: body['reasoning']={'effort':effort}
  d,h=post('https://api.openai.com/v1/responses',body,{'Authorization':'Bearer '+KEY})
  t='\n'.join(y.get('text','') for x in d.get('output',[]) for y in x.get('content',[]) if y.get('type')=='output_text'); u=d.get('usage',{}); i=u.get('input_tokens',0); o=u.get('output_tokens',0); rid=d.get('id')
 elif P=='google':
  z=[{'text':prompt}]
  for b in imgs:
   raw=base64.b64decode(b); mt='image/png' if raw[:4]==b'\x89PNG' else 'image/jpeg'; z.append({'inline_data':{'mime_type':mt,'data':b}})
  gconf={'maxOutputTokens':MAXTOK,'responseMimeType':'application/json','thinkingConfig':{'thinkingLevel':'low'}}
  if M!='gemini-3.8-flash': gconf['temperature']=0
  d,h=post('https://generativelanguage.googleapis.com/v1beta/models/'+M+':generateContent',{'contents':[{'role':'user','parts':z}],'generationConfig':gconf},{'x-goog-api-key':KEY})
  t='\n'.join(y.get('text','') for y in d.get('candidates',[{}])[0].get('content',{}).get('parts',[]));u=d.get('usageMetadata',{});i=u.get('promptTokenCount',0);o=u.get('candidatesTokenCount',0)+u.get('thoughtsTokenCount',0);rid=d.get('responseId')
 else:
  z=[{'type':'text','text':prompt}]
  for b in imgs:
   raw=base64.b64decode(b);mt='image/png' if raw[:4]==b'\x89PNG' else 'image/jpeg';z.append({'type':'image','source':{'type':'base64','media_type':mt,'data':b}})
  d,h=post('https://api.anthropic.com/v1/messages',{'model':M,'max_tokens':MAXTOK,'system':'Return only the requested JSON object.','messages':[{'role':'user','content':z}]},{'x-api-key':KEY,'anthropic-version':'2023-06-01'})
  t='\n'.join(y.get('text','') for y in d.get('content',[]) if y.get('type')=='text');u=d.get('usage',{});i=u.get('input_tokens',0);o=u.get('output_tokens',0);rid=d.get('id')
 c=(i*RATE_IN+o*RATE_OUT)/1e6
 with J.open('a') as f:f.write(json.dumps({'request_id':rid,'input_tokens':i,'output_tokens':o,'estimated_cost_usd':c})+'\n')
 SPENT+=c
 return t,i,o
class H(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def sendj(self,n,x):
  b=json.dumps(x).encode();self.send_response(n);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def do_GET(self):self.sendj(200,{'version':'proxy'}) if self.path.endswith('/api/version') else self.sendj(404,{})
 def do_POST(self):
  try:
   p=json.loads(self.rfile.read(int(self.headers['Content-Length'])));ms=p.get('messages',[]);t,i,o=invoke('\n'.join(m.get('content','') for m in ms),[b for m in ms for b in m.get('images',[])])
   self.sendj(200,{'model':M,'message':{'content':t},'prompt_eval_count':i,'eval_count':o,'done_reason':'stop'})
  except Exception as e:self.sendj(502,{'error':str(e)})
HTTPServer(('127.0.0.1',int(os.environ.get('PORT','11436'))),H).serve_forever()
