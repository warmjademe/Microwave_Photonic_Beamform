"""只服务完整864测试页面及明确列出的下载；旧实验路由返回410。"""
import argparse
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

HERE=Path(__file__).resolve().parent
PUBLIC={'baselines.html':'text/html; charset=utf-8','results-20260926.json':'application/json; charset=utf-8',
        'results-20260926.csv':'text/csv; charset=utf-8','groups-20260926.csv':'text/csv; charset=utf-8',
        'comparisons-20260926.csv':'text/csv; charset=utf-8'}
RETIRED={'/legacy.html','/native-calibration.html','/baselines-20260924.html','/baselines-data.json',
         '/baselines-signals.json','/baselines-results.csv','/review-code.zip',
         '/native-learning-update.html','/native-learning-update-2x.html'}


def main(port=4093):
    manifest=json.loads((HERE/'release.json').read_text())
    if (manifest.get('schema')!='mwp-final864-only-release-v1' or manifest.get('evaluation_environments')!=864
            or manifest.get('methods')!=21 or manifest.get('method_cases')!=308448
            or manifest.get('evaluation_split')!='test' or manifest.get('final_confirmation') is not True
            or manifest.get('historical_public_data') is not False):raise ValueError('非完整最终测试版本')
    assets={}
    for name,mime in PUBLIC.items():
        raw=(HERE/name).read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=manifest['files'][name]:raise ValueError('发布文件身份改变')
        assets['/'+name]=(raw,gzip.compress(raw,mtime=0),mime,digest)
    for route in ['/','/index.html','/baselines']:assets[route]=assets['/baselines.html']
    health=json.dumps(dict(status='ok',application='mwp-device-audit',evaluation_split='test',
        test_environments=864,methods=21,historical_public_data=False,
        baselines_html_sha256=assets['/baselines.html'][3])).encode()

    class Handler(BaseHTTPRequestHandler):
        server_version='PhotonicsResults';sys_version=''
        def do_GET(self):self.respond(False)
        def do_HEAD(self):self.respond(True)
        def respond(self,head):
            path=urlsplit(self.path).path;compressed=False;digest=None
            if path in assets:
                raw,packed,mime,digest=assets[path];compressed='gzip' in self.headers.get('Accept-Encoding','')
                code=200;body=packed if compressed else raw
            elif path=='/healthz':code,body,mime=200,health,'application/json; charset=utf-8'
            elif path in RETIRED or path=='/deep' or path.startswith('/deep/'):
                code,body,mime=410,'旧结果已下线。请访问 / 查看完整864环境的最终测试结果。\n'.encode(),'text/plain; charset=utf-8'
            else:code,body,mime=404,b'Not found\n','text/plain; charset=utf-8'
            self.send_response(code);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer')
            if digest:
                self.send_header('ETag','"'+digest+'"');self.send_header('Vary','Accept-Encoding')
            if compressed:self.send_header('Content-Encoding','gzip')
            if path in assets and path.endswith(('.json','.csv')):self.send_header('Content-Disposition','attachment; filename="'+path[1:]+'"')
            self.end_headers()
            if not head:
                try:self.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError):pass
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(json.dumps(dict(listening='127.0.0.1',port=server.server_port)),flush=True);server.serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=4093);main(p.parse_args().port)
