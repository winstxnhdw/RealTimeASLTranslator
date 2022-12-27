from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from app.translator import Translator


class Server(BaseHTTPRequestHandler):

    def __init__(self, translator: Translator):
        
        self.translator = translator


    def __call__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


    def do_GET(self):
        
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(translator.result.encode('utf-8'))


class HTTPDaemon:

    def __init__(self, host: str, port: int, translator: Translator):

        self.host = host
        self.port = port
        self.httpd = HTTPServer((self.host, self.port), Server(translator))

        self.server_thread: Thread


    def __enter__(self):

        print(f"Serving HTTP on {self.host} port {self.port} (http://{self.host}:{self.port}/)..")
        self.server_thread = Thread(target=self.httpd.serve_forever)
        self.server_thread.start()


    def __exit__(self, *_):
        
        print("\nServer closing..")
        self.httpd.shutdown()
        self.httpd.server_close()
        self.server_thread.join()
