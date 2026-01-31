import http.server
import socketserver
import cgi
import os

PORT = 8080

class M4tr1xHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        html = """
        <html><body style="background:black;color:lime;font-family:monospace;text-align:center;padding:50px;">
            <h1>🛡️ M4TR1X MESH NODE 🛡️</h1>
            <p>Connessione Protetta Localmente | Protocollo Alex Pretti</p>
            <form enctype="multipart/form-data" method="POST">
                <input type="file" name="video" style="background:lime;color:black;padding:10px;">
                <input type="submit" value="UPLOAD EVIDENCE" style="padding:10px;cursor:pointer;">
            </form>
        </body></html>
        """
        self.wfile.write(html.encode())

    def do_POST(self):
        if not os.path.exists("uploads"): os.makedirs("uploads")
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD':'POST'})
        file_item = form['video']
        if file_item.filename:
            fn = os.path.basename(file_item.filename)
            with open(f"uploads/{fn}", 'wb') as f:
                f.write(file_item.file.read())
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"EVIDENCE RECEIVED. THE M4TR1X HAS IT NOW.")

print(f"--- M4TR1X MESH ONLINE ---")
with socketserver.TCPServer(("", PORT), M4tr1xHandler) as httpd:
    httpd.serve_forever()
