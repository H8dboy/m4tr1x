import http.server
import socketserver

# Porta su cui girerà il sito m4tr1x locale
PORT = 8080

class M4tr1xHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        # Interfaccia semplicissima per i brosky in strada
        html = """
        <html>
        <body style="background:black; color:lime; font-family:monospace; text-align:center;">
            <h1>🛡️ M4TR1X MESH NODE 🛡️</h1>
            <p>Connessione Protetta Localmente</p>
            <form enctype="multipart/form-data" method="POST">
                <input type="file" name="video" style="background:lime; color:black; padding:20px;">
                <br><br>
                <input type="submit" value="UPLOAD EVIDENCE" style="font-size:20px;">
            </form>
            <p>L'evidenza verrà criptata e salvata su questo nodo.</p>
        </body>
        </html>
        """
        self.wfile.write(html.encode())

print(f"--- M4TR1X MESH ATTIVO ---")
print(f"Dì ai brosky di connettersi a: http://tuo-ip-locale:{PORT}")

with socketserver.TCPServer(("", PORT), M4tr1xHandler) as httpd:
    httpd.serve_forever()
