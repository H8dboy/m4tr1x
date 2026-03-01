# M4TR1X - The Unfiltered Eye

> Social decentralizzato open-source con AI verification integrata.
> > Proteggi, verifica e condividi video autentici.
> >
> > ## Cosa fa M4TR1X
> >
> > M4TR1X e' un social network decentralizzato basato su protocollo Nostr, progettato per la preservazione e verifica di prove video. Ogni video caricato viene analizzato da un'AI che determina se e' autentico o generato artificialmente.
> >
> > ## Architettura
> >
> > ```
> > Frontend (index.html)          ->  PWA mobile-first
> >     |                               Nostr auth (auth.html)
> >     v
> > API Server (api_server.py)     ->  FastAPI + CORS
> >     |
> >     v
> > AI Detector (ai_detector.py)   ->  EfficientNet-B0 / ONNX
> >     |
> >     v
> > Nostr Attestation              ->  NIP-78 evento firmato
> > ```
> >
> > ## Moduli
> >
> > - **AI Detector** (`ai_detector.py`): Analisi frame-by-frame con EfficientNet-B0. Estrae 16 frame equidistanti, li classifica (REAL/AI_GENERATED) e produce un verdetto aggregato con confidence score e consistency.
> >
> > - - **API Server** (`api_server.py`): Backend FastAPI con endpoint REST. Upload video, analisi AI, attestazione Nostr. CORS abilitato per frontend mobile.
> >  
> >   - - **Training** (`train_detector.py`): Script di addestramento con data augmentation, cosine annealing LR, early stopping. Supporta dataset custom.
> >    
> >     - - **Metadata Scrubbing** (`core.py`): Rimozione GPS/EXIF tramite ExifTool.
> >      
> >       - - **Video Encryption** (`uploader.py`): Crittografia Fernet per protezione file.
> >        
> >         - - **Mesh Server** (`mesh_server.py`): Nodo locale per comunicazione P2P.
> >          
> >           - - **Frontend** (`index.html`): Interfaccia web mobile-first con tema Matrix.
> >            
> >             - - **Nostr Auth** (`auth.html`): Autenticazione decentralizzata via Nostr.
> >              
> >               - ## Quick Start
> >              
> >               - ### Requisiti
> >               - - Python 3.10+
> >                 - - ExifTool (`sudo apt install libimage-exiftool-perl`)
> >
> > ### Installazione
> >
> > ```bash
> > git clone https://github.com/H8dboy/m4tr1x.git
> > cd m4tr1x
> > pip install -r requirements.txt
> > ```
> >
> > ### Avvio API Server
> >
> > ```bash
> > python api_server.py
> > # Server su http://localhost:8080
> > # Docs API su http://localhost:8080/docs
> > ```
> >
> > ### Analisi video da CLI
> >
> > ```bash
> > python ai_detector.py video.mp4
> > ```
> >
> > ### Training modello custom
> >
> > ```bash
> > # 1. Prepara dataset
> > mkdir -p data/train/real data/train/ai_generated
> > mkdir -p data/val/real data/val/ai_generated
> > # Inserisci frame/immagini nelle cartelle
> >
> > # 2. Addestra
> > python train_detector.py --epochs 15 --batch-size 16
> >
> > # 3. Esporta ONNX per mobile
> > python ai_detector.py --export-onnx
> > ```
> >
> > ### Docker
> >
> > ```bash
> > docker build -t m4tr1x .
> > docker run -p 8080:8080 m4tr1x
> > ```
> >
> > ## API Endpoints
> >
> > | Metodo | Endpoint | Descrizione |
> > |--------|----------|-------------|
> > | GET | `/` | Health check |
> > | POST | `/api/v1/analyze` | Upload + analisi video |
> > | GET | `/api/v1/analysis/{id}` | Risultato analisi |
> > | GET | `/api/v1/export-onnx` | Export modello ONNX |
> >
> > ### Esempio analisi
> >
> > ```bash
> > curl -X POST http://localhost:8080/api/v1/analyze \
> >   -F "video=@miovideo.mp4" \
> >   -H "X-Nostr-Pubkey: npub1..."
> > ```
> >
> > Risposta:
> > ```json
> > {
> >   "id": "a1b2c3d4e5f6",
> >   "verdict": "AUTHENTIC",
> >   "verdict_emoji": "✅",
> >   "confidence": {"authentic": 0.92, "ai_generated": 0.08},
> >   "consistency_score": 0.95,
> >   "frames_analyzed": 16,
> >   "video_hash_sha256": "abc123...",
> >   "nostr_attestation": {...}
> > }
> > ```
> >
> > ## Struttura Progetto
> >
> > ```
> > m4tr1x/
> > ├── ai_detector.py      # Modulo AI detection
> > ├── api_server.py       # Backend FastAPI
> > ├── train_detector.py   # Training script
> > ├── core.py             # Metadata scrubbing
> > ├── uploader.py         # Video encryption
> > ├── mesh_server.py      # P2P mesh node
> > ├── index.html          # Frontend PWA
> > ├── auth.html           # Nostr authentication
> > ├── requirements.txt    # Dipendenze Python
> > ├── Dockerfile          # Container config
> > ├── models/             # Modelli salvati (.pt, .onnx)
> > └── data/               # Dataset per training
> > ```
> >
> > ## Roadmap
> >
> > - [x] Frontend PWA mobile-first
> > - [ ] - [x] Autenticazione Nostr
> > - [ ] - [x] AI Detection backend (EfficientNet-B0)
> > - [ ] - [x] API REST con FastAPI
> > - [ ] - [x] Export ONNX per mobile
> > - [ ] - [x] Docker deployment
> > - [ ] - [ ] Training su dataset video AI recenti (Sora, Runway, Kling)
> > - [ ] - [ ] Detection in-browser via ONNX Runtime Web
> > - [ ] - [ ] App nativa Android/iOS (React Native + ONNX Mobile)
> > - [ ] - [ ] IPFS pinning automatico per video verificati
> > - [ ] - [ ] Feed Nostr con filtro verificato/non verificato
> > - [ ] - [ ] Relay Nostr dedicato M4TR1X
> >
> > - [ ] ## Licenza
> >
> > - [ ] MIT
> >
> > - [ ] *For the Truth.*
