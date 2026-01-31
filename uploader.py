import os
from cryptography.fernet import Fernet

# 1. Carichiamo la chiave di Nexus (generata da core.py)
def carica_chiave():
    if not os.path.exists("nexus.key"):
        print("ERRORE: Chiave nexus.key non trovata! Generala con core.py")
        return None
    return open("nexus.key", "rb").read()

# 2. Funzione per criptare il video dell'evidenza
def proteggi_evidenza(nome_file):
    key = carica_chiave()
    if not key: return
    
    f = Fernet(key)
    
    with open(nome_file, "rb") as file:
        dati_video = file.read()
    
    # Criptaggio dei dati
    dati_criptati = f.encrypt(dati_video)
    
    # Salvataggio del file "fantasma"
    nome_output = f"GHOST_{nome_file}.nexus"
    with open(nome_output, "wb") as file_output:
        file_output.write(dati_criptati)
    
    print(f"--- SUCCESSO ---")
    print(f"Il video è stato criptato in: {nome_output}")
    print("Ora puoi caricarlo su IPFS o inviarlo via Bluetooth.")
    # Opzionale: cancelliamo il file originale per sicurezza
    # os.remove(nome_file)

if __name__ == "__main__":
    file_da_nascondere = input("Trascina qui il video della protesta: ").strip()
    proteggi_evidenza(file_da_nascondere)
