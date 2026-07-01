import streamlit as st
import cv2
import easyocr
import os
import time  # TAMBAHAN: Diperlukan untuk fitur timer kecepatan AI
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Running Bib Finder", layout="wide")

# --- TAMBAHAN: Fitur Sidebar Informasi Elegan & Profesional ---
with st.sidebar:
    st.markdown("### 🏃‍♂️ Running BIB System")
    # Menggunakan gambar estetik pelari sebagai pemanis sidebar
    st.image("https://images.unsplash.com/photo-1476480862126-209bfaa8edc8?w=500", use_container_width=True)
    st.markdown("---")
    st.markdown("#### 👤 Profil Developer")
    st.write("**Nama:** Arief Daniel A. Silalahi")
    st.write("**NIM:** 2255301019")
    st.markdown("---")
    st.markdown("#### 🛠️ Tech Stack")
    st.code("YOLOv11 Nano\nEasyOCR\nStreamlit Cloud\nSession State Memory", language="text")

st.title("🏃‍♂️ Running Bib Finder")
st.write("Sistem pencarian dokumentasi foto lari otomatis berbasis YOLOv11 & OCR.")
st.write("By. Arief Daniel A. Silalahi")

# 1. INISIALISASI MEMORI (SESSION STATE) agar data tidak hilang saat klik download
if 'found_photos' not in st.session_state:
    st.session_state.found_photos = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'searched' not in st.session_state:
    st.session_state.searched = False
if 'execution_time' not in st.session_state:  # TAMBAHAN: Memori penyimpan durasi scan
    st.session_state.execution_time = 0.0

@st.cache_resource
def load_models():
    model = YOLO('best.pt') 
    reader = easyocr.Reader(['en'], gpu=True)
    return model, reader

model, reader = load_models()

raw_query = st.text_input("🔍 Masukkan Nomor Bib Anda (Termasuk Huruf Jika Ada):")

if st.button("Cari Foto Saya!"):
    search_query = raw_query.strip().upper() 
    
    if search_query:
        st.info("Memindai seluruh foto di dataset... Mohon tunggu ⏳")
        
        # TAMBAHAN: Mulai hitung stopwatch pemindaian
        start_time = time.time()
        
        # Reset memori setiap kali tombol cari diklik baru
        st.session_state.found_photos = []
        st.session_state.last_query = search_query
        st.session_state.searched = True
        
        image_files = []
        for root, dirs, files in os.walk('./dataset_lokal'):
            if 'runs' in root: continue
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        found_photos = []
        
        for img_path in image_files: 
            results = model(img_path)
            img = cv2.imread(img_path)
            
            for r in results:
                boxes = r.boxes.xyxy
                match_found = False 
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    gray_crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    
                    ocr_result = reader.readtext(gray_crop, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    
                    for detection in ocr_result:
                        text = str(detection[1]).replace(" ", "").strip().upper()
                        conf = detection[2]
                        
                        if search_query in text and conf > 0.20:
                            found_photos.append(img_path)
                            match_found = True
                            break 
                            
                    if match_found:
                        break 
        
        # TAMBAHAN: Hentikan stopwatch dan simpan hasilnya ke memori state
        end_time = time.time()
        st.session_state.execution_time = end_time - start_time
        
        # Simpan hasil pencarian ke dalam memori session_state
        st.session_state.found_photos = found_photos
        
        # TAMBAHAN: Efek Selebrasi Balon Terbang saat data sukses ditemukan
        if found_photos:
            st.balloons()
                        
    else:
        st.warning("Silakan masukkan nomor bib terlebih dahulu.")

# 2. LOGIKA MENAMPILKAN HASIL (Ditaruh di luar tombol cari, membaca dari memori)
if st.session_state.searched:
    if st.session_state.found_photos:
        st.success(f"Yeay! Ditemukan {len(st.session_state.found_photos)} foto untuk pelari nomor {st.session_state.last_query} 🎉")
        
        # TAMBAHAN: Menampilkan teks indikator kecepatan pemindaian AI
        st.caption(f"⚡ **Kecepatan AI:** Seluruh berkas dataset lokal selesai dipindai dalam waktu **{st.session_state.execution_time:.2f} detik**.")
        
        cols = st.columns(3)
        for i, photo_path in enumerate(st.session_state.found_photos):
            with cols[i % 3]:
                # Menampilkan foto dari memori
                st.image(Image.open(photo_path), use_container_width=True) 
                
                # Tombol Download (Akan tetap stay karena berada di luar if st.button cari)
                with open(photo_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Foto",
                        data=file,
                        file_name=os.path.basename(photo_path),
                        mime="image/jpeg",
                        key=f"download_{i}"  
                    )
    else:
        st.error(f"Maaf, foto dengan nomor bib '{st.session_state.last_query}' tidak ditemukan setelah memindai seluruh dataset.")
        st.caption(f"⚡ **Kecepatan AI:** Proses pemindaian folder selesai dalam **{st.session_state.execution_time:.2f} detik**.")
