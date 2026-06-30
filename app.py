import streamlit as st
import cv2
import easyocr
import os
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Running Bib Finder", layout="wide")
st.title("🏃‍♂️ Running Bib Finder")
st.write("Sistem pencarian dokumentasi foto lari otomatis berbasis YOLOv11 & OCR.")
st.write("By.Arief Daniel A.Silalahi")

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
        
        image_files = []
        for root, dirs, files in os.walk('/content'):
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
                        
                        # PERBAIKAN: Menggunakan "in" agar lebih fleksibel membaca teks yang menempel
                        if search_query in text and conf > 0.20:
                            found_photos.append(img_path)
                            match_found = True
                            break 
                            
                    if match_found:
                        break 
                            
        if found_photos:
            st.success(f"Yeay! Ditemukan {len(found_photos)} foto untuk pelari nomor {search_query} 🎉")
            cols = st.columns(3)
            for i, photo_path in enumerate(found_photos):
                with cols[i % 3]:
                    st.image(Image.open(photo_path), use_container_width=True) 
        else:
            st.error("Maaf, foto dengan nomor bib tersebut tidak ditemukan setelah memindai seluruh dataset.")
    else:
        st.warning("Silakan masukkan nomor bib terlebih dahulu.")
