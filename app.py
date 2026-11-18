import platform
import pathlib
import sys
import types
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from fastai.vision.all import *
import plotly.express as px

if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath
st.title('Ichimliklarni klasslovchi dastur')

# 2. Modelni xotiraga yuklash
@st.cache_resource
def get_model():
    return load_learner('transport_model.pkl', cpu=True)

try:
    model = get_model()
except Exception as err:
    st.error(f"Modelni yuklashda xatolik: {err}")
    st.stop()

# 3. FastAI predict xatosini aylanib o'tuvchi xavfsiz inferensiya funksiyasi
def predict_safely(img, learner):
    try:
        # Avval standart fastai usulini sinab ko'ramiz
        return learner.predict(img)
    except SystemError:
        # Fasttransform/TensorImage ichki SystemError berganda to'g'ridan-to'g'ri PyTorch orqali hisoblash
        nn_model = learner.model.eval()
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Rasmni RGB ga o'tkazib, tensor shakliga keltiramiz
        pil_img = Image.fromarray(np.array(img)).convert('RGB')
        tensor_img = transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = nn_model(tensor_img)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_id = int(torch.argmax(probs))
            pred = learner.dls.vocab[pred_id]
            
        return pred, pred_id, probs.numpy()

file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'jpg', 'gif', 'svg'])

if file:
    st.image(file)
    img = PILImage.create(file)
    
    # Bashorat
    pred, pred_id, probs = predict_safely(img, model)
    
    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')
    
    # Ehtimolliklar grafigi
    fig = px.bar(
        x=probs * 100, 
        y=list(model.dls.vocab), 
        orientation='h', 
        labels={'x': 'Ehtimollik (%)', 'y': 'Sinflar'}
    )
    st.plotly_chart(fig)