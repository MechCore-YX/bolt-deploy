import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="螺栓缺陷AI检测", layout="centered")
st.title("🔩 螺栓缺陷AI视觉检测系统")
st.write("上传一张螺栓图片，系统将自动识别缺陷类型（正常、锈蚀、划痕、变形、螺纹损坏）")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

uploaded_file = st.file_uploader("点击上传图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(img, conf=0.25)
    annotated = results[0].plot()
    st.image(annotated, channels="BGR", caption="检测结果")
    if len(results[0].boxes) > 0:
        st.success(f"检测到 {len(results[0].boxes)} 个目标")
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf)
            class_name = model.names[cls]
            st.write(f"- {class_name}: {conf:.2f}")
    else:
        st.warning("未检测到螺栓")
else:
    st.info("请上传一张图片开始检测。")