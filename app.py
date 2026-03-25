import streamlit as st
import numpy as np
from PIL import Image
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
    # 使用 PIL 读取图片
    image = Image.open(uploaded_file)
    # 转换为 RGB（确保格式）
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # 转为 numpy 数组（ultralytics 可以接受 PIL 图像，但直接传 numpy 更通用）
    img_array = np.array(image)
    # 检测
    results = model(img_array, conf=0.25)
    annotated = results[0].plot()  # 返回 BGR 格式的 numpy 数组
    # 显示结果（streamlit 需要 RGB）
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