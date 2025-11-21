import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# モデルとクラスの設定
model_path = "model/cat_model.pth"
classes = [
    "American Shorthair",
    "Maine Coon",
    "Norwegian Forest Cat",
    "Persian",
    "Russian Blue",
    "Scottish Fold",
    "Siamese"
]

# モデル読み込み
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# 前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("🐱 Cat Breed Classifier")
st.write("画像をアップロードして猫の品種を分類します。")

uploaded_file = st.file_uploader("画像ファイルを選択", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    try:
        st.image(image, caption="アップロード画像", use_container_width=True)
    except TypeError:
        # 古い Streamlit 版のフォールバック
        st.image(image, caption="アップロード画像", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        result = classes[predicted.item()]

    st.success(f"判定結果：{result}")
