
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Reciclaje Computación en la Nube",
    layout="centered"
)

st.title("Modelo predictivo Reciclaje clase de IA-UTH - 2026 - Felipe Alvarado")
st.write("Suba una imagen para clasificar con el modelo MobileNetV2 preentrenado")

IMG_SIZE = (224, 224)

MODEL_DIR = Path("modelo_reciclaje_mobilenet")
CLASS_PATH = MODEL_DIR / "class_names.json"

MODEL_PATHS = [
    MODEL_DIR / "waste_mobilenet.keras",
    MODEL_DIR / "waste_mobilenet.h5"
]

LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Basura"
}


def limpiar_config(config):

    if isinstance(config, dict):

        config.pop("quantization_config", None)

        for valor in config.values():
            limpiar_config(valor)

    elif isinstance(config, list):

        for elemento in config:
            limpiar_config(elemento)


def cargar_keras_compatible(path):

    archivo_temporal = None

    try:
        with zipfile.ZipFile(path, "r") as original:

            config = json.loads(
                original.read("config.json").decode("utf-8")
            )

            limpiar_config(config)

            with tempfile.NamedTemporaryFile(
                suffix=".keras",
                delete=False
            ) as temporal:

                archivo_temporal = temporal.name

            with zipfile.ZipFile(
                archivo_temporal,
                "w",
                compression=zipfile.ZIP_DEFLATED
            ) as nuevo:

                for elemento in original.infolist():

                    if elemento.filename == "config.json":

                        contenido = json.dumps(config).encode("utf-8")

                    else:

                        contenido = original.read(elemento.filename)

                    nuevo.writestr(elemento, contenido)

        modelo = tf.keras.models.load_model(
            archivo_temporal,
            compile=False
        )

        return modelo

    finally:

        if archivo_temporal and os.path.exists(archivo_temporal):
            os.remove(archivo_temporal)


@st.cache_resource
def cargar_modelo():

    errores = []

    for path in MODEL_PATHS:

        if path.exists():

            try:

                if path.suffix == ".keras":
                    modelo = cargar_keras_compatible(path)

                else:
                    modelo = tf.keras.models.load_model(
                        path,
                        compile=False
                    )

                st.success(f"Modelo cargado correctamente: {path.name}")

                return modelo

            except Exception as e:

                errores.append(f"{path.name}: {str(e)}")

    st.error("No se pudo cargar ninguno de los modelos:")

    for error in errores:
        st.error(error)

    st.stop()


@st.cache_data
def cargar_clases():

    if CLASS_PATH.exists():

        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash"
    ]


def preparar_imagen(img):

    img = img.convert("RGB").resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32)

    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

    return np.expand_dims(arr, axis=0)


def predecir(img):

    preds = modelo.predict(
        preparar_imagen(img),
        verbose=0
    )[0]

    top3 = np.argsort(preds)[-3:][::-1]

    return [
        (
            LABELS_ES.get(clases[i], clases[i]),
            float(preds[i]) * 100
        )
        for i in top3
    ]


modelo = cargar_modelo()
clases = cargar_clases()

if len(clases) != modelo.output_shape[-1]:
    st.error("La cantidad de clases no coincide con las salidas del modelo.")
    st.stop()

archivo = st.file_uploader(
    "Seleccione una imagen",
    type=["jpg", "jpeg", "png"]
)

if archivo:

    imagen = Image.open(archivo)

    st.image(
        imagen,
        caption="Imagen analizada",
        use_container_width=True
    )

    resultados = predecir(imagen)

    st.subheader("Resultado")

    st.success(
        f"Predicción principal: {resultados[0][0]} "
        f"({resultados[0][1]:.2f}%)"
    )

    st.write("Top 3 probabilidades:")

    for clase, prob in resultados:
        st.write(f"{clase}: {prob:.2f}%")

else:

    st.info("Cargue una imagen para iniciar la clasificación.")
