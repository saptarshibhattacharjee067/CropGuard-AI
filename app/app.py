import sys
import json
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "src")
)


from predict import (
    predict_image,
    get_prediction_status
)

from risk_engine import (
    calculate_risk
)


st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌱",
    layout="centered"
)


st.title("🌱 CropGuard AI")

st.subheader(
    "AI-Powered Crop Disease Detection"
)

st.write(
    "Upload a tomato leaf image and CropGuard AI "
    "will analyze it using a fine-tuned ResNet18 model."
)


uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded tomato leaf",
        width="stretch"
    )


    temp_path = (
        PROJECT_ROOT
        / "app"
        / "uploaded_image.jpg"
    )


    with open(
        temp_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )


    with st.spinner(
        "Analyzing leaf..."
    ):

        predictions = predict_image(
            temp_path,
            top_k=3
        )


    top_prediction = (
        predictions[0]
    )


    disease = (
        top_prediction["class"]
    )


    confidence = (
        top_prediction["confidence"]
    )


    status, status_message = (
        get_prediction_status(
            confidence
        )
    )


    st.divider()

    st.header(
        "🔍 AI Diagnosis"
    )


    st.success(
        disease.replace(
            "_",
            " "
        )
    )


    st.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )


    if status == "sufficiently_confident":

        st.success(
            status_message
        )

    else:

        st.warning(
            status_message
        )


    st.caption(
        "Note: Softmax confidence is a model score "
        "and should not be interpreted as a calibrated "
        "probability of disease."
    )


    st.header(
        "📊 Top 3 Predictions"
    )


    for position, prediction in enumerate(
        predictions,
        start=1
    ):

        prediction_name = (
            prediction["class"]
            .replace(
                "_",
                " "
            )
        )


        prediction_confidence = (
            prediction["confidence"]
        )


        st.write(
            f"**{position}. "
            f"{prediction_name}**"
        )


        st.progress(
            min(
                prediction_confidence / 100,
                1.0
            )
        )


        st.caption(
            f"{prediction_confidence:.2f}% confidence"
        )


    disease_info_path = (
        PROJECT_ROOT
        / "data"
        / "disease_info.json"
    )


    if disease_info_path.exists():

        with open(
            disease_info_path,
            "r",
            encoding="utf-8"
        ) as file:

            disease_info = json.load(
                file
            )


        info = disease_info.get(
            disease
        )


        if info:

            st.divider()

            st.header(
                "📋 Disease Information"
            )


            st.write(
                f"**Disease Type:** "
                f"{info['type']}"
            )


            st.subheader(
                "📖 Description"
            )


            st.write(
                info["description"]
            )


            st.subheader(
                "🔎 Common Symptoms"
            )


            for symptom in info[
                "symptoms"
            ]:

                st.write(
                    f"- {symptom}"
                )


            st.subheader(
                "🛠️ Management"
            )


            for item in info[
                "management"
            ]:

                st.write(
                    f"- {item}"
                )


            st.subheader(
                "🛡️ Prevention"
            )


            for item in info[
                "prevention"
            ]:

                st.write(
                    f"- {item}"
                )


    st.divider()


    st.header(
        "🌦️ Environmental Risk Assessment"
    )


    st.write(
        "Enter the current environmental conditions "
        "to estimate disease-favorable conditions."
    )


    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=27.0,
        step=0.5
    )


    humidity = st.slider(
        "Relative Humidity (%)",
        min_value=0,
        max_value=100,
        value=70
    )


    recent_rainfall = st.checkbox(
        "Recent rainfall"
    )


    if st.button(
        "🌦️ Assess Environmental Risk"
    ):

        risk_result = calculate_risk(
            disease=disease,
            temperature=temperature,
            humidity=humidity,
            recent_rainfall=recent_rainfall
        )


        risk_level = (
            risk_result["risk_level"]
        )


        risk_score = (
            risk_result["risk_score"]
        )


        st.subheader(
            "Environmental Risk"
        )


        if risk_level == "HIGH":

            st.error(
                f"🔴 HIGH RISK"
            )

        elif risk_level == "MODERATE":

            st.warning(
                f"🟡 MODERATE RISK"
            )

        else:

            st.success(
                f"🟢 LOW RISK"
            )


        st.metric(
            "Risk Score",
            risk_score
        )


        if risk_result[
            "factors"
        ]:

            st.subheader(
                "⚠️ Contributing Factors"
            )


            for factor in risk_result[
                "factors"
            ]:

                st.write(
                    f"- {factor}"
                )


        st.subheader(
            "💡 Recommendations"
        )


        for recommendation in risk_result[
            "recommendations"
        ]:

            st.write(
                f"- {recommendation}"
            )


        st.info(
            "Environmental risk is an analytical "
            "screening result based on the supplied "
            "conditions. It is not a definitive disease "
            "forecast. Verify management decisions with "
            "appropriate local agricultural guidance."
        )


    st.divider()


    st.header(
        "🧠 Model Attention — Grad-CAM"
    )


    st.write(
        "The Grad-CAM visualization shows image regions "
        "that contributed strongly to the model's prediction."
    )


    gradcam_path = (
        PROJECT_ROOT
        / "results"
        / "gradcam_resnet18.png"
    )


    if gradcam_path.exists():

        st.image(
            gradcam_path,
            caption="Model attention visualization",
            width="stretch"
        )

    else:

        st.info(
            "Grad-CAM visualization is not available yet."
        )


    st.info(
        "CropGuard AI provides an AI-based screening "
        "result and should not be treated as a definitive "
        "agricultural diagnosis. For real-world treatment "
        "decisions, verify the result with appropriate "
        "agricultural guidance."
    )