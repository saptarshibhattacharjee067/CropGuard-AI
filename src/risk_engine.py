def calculate_risk(
    disease,
    temperature,
    humidity,
    recent_rainfall
):
    """
    Calculate environmental disease risk.

    Parameters:
        disease: predicted disease name
        temperature: temperature in Celsius
        humidity: relative humidity percentage
        recent_rainfall: True if there was recent rainfall

    Returns:
        Dictionary containing risk level,
        score, factors, and recommendations.
    """

    risk_score = 0

    factors = []

    recommendations = []


    fungal_diseases = [
        "Early_blight",
        "Late_blight",
        "Leaf_Mold",
        "Septoria_leaf_spot"
    ]


    bacterial_diseases = [
        "Bacterial_spot"
    ]


    mite_diseases = [
        "Spider_mites Two-spotted_spider_mite"
    ]


    if disease in fungal_diseases:

        if humidity >= 80:

            risk_score += 2

            factors.append(
                "High humidity may favor fungal disease development."
            )

        elif humidity >= 65:

            risk_score += 1

            factors.append(
                "Moderately high humidity may support fungal disease development."
            )


        if recent_rainfall:

            risk_score += 2

            factors.append(
                "Recent rainfall can increase leaf wetness and disease-favorable conditions."
            )


        if 20 <= temperature <= 30:

            risk_score += 2

            factors.append(
                "Temperature is within a range that can support fungal disease development."
            )


    elif disease in bacterial_diseases:

        if humidity >= 80:

            risk_score += 2

            factors.append(
                "High humidity can favor bacterial disease development."
            )

        elif humidity >= 65:

            risk_score += 1

            factors.append(
                "Moderately high humidity may support bacterial disease development."
            )


        if recent_rainfall:

            risk_score += 2

            factors.append(
                "Rainfall can spread bacterial pathogens through water splash."
            )


        if 20 <= temperature <= 32:

            risk_score += 2

            factors.append(
                "Temperature is within a range that may support bacterial disease development."
            )


    elif disease in mite_diseases:

        if humidity <= 50:

            risk_score += 2

            factors.append(
                "Low humidity can favor spider mite population growth."
            )

        elif humidity <= 65:

            risk_score += 1

            factors.append(
                "Moderately dry conditions may support spider mite activity."
            )


        if temperature >= 27:

            risk_score += 2

            factors.append(
                "Warm temperatures can favor spider mite development."
            )


    elif disease == "Tomato_Yellow_Leaf_Curl_Virus":

        if temperature >= 25:

            risk_score += 1

            factors.append(
                "Warm conditions can favor vector activity associated with virus transmission."
            )


        if humidity <= 65:

            risk_score += 1

            factors.append(
                "Relatively dry conditions can support vector activity."
            )


    else:

        if humidity >= 80:

            risk_score += 1

            factors.append(
                "High humidity may increase general plant disease pressure."
            )


        if recent_rainfall:

            risk_score += 1

            factors.append(
                "Recent rainfall can increase leaf wetness."
            )


    if disease in fungal_diseases:

        recommendations.extend([
            "Improve air circulation around plants.",
            "Avoid prolonged leaf wetness.",
            "Remove heavily infected leaves.",
            "Monitor nearby plants for new symptoms."
        ])


    elif disease in bacterial_diseases:

        recommendations.extend([
            "Avoid working with plants while foliage is wet.",
            "Remove heavily infected plant material.",
            "Improve plant spacing and airflow.",
            "Monitor nearby plants for new symptoms."
        ])


    elif disease in mite_diseases:

        recommendations.extend([
            "Inspect leaves for mites and fine webbing.",
            "Monitor plants during warm and dry conditions.",
            "Reduce excessive plant stress where practical.",
            "Use locally appropriate pest-management guidance."
        ])


    elif disease == "Tomato_Yellow_Leaf_Curl_Virus":

        recommendations.extend([
            "Inspect plants for whiteflies and other potential vectors.",
            "Remove severely affected plants where appropriate.",
            "Monitor nearby plants for symptoms.",
            "Follow locally appropriate vector-management guidance."
        ])


    else:

        recommendations.extend([
            "Continue monitoring the plant.",
            "Maintain good airflow and appropriate irrigation.",
            "Inspect nearby plants for similar symptoms."
        ])


    if risk_score >= 5:

        risk_level = "HIGH"

    elif risk_score >= 3:

        risk_level = "MODERATE"

    else:

        risk_level = "LOW"


    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "factors": factors,
        "recommendations": recommendations
    }


if __name__ == "__main__":

    print()
    print("CropGuard AI Environmental Risk Engine")
    print("======================================")
    print()


    disease = input(
        "Disease: "
    )


    temperature = float(
        input(
            "Temperature (°C): "
        )
    )


    humidity = float(
        input(
            "Humidity (%): "
        )
    )


    rainfall_input = input(
        "Recent rainfall? (yes/no): "
    ).lower()


    recent_rainfall = (
        rainfall_input == "yes"
    )


    result = calculate_risk(
        disease=disease,
        temperature=temperature,
        humidity=humidity,
        recent_rainfall=recent_rainfall
    )


    print()

    print(
        "ENVIRONMENTAL RISK"
    )

    print(
        "------------------"
    )


    print(
        f"Risk level: "
        f"{result['risk_level']}"
    )


    print(
        f"Risk score: "
        f"{result['risk_score']}"
    )


    print()

    print(
        "Contributing factors:"
    )


    for factor in result[
        "factors"
    ]:

        print(
            f"- {factor}"
        )


    print()

    print(
        "Recommendations:"
    )


    for recommendation in result[
        "recommendations"
    ]:

        print(
            f"- {recommendation}"
        )