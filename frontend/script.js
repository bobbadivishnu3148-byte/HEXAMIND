// ============================================
// QUANTUM HEALTH AI - FRONTEND
// ============================================

const API_URL = "http://127.0.0.1:8000/predict";


// ============================================
// PAGE NAVIGATION
// ============================================

function showPage(pageId) {

    const pages = document.querySelectorAll(".page");

    pages.forEach(page => {
        page.classList.add("hidden");
    });

    const selectedPage = document.getElementById(pageId);

    if (selectedPage) {
        selectedPage.classList.remove("hidden");
    }


    // Update active navigation button

    const buttons = document.querySelectorAll(".nav-btn");

    buttons.forEach(button => {
        button.classList.remove("active");
    });

}


// ============================================
// MODEL FEATURE ORDER
// MUST MATCH TRAINING DATASET ORDER
// ============================================

const featureIds = [

    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",

    "radius_error",
    "texture_error",
    "perimeter_error",
    "area_error",
    "smoothness_error",
    "compactness_error",
    "concavity_error",
    "concave_points_error",
    "symmetry_error",
    "fractal_dimension_error",

    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension"

];


// Human-readable names

const featureNames = {

    mean_radius: "Mean Radius",
    mean_texture: "Mean Texture",
    mean_perimeter: "Mean Perimeter",
    mean_area: "Mean Area",
    mean_smoothness: "Mean Smoothness",
    mean_compactness: "Mean Compactness",
    mean_concavity: "Mean Concavity",
    mean_concave_points: "Mean Concave Points",
    mean_symmetry: "Mean Symmetry",
    mean_fractal_dimension: "Mean Fractal Dimension",

    radius_error: "Radius Error",
    texture_error: "Texture Error",
    perimeter_error: "Perimeter Error",
    area_error: "Area Error",
    smoothness_error: "Smoothness Error",
    compactness_error: "Compactness Error",
    concavity_error: "Concavity Error",
    concave_points_error: "Concave Points Error",
    symmetry_error: "Symmetry Error",
    fractal_dimension_error: "Fractal Dimension Error",

    worst_radius: "Worst Radius",
    worst_texture: "Worst Texture",
    worst_perimeter: "Worst Perimeter",
    worst_area: "Worst Area",
    worst_smoothness: "Worst Smoothness",
    worst_compactness: "Worst Compactness",
    worst_concavity: "Worst Concavity",
    worst_concave_points: "Worst Concave Points",
    worst_symmetry: "Worst Symmetry",
    worst_fractal_dimension: "Worst Fractal Dimension"

};


// ============================================
// COLLECT 30 MODEL FEATURES
// ============================================

function collectFeatures() {

    const features = [];

    for (const id of featureIds) {

        const input = document.getElementById(id);

        if (!input) {
            throw new Error(
                `Input field not found: ${id}`
            );
        }

        const value = Number(input.value);

        if (!Number.isFinite(value)) {

            alert(
                `Please enter a valid value for ${featureNames[id]}.`
            );

            input.focus();

            return null;
        }

        features.push(value);
    }

    return features;
}


// ============================================
// RUN REAL HYBRID QNN PREDICTION
// ============================================

async function runPrediction() {

    const name =
        document.getElementById("patientName").value.trim();

    const age =
        Number(document.getElementById("age").value);


    // Basic patient information validation

    if (!name) {

        alert("Please enter the patient name.");

        return;
    }


    if (!Number.isFinite(age) || age <= 0) {

        alert("Please enter a valid patient age.");

        return;
    }


    // Collect the 30 WDBC features

    const features = collectFeatures();

    if (!features) {
        return;
    }


    // Show prediction page

    showPage("prediction");


    const processingText =
        document.getElementById("processingText");

    const riskValue =
        document.getElementById("riskValue");

    const riskLevel =
        document.getElementById("riskLevel");

    const riskBar =
        document.getElementById("riskBar");

    const recommendationText =
        document.getElementById("recommendationText");

    const featureContributions =
        document.getElementById("featureContributions");


    // Loading state

    processingText.innerText =
        "Running hybrid classical-quantum inference...";

    riskValue.innerText = "...";

    riskLevel.innerText =
        "Analyzing...";

    riskBar.style.width = "0%";

    recommendationText.innerText =
        "The hybrid QNN is processing the diagnostic features.";

    featureContributions.innerHTML =
        "<p>Calculating model explanation...</p>";


    try {

        // ========================================
        // SEND DATA TO FASTAPI
        // ========================================

        const response = await fetch(API_URL, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                features: features
            })

        });


        // Handle HTTP errors

        if (!response.ok) {

            let errorMessage =
                `API request failed (${response.status})`;

            try {

                const errorData =
                    await response.json();

                if (errorData.detail) {
                    errorMessage =
                        errorData.detail;
                }

            } catch (e) {
                // Ignore JSON parsing error
            }

            throw new Error(errorMessage);
        }


        // Read prediction

        const result =
            await response.json();


        console.log(
            "Hybrid QNN API response:",
            result
        );


        // ========================================
        // READ API RESULT
        // ========================================

        const probability =
            getProbability(result);


        if (probability === null) {

            throw new Error(
                "The API response did not contain a valid probability."
            );
        }


        const percentage =
            Math.round(probability * 100);


        // ========================================
        // DETERMINE DISPLAY LEVEL
        // ========================================

        let level;

        if (percentage >= 70) {

            level = "High Risk";

        } else if (percentage >= 35) {

            level = "Moderate Risk";

        } else {

            level = "Low Risk";

        }


        // ========================================
        // UPDATE RESULT UI
        // ========================================

        riskValue.innerText =
            percentage + "%";

        riskLevel.innerText =
            level;

        riskBar.style.width =
            percentage + "%";


        processingText.innerText =
            "Hybrid quantum-classical inference complete.";


        // ========================================
        // RECOMMENDATION
        // ========================================

        if (percentage >= 50) {

            recommendationText.innerText =
                "The model predicts a higher probability of a malignant classification. This is a screening result, not a diagnosis. Please seek evaluation by a qualified healthcare professional.";

        } else {

            recommendationText.innerText =
                "The model predicts a higher probability of a benign classification. This is a screening result, not a diagnosis. Clinical evaluation should still be based on appropriate medical assessment.";

        }


        // ========================================
        // FEATURE CONTRIBUTIONS
        // ========================================

        displayFeatureContributions(
            result
        );


        // ========================================
        // UPDATE DASHBOARD
        // ========================================

        updateDashboard(
            percentage,
            level
        );


        // ========================================
        // ADD TO HISTORY
        // ========================================

        addHistory(
            name,
            age,
            level
        );


    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        processingText.innerText =
            "Prediction failed.";


        riskValue.innerText =
            "--";


        riskLevel.innerText =
            "Connection Error";


        riskBar.style.width =
            "0%";


        recommendationText.innerText =
            "Could not connect to the Python API. Make sure the FastAPI server is running at http://127.0.0.1:8000.";


        featureContributions.innerHTML =
            `<p>${escapeHTML(error.message)}</p>`;


        alert(
            "Could not connect to the Hybrid QML backend.\n\n" +
            "Make sure this is running:\n" +
            "python -m uvicorn api:app --reload"
        );

    }

}


// ============================================
// GET PROBABILITY FROM API
// ============================================

function getProbability(result) {

    /*
        Your API currently returns probability
        information. This function accepts the
        common field names so the frontend is
        slightly more robust.
    */

    const possibleFields = [

        "malignant_probability",
        "malignantProbability",
        "probability",
        "risk_probability",
        "riskProbability"

    ];


    for (const field of possibleFields) {

        if (
            result[field] !== undefined &&
            result[field] !== null
        ) {

            const value =
                Number(result[field]);

            if (Number.isFinite(value)) {

                // Convert percentage to 0-1
                // if API happens to return 72 instead of 0.72

                if (value > 1) {
                    return value / 100;
                }

                return value;
            }
        }
    }


    return null;
}


// ============================================
// FEATURE CONTRIBUTIONS
// ============================================

function displayFeatureContributions(result) {

    const container =
        document.getElementById(
            "featureContributions"
        );


    if (!container) {
        return;
    }


    const contributions =
        result.feature_contributions ||
        result.featureContributions ||
        result.top_features ||
        result.topFeatures;


    if (!Array.isArray(contributions) ||
        contributions.length === 0) {

        container.innerHTML =
            "<p>No feature explanation was returned by the API.</p>";

        return;
    }


    container.innerHTML = "";


    contributions.slice(0, 5).forEach(
        (item, index) => {

            let name =
                item.feature ||
                item.name ||
                item.feature_name ||
                `Feature ${index + 1}`;


            // Convert technical feature name

            if (featureNames[name]) {
                name = featureNames[name];
            }


            const change =
                Number(
                    item.change ??
                    item.contribution ??
                    item.impact ??
                    0
                );


            const div =
                document.createElement("div");

            div.style.marginBottom =
                "10px";


            const direction =
                change >= 0
                    ? "increased"
                    : "decreased";


            div.innerHTML = `

                <strong>
                    ${index + 1}. ${escapeHTML(name)}
                </strong>

                <br>

                <span>
                    Model probability ${direction}
                    by ${Math.abs(change).toFixed(4)}
                </span>

            `;


            container.appendChild(div);

        }
    );

}


// ============================================
// DASHBOARD UPDATE
// ============================================

function updateDashboard(
    percentage,
    level
) {

    const dashboardRisk =
        document.getElementById(
            "dashboardRisk"
        );

    const dashboardRiskLevel =
        document.getElementById(
            "dashboardRiskLevel"
        );


    if (dashboardRisk) {

        dashboardRisk.innerText =
            percentage + "%";

    }


    if (dashboardRiskLevel) {

        dashboardRiskLevel.innerText =
            level;

    }

}


// ============================================
// HISTORY
// ============================================

function addHistory(
    name,
    age,
    level
) {

    const table =
        document.getElementById(
            "historyTable"
        );


    if (!table) {
        return;
    }


    const row =
        document.createElement("tr");


    const riskClass =
        level === "High Risk"
            ? "high"
            : level === "Moderate Risk"
                ? "moderate"
                : "low";


    row.innerHTML = `

        <td>${escapeHTML(name)}</td>

        <td>${age}</td>

        <td>Breast Cancer</td>

        <td>

            <span class="risk ${riskClass}">
                ${escapeHTML(level)}
            </span>

        </td>

        <td>
            ${new Date().toLocaleDateString()}
        </td>

    `;


    table.prepend(row);


    updateCounters(level);

}


// ============================================
// DASHBOARD COUNTERS
// ============================================

function updateCounters(level) {

    const patientCount =
        document.getElementById(
            "patientCount"
        );

    const highRiskCount =
        document.getElementById(
            "highRiskCount"
        );


    if (patientCount) {

        const current =
            Number(
                patientCount.innerText
            ) || 0;

        patientCount.innerText =
            current + 1;

    }


    if (
        highRiskCount &&
        level === "High Risk"
    ) {

        const current =
            Number(
                highRiskCount.innerText
            ) || 0;

        highRiskCount.innerText =
            current + 1;

    }

}


// ============================================
// HTML ESCAPING
// ============================================

function escapeHTML(value) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");

}


// ============================================
// INITIAL PAGE
// ============================================

showPage("dashboard");