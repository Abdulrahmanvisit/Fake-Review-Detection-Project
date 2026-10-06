from flask import Flask, jsonify, request
from flask_cors import CORS

from backend.app.api.predict import predict_review


app = Flask(__name__)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ]
        }
    },
)


@app.get("/api/health")
def health_check():
    return jsonify(
        {
            "status": "ok",
            "message": "Fake review detection API is running.",
        }
    )


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify(
            {"error": "Request body must be a JSON object."}
        ), 400

    review_text = data.get("review_text")
    rating = data.get("rating", 0)
    verified_purchase = data.get("verified_purchase", "N")

    if not isinstance(review_text, str):
        return jsonify(
            {"error": "review_text must be a string."}
        ), 400

    if not review_text.strip():
        return jsonify(
            {"error": "review_text cannot be empty."}
        ), 400

    try:
        rating = float(rating)
    except (TypeError, ValueError):
        return jsonify(
            {"error": "rating must be a number between 0 and 5."}
        ), 400

    try:
        result = predict_review(
            review_text=review_text,
            rating=rating,
            verified_purchase=verified_purchase,
        )

        return jsonify(result), 200

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        return jsonify(
            {
                "error": (
                    "An unexpected error occurred while "
                    "analysing the review."
                )
            }
        ), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )