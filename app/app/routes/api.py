from flask import Blueprint, request, jsonify
from app.db.queries import load_all_evaluations, get_model_evaluation
#from app.ml.client import get_prediction
#from app.db.queries import run_analysis_query

api = Blueprint('api', __name__, url_prefix="/api")

@api.route('/models', methods=['GET'])
def list_models():
    data = load_all_evaluations()
    return jsonify({
        "models": [
            {
                "model_id": m["model_id"],
                "model_name": m["model_name"],
                "version": m["version"],
                "metrics": m["metrics"]
            }
            for m in data["models"]
            ]})

@api.route("/models/<model_id>/evaluation", methods=["GET"])
def model_evaluation(model_id):
    return jsonify(get_model_evaluation(model_id))

@api.route("/evaluation", methods=["GET"])
def default_evaluation():
    model_id = request.args.get("model_id")
    return jsonify(get_model_evaluation(model_id))

"""
@api.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comment_text = data.get("comment_text", "")
    if not comment_text:
        return jsonify({"error": "No comment_text provided"}), 400
    result = get_prediction(comment_text)
    return jsonify(result)

@api.route('/analysis', methods=['GET'])
def analysis():
    query_type = request.args.get('type', 'summary')
    result = run_analysis_query(query_type)
    return jsonify(result)
"""