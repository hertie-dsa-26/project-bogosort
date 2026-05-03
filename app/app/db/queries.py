import json
from pathlib import Path
#from sqlalchemy import create_engine, text
#from flask import current_app
#engine = None

DATA_PATH = Path("data/model_evaluations.json")

def load_all_evaluations():
    if not DATA_PATH.exists():
        return {"models": []}
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_model_evaluation(model_id=None):
    data = load_all_evaluations()
    if not model_id:
        return data["models"][0] if data["models"] else {}
    for model in data["models"]:
        if model["model_id"] == model_id:
            return model
    return {}

"""
def init_db(app):
    global engine
    engine = create_engine(app.config['SQL_URI'])

def run_analysis_query(query_type):
    if not engine:
        return {"error": "DB not initialized"}
    
    with engine.connect() as conn:
        if query_type == 'summary':
            sql = text("
                SELECT 
                    AVG(toxicity) as avg_toxicity,
                    AVG(severe_toxicity) as avg_severe_toxicity,
                    AVG(obscene) as avg_obscene,
                    AVG(threat) as avg_threat,
                    AVG(insult) as avg_insult,
                    AVG(identity_hate) as avg_identity_hate
                FROM dataset
            ")
            result = conn.execute(sql).fetchone()
            return dict(result._mapping)
        elif query_type == 'detailed':
            sql = text("SELECT * FROM dataset LIMIT 100")
            result = conn.execute(sql).fetchall()
            return [dict(row._mapping) for row in result]
        else:
            return {"error": "Unknown query type"}
"""