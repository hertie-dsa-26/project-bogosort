import argparse
import importlib

MODEL_REGISTRY = {
    "baseline": "analysis.models.baseline.baseline_model",
    "lasso":    "analysis.models.lasso_log_reg.lasso",
}

def model_run(model_name, data_path, mode, save_predictions, save_model):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found! Please choose a valid model.")
    
    model_path = MODEL_REGISTRY[model_name]
    model = importlib.import_module(model_path)

    model.run(data_path=data_path,
        mode=mode,
        save_predictions=save_predictions,
        save_model=save_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model", default="baseline")
    parser.add_argument("--model", type=str, required=True, default="baseline")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--data_path", default="data/processed/")
    parser.add_argument("--save_predictions", type=bool, default=True)
    parser.add_argument("--save_model", type=bool, default=True)

    args = parser.parse_args()

    model_run(args.model,
        args.data_path,
        args.mode,
        args.save_predictions,
        args.save_model)

#model_run("baseline", "data/processed/")