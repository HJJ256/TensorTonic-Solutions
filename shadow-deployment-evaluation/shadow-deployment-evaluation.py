import math
import pandas as pd
def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """
    production_log_df = pd.DataFrame(production_log)
    shadow_log_df = pd.DataFrame(shadow_log)
    production_log_df["match"] = production_log_df["actual"] == production_log_df["prediction"]
    shadow_log_df["match"] = shadow_log_df["actual"] == shadow_log_df["prediction"]
    p95_latency_shadow = shadow_log_df["latency_ms"].sort_values()[math.ceil(0.95*len(shadow_log_df))-1]
    prod_acc = production_log_df["match"].sum()/len(production_log_df)
    shadow_acc = shadow_log_df["match"].sum()/len(shadow_log_df)
    gain = shadow_acc - prod_acc
    merge_df = pd.merge(production_log_df,shadow_log_df,on="input_id", suffixes=("_prod","_shadow"))
    merge_df["agreement"] = merge_df["prediction_prod"] == merge_df["prediction_shadow"]
    agreement_rate = merge_df["agreement"].sum()/len(merge_df)
    metrics = {
        "shadow_accuracy": shadow_acc,
        "production_accuracy": prod_acc,
        "accuracy_gain": gain,
        "shadow_latency_p95": int(p95_latency_shadow),
        "agreement_rate": agreement_rate,
    }
    promote = gain>=criteria["min_accuracy_gain"] and p95_latency_shadow<=criteria["max_latency_p95"] and agreement_rate>=criteria["min_agreement_rate"]
    return {"promote": bool(promote), "metrics": metrics}