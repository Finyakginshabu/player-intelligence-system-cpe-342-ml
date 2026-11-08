"""Task 2 latent-axis composite features (round 2, 2026-11-08; +0.0029 macro F1, all 5 folds up).

Segments are a 2x2 of engagement (Grinder/Whale high) x spending (Social/Whale high). Each axis is spread over
many noisy, ~8%-missing features; averaging their z-scores denoises it. Fit the z-score stats on the training
fold only (pass it as `fit_src`).
"""
import pandas as pd

ENGAGE = ["play_frequency", "avg_session_duration", "total_playtime_hours", "login_streak", "friend_count",
          "chat_activity_score", "friend_invites_sent", "gifts_sent_received", "tournament_entries",
          "speed_of_progression", "peak_concurrent_hours", "team_play_percentage", "win_rate_ranked"]
ENGAGE_NEG = ["days_since_last_login"]  # lower = more engaged
SPEND = ["total_spending_thb", "avg_monthly_spending", "spending_frequency", "rare_items_count",
         "item_type_preference_cosmetic", "achievement_completion_rate", "collection_progress"]
ORD = {"loyalty_tier": ["Bronze", "Silver", "Gold", "Platinum", "Diamond"],
       "engagement_level": ["Low", "Mid", "High", "Veteran"]}


def add_composites(fit_src: pd.DataFrame, frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    cols = ENGAGE + ENGAGE_NEG + SPEND
    mu, sd = fit_src[cols].mean(), fit_src[cols].std()
    out = []
    for X in frames:
        X = X.copy()
        Z = (X[cols] - mu) / sd
        Z[ENGAGE_NEG] = -Z[ENGAGE_NEG]
        X["eng_score"] = Z[ENGAGE + ENGAGE_NEG].mean(axis=1)
        X["spend_score"] = Z[SPEND].mean(axis=1)
        X["eng_minus_spend"] = X["eng_score"] - X["spend_score"]
        X["eng_plus_spend"] = X["eng_score"] + X["spend_score"]
        for c, levels in ORD.items():
            X[f"{c}_ord"] = X[c].map({l: i for i, l in enumerate(levels)}).astype(float)
        X["spend_per_hour"] = X["total_spending_thb"] / (X["total_playtime_hours"] + 1)
        X["rare_per_spend"] = X["rare_items_count"] / (X["total_spending_thb"] + 1)
        out.append(X)
    return out
