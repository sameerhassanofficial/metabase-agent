import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List, Optional

def get_numeric_summaries(df: pd.DataFrame) -> Dict[str, Any]:
    """Return mean, median, std, min, max for all numeric columns."""
    summaries = {}
    for col in df.select_dtypes(include='number').columns:
        arr = df[col].dropna().values
        if arr.size > 0:
            summaries[col] = {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'sum': float(np.sum(arr)),
            }
    return summaries

def predict_trend(df: pd.DataFrame, x_col: str, y_col: str, periods_ahead: int = 1) -> Optional[float]:
    """
    Fit a linear regression to predict y_col from x_col and forecast periods_ahead into the future.
    x_col should be numeric or datetime (will be converted to ordinal).
    Returns the predicted y value for the next period.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return None
    x = df[x_col]
    y = df[y_col]
    if np.issubdtype(x.dtype, np.datetime64):
        x = x.map(pd.Timestamp.toordinal)
    X = np.array(x).reshape(-1, 1)
    y = np.array(y)
    if len(X) < 2:
        return None
    model = LinearRegression()
    model.fit(X, y)
    next_x = X[-1][0] + periods_ahead
    y_pred = model.predict(np.array([[next_x]]))
    return float(y_pred[0])

def find_relevant_cards_sklearn(question: str, all_cards: List[Dict], top_n: int = 5) -> List[str]:
    """
    Use TF-IDF vectorization and cosine similarity to find the most relevant cards for a question.
    Returns a list of card names.
    """
    card_texts = []
    card_names = []
    for c in all_cards:
        card_data = c.get("card", {})
        if c.get("card_id"):
            text = (card_data.get("name") or "") + " " + (card_data.get("description") or "")
            text += " " + " ".join(col["name"] for col in card_data.get("schema", []))
            card_texts.append(text)
            card_names.append(card_data.get("name"))
    if not card_texts:
        return []
    vectorizer = TfidfVectorizer().fit(card_texts + [question])
    card_vecs = vectorizer.transform(card_texts)
    question_vec = vectorizer.transform([question])
    sims = cosine_similarity(question_vec, card_vecs)[0]
    top_idx = np.argsort(sims)[::-1][:top_n]
    return [card_names[i] for i in top_idx if sims[i] > 0] 