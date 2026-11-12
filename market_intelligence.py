"""Transparent inference-time market context and abstention layer."""
from dataclasses import dataclass,asdict
import math
@dataclass
class Decision:
    action:str
    confidence:float
    probability_up:float
    expected_return:float
    estimated_round_trip_cost:float
    edge_after_cost:float
    regime:str
    reasons:list
def classify_regime(volatility,trend_strength):
    if volatility>=.80:return "high_volatility"
    if volatility<=.20 and abs(trend_strength)<.25:return "compression"
    if trend_strength>=.50:return "uptrend"
    if trend_strength<=-.50:return "downtrend"
    return "range"
def decide(prob_up,q10,q50,q90,volatility,trend_strength,round_trip_cost,min_edge=.00015,min_confidence=.58):
    vals=[prob_up,q10,q50,q90,volatility,trend_strength,round_trip_cost]
    if not all(math.isfinite(float(x)) for x in vals):raise ValueError("Inputs must be finite")
    regime=classify_regime(volatility,trend_strength)
    expected=(prob_up*max(q50,0)+(1-prob_up)*min(q50,0))
    long_edge=q10-round_trip_cost; short_edge=-q90-round_trip_cost
    confidence=max(prob_up,1-prob_up); reasons=[f"regime={regime}",f"p_up={prob_up:.3f}",f"round_trip_cost={round_trip_cost:.6f}"]
    if confidence<min_confidence:action="WAIT";edge=max(long_edge,short_edge);reasons.append("confidence below threshold")
    elif prob_up>=.5 and long_edge>=min_edge:action="BUY";edge=long_edge;reasons.append("bullish conservative quantile clears costs")
    elif prob_up<.5 and short_edge>=min_edge:action="SELL";edge=short_edge;reasons.append("bearish conservative quantile clears costs")
    else:action="WAIT";edge=max(long_edge,short_edge);reasons.append("no conservative edge after costs")
    if regime=="high_volatility" and action!="WAIT":reasons.append("high volatility; abstain");action="WAIT"
    return asdict(Decision(action,confidence,prob_up,expected,round_trip_cost,edge,regime,reasons))
