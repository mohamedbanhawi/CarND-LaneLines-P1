# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

## PID Parameters Tuning

I relied on manual tuning for the PID gains. I started by tuning P, followed by D gain and finally the I term.

- The effect of a high P value was that the vehicle was unstable and was to sensitive to any offsets. The overall effect of using only P is that the vehicle response was either damped and never converged (low P) or very sensitive.
- Next was the D term, this has the effect of stabilizing the vehicle and avoiding osciliastions in the response.
- I found that the I-term had no effect on the performance, this was obviously a qualititve assessment. In general, the I term is supposed to remove any offsets (i.e. bias) in the system. 

## Final Parameters

| Term  | Value   | 
|---|---|
| P  |  0.14 | 
| I  |  0.0001 |
| D  |  2.5 |

