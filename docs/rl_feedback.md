# Reinforcement Learning Feedback Loop

This document explains how the DQN-driven correction layer integrates with the multimodal sentiment analyzer.

## Architecture Overview

1. **Base inference** – Text + image analyzers generate probabilities and compound scores.
2. **State vector** – The backend concatenates:
   - Combined negative/neutral/positive probabilities
   - Text-only probabilities (or zeros)
   - Image-only probabilities (or zeros)
   - Compound score, normalized text length, has-image flag
3. **Correction layer** – A lightweight linear module maps the state to delta logits.
4. **DQN agent** – The state is also passed into a DQN that chooses one of four actions: keep, force positive, force neutral, force negative. Its replay buffer is updated whenever feedback arrives.
5. **Adjusted output** – The correction layer logits + DQN action bias re-weight the final probabilities and update the returned sentiment.
6. **Feedback** – The UI posts `/api/feedback` with `{ state, action, correct }`. Reward = `+1` on 👍 and `-1` on 👎.
7. **Training** – Each feedback call pushes a transition into the replay buffer, triggers a short optimization step on the policy net, nudges the correction layer, and persists checkpoints.

## Key Files

- `backend/app/services/correction.py` – DQN agent, replay buffer, correction layer helpers.
- `backend/app/main.py`
  - Builds state vectors inside `/api/analyze`
  - Adjusts predictions via correction layer + DQN action
  - Exposes `/api/feedback`
- `backend/app/services/database.py`
  - Adds `feedback` table + `save_feedback`
  - Dashboard stats now include `feedback_curve`
- `frontend/src/components/SentimentAnalyzer.js`
  - Adds 👍/👎 buttons and learning indicator
  - Posts feedback payloads to the backend
- `frontend/src/components/Dashboard.js`
  - Displays the RL learning curve via Recharts

## Persistence

- Checkpoints are stored under `backend/checkpoints/` (`dqn_policy.pt`, `dqn_target.pt`, `correction_layer.pt`).
- Feedback entries (state, action, reward) live in the SQLite database and surface through the dashboard.

## Workflow Summary

1. User submits text/image → `/api/analyze` responds with corrected probabilities and RL metadata.
2. UI renders prediction plus feedback buttons.
3. User clicks 👍/👎 → `/api/feedback` receives the payload, updates the replay buffer, tweaks weights, and records the reward curve.
4. Dashboard shows cumulative RL signal so you can monitor improvement over time.
