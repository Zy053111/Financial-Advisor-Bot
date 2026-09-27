# Financial Advisor Bot (FAB)

An institutional-grade, Explainable Deep Reinforcement Learning (DRL) Portfolio Management System for the Magnificent 7 equities basket. The platform integrates Proximal Policy Optimization (PPO), out-of-sample backtesting with market friction models, and local LLM-driven Explainable AI (XAI).

---

## Architecture Overview

* **DRL Engine**: PPO (Continuous Actor-Critic via Stable-Baselines3) managing continuous portfolio weights on the simplex.
* **Feature Engineering**: Multi-asset Log Returns, 14-day RSI, and MACD (26, 12, 9).
* **Backtesting**: Backtrader simulation engine featuring 0.10% commission fees, 0.05% execution slippage, and rolling Markowitz MPT / 1/N equal-weight benchmarks.
* **XAI Dialogue**: Local Llama-3 via Ollama generating persona-aligned strategy narratives and telemetry-grounded Q&A.
* **Frontend**: React, Tailwind CSS, Lucide Icons, and Recharts interactive visualizations.

---

## Prerequisites

Ensure you have the following installed on your system:
* Python 3.10+
* Node.js 18+ & npm
* Ollama (for local Llama-3 inference)

---

## Installation & Environment Setup

### 1. Local LLM Setup (Ollama)
Ensure the Ollama daemon is active and download the Llama-3 model:

    ollama run llama3

Keep this service running in the background at http://localhost:11434.

### 2. Backend Setup (Python / Django)
Open a new terminal, navigate to the backend directory, and activate your virtual environment:

    cd backend
    python -m venv venv

    # Windows (Command Prompt / PowerShell)
    venv\Scripts\activate

    # macOS / Linux
    source venv/bin/activate

    # Install required Python dependencies
    pip install torch torchvision stable-baselines3 gymnasium backtrader yfinance pandas numpy scipy matplotlib requests django djangorestframework django-cors-headers

### 3. Frontend Setup (React)
Open a separate terminal and configure the frontend dependencies:

    cd frontend
    npm install

---

## Running Model Training & Experimental Artifacts

Note: Pre-trained weights and pre-computed backtest histories are saved in backend/saved_models/ and backend/backtest_history.json. You only need to run these if retraining or reproducing thesis figures.

* Train Primary PPO Agent (In-Sample: 2020-01-01 to 2024-12-31):

    python backend/train_ppo.py

* Run Out-of-Sample Backtest (OOS: 2025-01-01 to 2026-08-30):
  Generates backtest_result.png (DRL vs. Markowitz vs. 1/N) and updates backtest_history.json.

    python backend/backtest.py

* Execute Thesis Ablation Study:
  Trains the ablated baseline (Returns-only) and exports ablation_result.png:

    python backend/ablation_study.py

---

## Starting the Application (Demo & Presentation)

Run the backend and frontend development servers concurrently:

### Terminal 1: Backend API

    cd backend
    venv\Scripts\activate  # Windows
    python manage.py runserver

The Django REST API runs on http://127.0.0.1:8000.

### Terminal 2: Frontend Dashboard

    cd frontend
    npm run dev

Open your browser and navigate to the local Vite URL (typically http://localhost:5173) to interact with the dashboard.

---

## Core Features Walkthrough

1. Mandate Selection: Toggle between Conservative, Moderate, and Aggressive profiles to observe dynamic weight transformations in the donut chart and table.
2. Execution Order Book: Adjust portfolio capital to inspect buy/sell deltas and estimated shares calculated against the equal-weighted 1/N baseline.
3. Out-of-Sample Trajectory: Inspect interactive equity curves comparing the PPO Agent, rolling Markowitz MPT, and the 1/N Benchmark under friction models.
4. XAI Dialogue & Suggestion Chips: Click pre-configured prompt chips or submit queries to chat with the Llama-3 institutional advisor grounded in live telemetry.
